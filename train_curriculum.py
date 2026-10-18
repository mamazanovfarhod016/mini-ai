"""
Мультистадийное обучение (curriculum) Mini AI: read → reason → context → self-play.

    .venv/bin/python train_curriculum.py --plan docs/plan_tonight.json --out /kaggle/working/out \
        --init model_transformer_school.json

План — JSON со списком стадий (см. default_plan() ниже за образец). Каждая стадия:
    name, steps, lr, warmup, max_len, token_budget, sources:[{type, ..., weight}]
Движок — TorchEngine (GPU если есть). Чекпойнты каждые ckpt_minutes минут + в конце стадии
(export_model JSON) в out/ckpt/. Resume: если out/ckpt/state.json существует — продолжаем с
того (stage_idx, step), источники для уже пройденных стадий просто не выполняются повторно.

Экзамены после каждой стадии (и финальный) → out/exams.json (список записей).
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from mini_ai.curriculum import (
    build_dynamic_batch, load_lm_lines, make_pool_from_source, pack_lm_windows, pad_batch,
    save_checkpoint, load_latest_checkpoint, setup_model,
)
from mini_ai.exams import run_exams
from mini_ai.transformer import lr_schedule, n_params
from mini_ai.torch_engine import TorchEngine
from mini_ai.selfplay_stage import run_selfplay

ROOT = Path(__file__).resolve().parent


# ======================================================================================
# План по умолчанию для сегодняшней ночи (Kaggle T4, ~8ч вкл. экзамены). Шаги — ЗАГЛУШКА,
# реальное число шагов выставляется throughput_probe() в начале запуска (log'ируется).
# ======================================================================================
def default_plan():
    corpus = ROOT / "data/corpus"
    return {
        "init_model": str(ROOT / "model_transformer_school.json"),
        "target_vocab": 5000,
        "arch": {"d": 192, "n_layer": 20, "n_head": 8, "block": 10000, "pos": "rope"},
        "time_budget_hours": 8.0,
        "stages": [
            {
                "name": "read",
                "max_len": 512, "token_budget": 16384, "lr": 2e-3, "warmup": 300,
                "time_frac": 0.30,
                "sources": [
                    {"type": "lm", "weight": 0.85,
                     "files": [str(corpus / "uz.txt"), str(corpus / "ru.txt"), str(corpus / "en.txt")]},
                    {"type": "qa", "weight": 0.15, "file": str(ROOT / "data/intents_pairs.json")},
                ],
            },
            {
                "name": "reason",
                "max_len": 400, "token_budget": 12288, "lr": 1.5e-3, "warmup": 200,
                "time_frac": 0.35,
                "sources": [
                    {"type": "qa", "weight": 0.22, "file": str(ROOT / "data/reasoning.json")},
                    {"type": "qa", "weight": 0.18, "file": str(ROOT / "data/school_math.json")},
                    {"type": "qa", "weight": 0.12, "file": str(ROOT / "data/math_pairs.json")},
                    {"type": "qa", "weight": 0.13, "file": str(ROOT / "data/parallel.json")},
                    {"type": "qa", "weight": 0.10, "file": str(ROOT / "data/transfer.json")},
                    {"type": "qa", "weight": 0.05, "file": str(ROOT / "data/intents_pairs.json")},
                    {"type": "qa", "weight": 0.10, "file": str(ROOT / "data/python_tasks.json"), "raw": True},
                    {"type": "lm", "weight": 0.10,
                     "files": [str(corpus / "uz.txt"), str(corpus / "ru.txt"), str(corpus / "en.txt")]},
                ],
            },
            {
                "name": "context",
                "max_len": 10000, "token_budget": 8192, "lr": 1e-3, "warmup": 150,
                "time_frac": 0.25,
                "sources": [
                    {"type": "context", "weight": 0.75, "file": str(ROOT / "data/context_tasks.json")},
                    {"type": "qa", "weight": 0.15, "file": str(ROOT / "data/reasoning.json")},
                    {"type": "lm", "weight": 0.10,
                     "files": [str(corpus / "uz.txt"), str(corpus / "ru.txt"), str(corpus / "en.txt")]},
                ],
            },
            {
                "name": "selfplay",
                "max_len": 400, "token_budget": 8192, "lr": 5e-4, "warmup": 50,
                "time_frac": 0.10,
                "selfplay": {
                    "task_files": [str(ROOT / "data/reasoning.json"), str(ROOT / "data/school_math.json")],
                    "n": 400, "tries": 2, "temp": 0.7,
                },
                "sources": [
                    {"type": "selfplay", "weight": 0.6, "file": "__selfplay_out__"},
                    {"type": "qa", "weight": 0.2, "file": str(ROOT / "data/reasoning.json")},
                    {"type": "qa", "weight": 0.2, "file": str(ROOT / "data/intents_pairs.json")},
                ],
            },
        ],
    }


# ======================================================================================
# Throughput probe: 60с пробного обучения на первой стадии, чтобы оценить шаги/бюджет времени
# ======================================================================================
def throughput_probe(engine, pools, weights, max_len, token_budget, lr, seconds=60, log=print):
    rng = np.random.default_rng(0)
    t0 = time.time()
    n_steps = 0
    while time.time() - t0 < seconds:
        batch = build_dynamic_batch(pools, weights, token_budget, max_len, rng)
        idx, tgt, mask = pad_batch(*zip(*batch))
        engine.step(idx, tgt, mask, lr)
        n_steps += 1
    elapsed = time.time() - t0
    steps_per_sec = n_steps / max(elapsed, 1e-6)
    log(f"Throughput probe: {n_steps} шагов за {elapsed:.1f}с -> {steps_per_sec:.3f} шаг/с")
    return steps_per_sec


def exam_data_for_stage(stage_name, root):
    """Собрать exam_data (пути/данные теста) для run_exams — используем *_test.txt и тесты
    генераторов, если они существуют локально (на Kaggle всё это будет сгенерировано перед
    запуском train_curriculum.py)."""
    corpus = root / "data/corpus"
    out = {"lm_test_files": {}}
    for lang in ("uz", "ru", "en"):
        f = corpus / f"{lang}_test.txt"
        if f.exists():
            out["lm_test_files"][lang] = [str(f)]
    def load_test(path, key="test"):
        p = root / path
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8")).get(key, [])
            except Exception:
                return []
        return []
    def sample(items, n):            # экзамен на фиксированной случайной выборке — быстро и одинаково для всех стадий
        return random.Random(0).sample(items, min(n, len(items))) if items else []
    out["school_math_test"] = sample(load_test("data/school_math.json"), 1500)
    out["reasoning_test"] = sample(load_test("data/reasoning.json"), 600)
    out["transfer_test"] = load_test("data/transfer.json")
    out["intents_test"] = load_test("data/intents_pairs.json")
    py = [x for x in load_test("data/python_tasks.json") if x.get("kind") != "what_prints"]
    out["python_tasks_test"] = random.Random(0).sample(py, min(300, len(py))) if py else []
    if stage_name in ("context", "selfplay"):
        ctx = root / "data/context_tasks.json"
        if ctx.exists():
            from tools.make_context_tasks import load as load_ctx
            try:
                d = load_ctx(ctx)
                ct = d.get("test", [])
                # по 40 на каждый вид и длину (needle s/m/l/xl ≈ до 10k токенов) — честно и не слишком долго
                by = {}
                for it in ct:
                    by.setdefault((it.get("kind"), it.get("bucket"), it.get("lang")), []).append(it)
                out["context_test"] = [x for k in sorted(by, key=str) for x in random.Random(0).sample(by[k], min(14, len(by[k])))]
            except Exception:
                out["context_test"] = []
    return out


def run_stage(engine, tok, stage, rng, ckpt_dir, stage_idx, start_step, ckpt_minutes, log=print,
              steps_total=200, raw_default=False):
    max_len = stage["max_len"]
    token_budget = stage["token_budget"]
    lr, warmup = stage["lr"], stage["warmup"]

    if stage.get("type") == "selfplay" or "selfplay" in stage:
        sp = stage["selfplay"]
        from mini_ai.torch_engine import TorchEngine  # noqa
        p_np = engine.params_numpy()
        out_path = Path(ckpt_dir) / "selfplay_data.json"
        run_selfplay(engine, tok, sp["task_files"], out_path, n=sp.get("n", 400),
                     tries=sp.get("tries", 2), temp=sp.get("temp", 0.7), log=log)
        for src in stage["sources"]:
            if src.get("file") == "__selfplay_out__":
                src["file"] = str(out_path)

    pools = {}
    weights = {}
    for i, src in enumerate(stage["sources"]):
        name = f"{src['type']}_{i}"
        pools[name] = make_pool_from_source(tok, src, max_len, rng, raw_default)
        weights[name] = src.get("weight", 1.0)
        log(f"  источник {name}: {len(pools[name].examples)} примеров, вес={weights[name]}")

    t_last_ckpt = time.time()
    last_loss = None
    for step in range(start_step, steps_total):
        batch = build_dynamic_batch(pools, weights, token_budget, max_len, rng)
        if not batch:
            break
        idx, tgt, mask = pad_batch(*zip(*batch))
        lr_t = lr_schedule(step, steps_total, lr, warmup)
        loss, gnorm = engine.step(idx, tgt, mask, lr_t)
        last_loss = loss
        if step % max(1, steps_total // 20) == 0:
            log(f"  [{stage['name']}] шаг {step}/{steps_total} loss={loss:.3f} lr={lr_t:.5f} "
                f"||g||={gnorm:.2f} batch={len(batch)} T={idx.shape[1]}")
        if (time.time() - t_last_ckpt) / 60 >= ckpt_minutes:
            p_np = engine.params_numpy()
            save_checkpoint(ckpt_dir, p_np, engine.cfg, tok, {"max_q": 40}, stage_idx, step,
                             "bpe" if hasattr(tok, "merges") else "char")
            t_last_ckpt = time.time()
    # чекпойнт в конце стадии
    p_np = engine.params_numpy()
    save_checkpoint(ckpt_dir, p_np, engine.cfg, tok, {"max_q": 40}, stage_idx, steps_total - 1,
                     "bpe" if hasattr(tok, "merges") else "char")
    return last_loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default="", help="JSON-файл с планом; пусто = default_plan()")
    ap.add_argument("--init", default=str(ROOT / "model_transformer_school.json"))
    ap.add_argument("--out", default="/tmp/mini-ai-curriculum-out")
    ap.add_argument("--ckpt_minutes", type=float, default=15.0)
    ap.add_argument("--probe_seconds", type=int, default=60)
    ap.add_argument("--steps_total_override", type=int, default=0,
                     help="для тестов/smoke: фиксированное число шагов на стадию вместо probe")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "ckpt"
    log_path = out_dir / "log.txt"

    def log(msg):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")

    plan = json.loads(Path(a.plan).read_text(encoding="utf-8")) if a.plan else default_plan()
    rng = np.random.default_rng(a.seed)

    resume = load_latest_checkpoint(ckpt_dir)
    if resume:
        log(f"Resume: стадия {resume['stage_idx']}, шаг {resume['step']}")
        p, cfg, tok = resume["p"], resume["cfg"], resume["tok"]
        tokenizer_type = resume["meta"].get("tokenizer", "bpe")
        max_q = resume["meta"].get("max_q", 40)
        start_stage_idx = resume["stage_idx"]
        start_step = resume["step"] + 1
    else:
        train_texts = []
        for stage in plan["stages"]:
            for src in stage["sources"]:
                if src["type"] == "lm":
                    train_texts.extend(load_lm_lines(src["files"])[:2000])
                elif src["type"] in ("qa", "context") and Path(src["file"]).exists():
                    try:
                        data = json.loads(Path(src["file"]).read_text(encoding="utf-8"))
                        for it in data.get("train", [])[:500]:
                            train_texts.append(str(it.get("q", "")))
                            train_texts.append(str(it.get("a", "")))
                    except Exception:
                        pass
        arch = plan["arch"]
        p, cfg, tok, tokenizer_type, max_q = setup_model(
            a.init, train_texts, target_vocab=plan.get("target_vocab", 5000),
            d=arch["d"], n_layer=arch["n_layer"], n_head=arch["n_head"],
            block=arch["block"], pos=arch["pos"], log=log)
        start_stage_idx = 0
        start_step = 0

    engine = TorchEngine({k: np.asarray(v, dtype=np.float32) for k, v in p.items()}, cfg,
                         lr=1e-3, wd=0.01, seed=a.seed)
    log(f"Устройство: {engine.dev}")

    exams_all = []
    exams_path = out_dir / "exams.json"
    if exams_path.exists():
        try:
            exams_all = json.loads(exams_path.read_text(encoding="utf-8"))
        except Exception:
            exams_all = []

    total_budget_s = plan.get("time_budget_hours", 8.0) * 3600
    t_run_start = time.time()
    for stage_idx, stage in enumerate(plan["stages"]):
        if stage_idx < start_stage_idx:
            continue
        log(f"=== Стадия {stage_idx}: {stage['name']} ===")
        step0 = start_step if stage_idx == start_stage_idx else 0

        if a.steps_total_override:
            steps_total = a.steps_total_override
        else:
            # throughput probe на первой запускаемой стадии; дальше масштабируем по time_frac
            probe_pools = {f"src_{i}": make_pool_from_source(tok, s, stage["max_len"], rng)
                           for i, s in enumerate(stage["sources"]) if s["type"] != "selfplay"}
            probe_weights = {f"src_{i}": s.get("weight", 1.0) for i, s in enumerate(stage["sources"])
                             if s["type"] != "selfplay"}
            if probe_pools and any(p_.examples for p_ in probe_pools.values()):
                sps = throughput_probe(engine, probe_pools, probe_weights, stage["max_len"],
                                        stage["token_budget"], stage["lr"], a.probe_seconds, log)
                stage_time_s = total_budget_s * stage.get("time_frac", 0.25)
                steps_total = max(10, int(sps * stage_time_s))
            else:
                steps_total = 50
            log(f"  шагов для стадии '{stage['name']}': {steps_total}")

        run_stage(engine, tok, stage, rng, ckpt_dir, stage_idx, step0, a.ckpt_minutes, log,
                  steps_total=steps_total)
        start_step = 0

        exam_data = exam_data_for_stage(stage["name"], ROOT)
        log(f"Экзамен после стадии '{stage['name']}'...")
        res = run_exams(engine, tok, tokenizer_type, max_q, exam_data, log)
        exams_all.append({"stage": stage["name"], "stage_idx": stage_idx, "results": res,
                           "elapsed_s": time.time() - t_run_start})
        exams_path.write_text(json.dumps(exams_all, ensure_ascii=False, indent=1), encoding="utf-8")

    # финальная модель
    p_final = engine.params_numpy()
    from mini_ai.transformer import export_model
    final_doc = export_model(p_final, engine.cfg, tok, {"max_q": max_q}, tokenizer_type=tokenizer_type,
                              max_q=max_q, merges=getattr(tok, "merges", None))
    (out_dir / "model_final.json").write_text(json.dumps(final_doc, ensure_ascii=False), encoding="utf-8")
    log(f"Готово. Параметров: {n_params(p_final):,}. Модель: {out_dir / 'model_final.json'}")


if __name__ == "__main__":
    main()
