"""
Экзамены по ТИПАМ МЫШЛЕНИЯ (план, неделя 1: docs/PLAN_3_MONTHS.md). Только скрипт, без ИИ-судьи —
каждый тип проверяется готовым чек-модулем (server/reasoncheck.py, mathcheck.py, contextcheck.py,
tools/make_transfer.py:score, train_transformer.py:intent_of_answer).

    .venv/bin/python -m server.exams_thinking MODEL.json [--label "точка отсчёта"] [--n 300]

Какой тип мышления чем проверяется сейчас (19.10.2026, план docs/PLAN_3_MONTHS.md §«Типы мышления»):
  Аналитическое            — data/intents_pairs.json (понять намерение вопроса)
  Логическое/математическое— data/reasoning.json (шаги + проверка) + data/school_math.json (точный ответ)
  Абстрактное (перенос)    — data/transfer.json (факт на одном языке → вопрос на двух других)
  Критическое, Стратегическое, Креативное, Понимание человека, «Думает сама» — ДАННЫХ ЕЩЁ НЕТ
      (prm800k/критическое, многошаговый GSM8K/стратегическое, TinyStories-творчество, диалоги —
      по плану позже; честно помечаем «нет данных», а не подделываем цифру).
Результат → docs/EXAMS_THINKING.md (история, «было → стало») + server_data/exams_thinking.json.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

N_DEFAULT = 300
NO_DATA = "нет данных (план: позже)"


def _load_test(path, key="test"):
    p = ROOT / path
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8")).get(key, [])
    except Exception:
        return []


def _sample(items, n, seed=0):
    return random.Random(seed).sample(items, min(n, len(items))) if items else []


def build_exam_data(n=N_DEFAULT):
    """Собрать выборки для run_exams (mini_ai/exams.py) — те же файлы, что в train_curriculum.py,
    но целиком независимо (можно звать без обучения, на готовой модели, на CPU)."""
    out = {}
    out["intents_test"] = _sample(_load_test("data/intents_pairs.json"), n)
    out["reasoning_test"] = _sample(_load_test("data/reasoning.json"), n)
    out["school_math_test"] = _sample(_load_test("data/school_math.json"), n)
    # data/transfer.json: test ~1300 строк, но exam_transfer без KV-кэша на CPU — дорого; берём выборку
    out["transfer_test"] = _sample(_load_test("data/transfer.json"), min(n, 120))
    py = [x for x in _load_test("data/python_tasks.json") if x.get("kind") != "what_prints"]
    out["python_tasks_test"] = _sample(py, min(n, 100))
    return out


def _engine_for(model_path):
    """CPU-движок (TorchEngine без обучения — нужен только forward для batched_generate/run_exams)."""
    import numpy as np
    from mini_ai.torch_engine import TorchEngine
    from mini_ai.transformer import load_model
    p, cfg, tok, meta = load_model(model_path)
    engine = TorchEngine({k: np.asarray(v, dtype=np.float32) for k, v in p.items()}, cfg, seed=0)
    return engine, tok, meta


def run(model_path, label="", n=N_DEFAULT, log=print):
    from mini_ai.exams import run_exams
    t0 = time.time()
    engine, tok, meta = _engine_for(model_path)
    tokenizer_type = "bpe" if hasattr(tok, "merges") else "char"
    max_q = meta.get("max_q", 24)
    exam_data = build_exam_data(n)
    res = run_exams(engine, tok, tokenizer_type, max_q, exam_data, log=log)

    thinking = {
        "Аналитическое": res.get("intents", {}).get("acc"),
        "Логическое/математическое": None,
        "Абстрактное (перенос)": None,
        "Критическое": NO_DATA,
        "Стратегическое": NO_DATA,
        "Креативное": NO_DATA,
        "Понимание человека": NO_DATA,
        "«Думает сама» (R1)": NO_DATA,
    }
    r_acc = res.get("reasoning", {}).get("answer_ok")
    m_acc = res.get("school_math", {}).get("acc")
    parts = [v for v in (r_acc, m_acc) if v is not None]
    if parts:
        thinking["Логическое/математическое"] = sum(parts) / len(parts)
    tr = res.get("transfer", {})
    if tr.get("transfer") is not None:
        thinking["Абстрактное (перенос)"] = tr["transfer"]

    import numpy as np
    n_params = int(sum(v.size for v in engine.params_numpy().values()))
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M"), "label": label, "model": str(model_path),
           "params": n_params, "minutes": round((time.time() - t0) / 60, 1),
           "raw": res, "thinking": thinking}
    hist_path = ROOT / "server_data/exams_thinking.json"
    hist = json.loads(hist_path.read_text()) if hist_path.exists() else []
    hist.append(rec)
    hist_path.write_text(json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")
    _write_md(hist)
    return rec


def _fmt(v):
    if v is None:
        return "—"
    if isinstance(v, str):
        return v
    return f"{v * 100:.1f}%"


def _write_md(hist):
    lines = ["# Экзамены по типам мышления Mini AI", "",
             "Скриптовые проверки (без ИИ-судьи), план: docs/PLAN_3_MONTHS.md §«Типы мышления». "
             "«нет данных» — тип мышления будет добавлен позже по плану (критическое/стратегическое/"
             "креативное/понимание человека/R1).", "",
             "| Когда | Этап | Параметров | Аналитическое | Логическое/математическое | "
             "Абстрактное (перенос) | Критическое | Стратегическое | Креативное | "
             "Понимание человека | «Думает сама» |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in hist:
        t = r["thinking"]
        lines.append(f"| {r['ts']} | {r['label']} | {r['params'] / 1e6:.1f} млн | "
                     f"{_fmt(t['Аналитическое'])} | {_fmt(t['Логическое/математическое'])} | "
                     f"{_fmt(t['Абстрактное (перенос)'])} | {_fmt(t['Критическое'])} | "
                     f"{_fmt(t['Стратегическое'])} | {_fmt(t['Креативное'])} | "
                     f"{_fmt(t['Понимание человека'])} | {_fmt(t['«Думает сама» (R1)'])} |")
    (ROOT / "docs/EXAMS_THINKING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--label", default="")
    ap.add_argument("--n", type=int, default=N_DEFAULT)
    a = ap.parse_args()
    rec = run(a.model, a.label, a.n)
    for k, v in rec["thinking"].items():
        print(f"{k:30s} {_fmt(v)}")


if __name__ == "__main__":
    main()
