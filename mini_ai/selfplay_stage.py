"""
Стадия 'selfplay': модель решает свои train-задания (reasoning + school word problems) батчами
на torch, скрипт проверяет, верные решения сохраняются как qa-источник для следующей стадии.
Никогда не использует test-примеры (только item из train части файлов заданий).
"""
import json
import random
from pathlib import Path

from mini_ai.batched_generate import batched_generate
from mini_ai.curriculum import encode_prompt
from mini_ai.transformer import BOS, SEP


def run_selfplay(engine, tok, task_files, out_path, max_q=40, n=400, tries=2, temp=0.7,
                  batch=16, max_new=200, seed=0, log=print):
    """task_files: список путей к json {train:[...], test:[...]} — ТОЛЬКО train используется.
    Возвращает путь к out_path с {"train": [...]}; weak kinds (по fail_rate) получают больше попыток
    в следующем вызове — здесь просто пишем fail_rate в out_path.stats.json."""
    from server.selfplay import checker, final_answer  # noqa: F401 (checker используется)
    check = checker()
    rng = random.Random(seed)
    pool = []
    for f in task_files:
        p = Path(f)
        if not p.exists():
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        pool.extend(data.get("train", []))
    picked = rng.sample(pool, min(n, len(pool))) if pool else []
    good = []
    tried, solved = {}, {}
    for try_i in range(tries):
        temp_i = 0.0 if try_i == 0 else temp
        remaining = [it for it in picked if id(it) not in {id(g["_src"]) for g in good}]
        for s in range(0, len(remaining), batch):
            chunk = remaining[s:s + batch]
            prompts = [[BOS] + encode_prompt(tok, it["q"], False)[:max_q] + [SEP] for it in chunk]
            outs = batched_generate(engine, prompts, max_new, temperature=temp_i, seed=seed + try_i)
            for it, out in zip(chunk, outs):
                kind = it.get("kind", "default")
                tried[kind] = tried.get(kind, 0) + (1 if try_i == 0 else 0)
                text = tok.decode(out)
                r = check(it, text)
                if r.get("passed"):
                    good.append({"q": it["q"], "a": text, "kind": kind, "lang": it.get("lang"),
                                 "source": "selfplay", "_src": it})
                    solved[kind] = solved.get(kind, 0) + 1
    fail_rate = {k: round(1 - solved.get(k, 0) / max(1, tried.get(k, 1)), 3) for k in tried}
    out = {"train": [{k: v for k, v in g.items() if k != "_src"} for g in good]}
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    Path(out_path).with_suffix(".stats.json").write_text(
        json.dumps({"fail_rate": fail_rate, "n_good": len(good), "n_tried": len(picked)}), encoding="utf-8")
    log(f"  selfplay: {len(good)}/{len(picked)} верных решений, слабые виды: "
        f"{sorted(fail_rate.items(), key=lambda x: -x[1])[:3]}")
    return out_path
