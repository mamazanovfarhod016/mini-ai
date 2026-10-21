"""reason_bench.py — честный счётчик РАССУЖДЕНИЯ ядра: все 20 задач bAbI (Weston et al. 2015), по 1000 вопросов, без обучения.
Каждая задача = отдельное умение: 2–3 факта подряд, местоимения, союзы, счёт, списки, время, дедукция/индукция по типу,
позиция/размер, путь, мотивы. Ядро — rules.World (правила вывода), а не веса.
Печатает таблицу и пишет ~/Developer/mini-ai-runs/learning-core/reason_<дата>.json + reason_latest.json.
Запуск: python3 tools/reason_bench.py [task ...] [miss] [n=200]   (без задач — все 20; miss — печатать промахи; n — вопросов на задачу)"""
import os, sys, json, time, glob, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from mini_ai.rules import World
OUT = os.path.expanduser("~/Developer/mini-ai-runs/learning-core"); DATA = os.path.expanduser("~/Developer/mini-ai/data/babi")
ALL = sorted([re.search(r"task_(.+)_test", p).group(1) for p in glob.glob(f"{DATA}/task_*_test.parquet")], key=lambda t: int(t.split("-")[0]))
show_miss = "miss" in sys.argv; N = next((int(a[2:]) for a in sys.argv[1:] if a.startswith("n=")), 1000)
args = [a for a in sys.argv[1:] if a != "miss" and not a.startswith("n=")]
tasks = [t for t in ALL if not args or any(t.split("-")[0] == a for a in args)]

def norm(x): return set(str(x).lower().replace(",", " ").split())
res = {"when": time.strftime("%Y-%m-%d %H:%M"), "tasks": {}}; t0 = time.time()
for t in tasks:
    df = pd.read_parquet(f"{DATA}/task_{t}_test.parquet").head(N); ok = 0; miss = []
    for _, r in df.iterrows():
        W = World(); [W.read(s) for s in r["passage"].split("\n") if s.strip()]
        try: a, _ = W.answer(r["question"])
        except Exception as e: a = f"ERR {e}"
        g = r["answer"].lower(); hit = a is not None and (norm(a) == norm(g) or g in str(a).lower())
        ok += hit
        if not hit and len(miss) < 4: miss.append((r["passage"][-220:].replace("\n", " | "), r["question"], g, str(a)))
    res["tasks"][t] = round(ok / len(df), 3); print(f"{t:32s} {res['tasks'][t]:.3f}", flush=True)
    if show_miss:
        for p, q, g, a in miss: print(f"      ✗ {q} → {a!r} (ждали {g!r})\n        …{p}")
res["mean"] = round(sum(res["tasks"].values()) / len(res["tasks"]), 3); res["seconds"] = round(time.time() - t0)
print(f"MEAN {res['mean']}   ({len(tasks)} задач, {res['seconds']}s)")
if not args and N >= 1000:
    json.dump(res, open(os.path.join(OUT, f"reason_{time.strftime('%Y%m%d_%H%M')}.json"), "w"), indent=1)
    json.dump(res, open(os.path.join(OUT, "reason_latest.json"), "w"), indent=1)
