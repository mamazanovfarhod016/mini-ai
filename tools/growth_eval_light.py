"""growth_eval_light.py — быстрый честный тест САМОРАЗВИТИЯ для помощников (≈4–6 мин, ≤1.6 ГБ, без нейросетей).
Схема: срез SciQ train → первые 120 вопросов с учителем («yes» / «no, it's X»), затем 80 свежих БЕЗ учителя.
Сравнение: холодная (без учителя) vs выращенная на тех же 80 свежих. Печатает одну строку на срез и итог.
    tools/light.sh tools/growth_eval_light.py [slices=2000,5000] [teach=120] [test=80] [tag=name]
Итог: GROWTH mean +x.xx — это и есть метрика. Положительное стабильно на всех срезах = рост работает.
"""
import sys, shutil, gc, os, json, time; sys.path.insert(0, ".")
import pandas as pd
from mini_ai.experimental import ExperimentalMind
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
slices = [int(x) for x in args.get("slices", "2000,5000").split(",")]; TEACH = int(args.get("teach", 120)); TEST = int(args.get("test", 80)); tag = args.get("tag", "run")
dump = args.get("dump"); per_q = []                     # dump=path.json → по-вопросный разбор (cold/grown ответ, how) для A/B won/lost
tr = pd.read_parquet("data/school/sciq/data/train-00000-of-00001.parquet").dropna(subset=["support"])
def hit(o, g):
    a = ((o.get("text") or "") + " " + str(o.get("answer") or "")).lower(); return str(g).lower() in a
deltas = []; t0 = time.time()
for lo in slices:
    rows = list(tr.iloc[lo:lo + TEACH + TEST].iterrows()); h1, h2 = rows[:TEACH], rows[TEACH:]
    sd = f"/tmp/gel_{tag}_{lo}"
    shutil.rmtree(sd + "0", ignore_errors=True); m0 = ExperimentalMind(school=False, wiki=False, state_dir=sd + "0")
    for _, r in rows: m0.kb.read_text(str(r["support"]), "учебник")
    o0 = [m0.ask(r["question"], internet=False) for _, r in h2]
    cold = sum(hit(o, r["correct_answer"]) for o, (_, r) in zip(o0, h2)) / len(h2); del m0; gc.collect()
    shutil.rmtree(sd + "1", ignore_errors=True); m1 = ExperimentalMind(school=False, wiki=False, state_dir=sd + "1")
    for _, r in rows: m1.kb.read_text(str(r["support"]), "учебник")
    nc = 0
    for _, r in h1:
        o = m1.ask(r["question"], internet=False)
        if hit(o, r["correct_answer"]): m1.ask("yes", internet=False)
        elif o.get("kind") in ("answer", "clarify", "unknown"): m1.ask(f"no, it's {r['correct_answer']}", internet=False); nc += 1
    o1 = [m1.ask(r["question"], internet=False) for _, r in h2]
    grown = sum(hit(o, r["correct_answer"]) for o, (_, r) in zip(o1, h2)) / len(h2)
    for a, b, (_, r) in zip(o0, o1, h2):
        per_q.append(dict(lo=lo, q=r["question"], gold=str(r["correct_answer"]), cold=hit(a, r["correct_answer"]), grown=hit(b, r["correct_answer"]),
                          a_cold=str(a.get("answer"))[:60], a_grown=str(b.get("answer"))[:60], how_cold=(a.get("how") or "")[:30], how_grown=(b.get("how") or "")[:30]))
    if hasattr(m1, "reform"): extra_r = " | " + m1.reform.report()[:160]
    else: extra_r = ""
    deltas.append(grown - cold)
    extra = ""
    if hasattr(m1, "cases"): extra += f" cases {len(m1.cases.items)}"
    print(f"[{lo}] cold {cold:.3f} → grown {grown:.3f} ({grown - cold:+.3f}) corrections {nc}{extra}{extra_r}", flush=True)
    del m1; gc.collect()
if dump: json.dump(per_q, open(dump, "w"), ensure_ascii=False, indent=0)
print(f"GROWTH mean {sum(deltas) / len(deltas):+.3f} over {len(deltas)} slices ({int(time.time() - t0)}s) tag={tag}", flush=True)
