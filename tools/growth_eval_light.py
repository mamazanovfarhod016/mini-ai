"""growth_eval_light.py — быстрый честный тест САМОРАЗВИТИЯ для помощников (≈4–6 мин, ≤1.6 ГБ, без нейросетей).
Схема: срез SciQ train → первые 120 вопросов с учителем («yes» / «no, it's X»), затем 80 свежих БЕЗ учителя.
Сравнение: холодная (без учителя) vs выращенная на тех же 80 свежих. Печатает одну строку на срез и итог.
    tools/light.sh tools/growth_eval_light.py [slices=2000,5000] [teach=120] [test=80] [tag=name] [dump=/tmp/x.json]
Итог: GROWTH mean +x.xx — это и есть метрика. Положительное стабильно на всех срезах = рост работает.
dump= — по-вопросный разбор (холодная/выращенная: способ, ответ, была ли подсказка случаев) для диагностики won/lost.
"""
import sys, shutil, gc, os, json, time; sys.path.insert(0, ".")
import pandas as pd
from mini_ai.experimental import ExperimentalMind
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
slices = [int(x) for x in args.get("slices", "2000,5000").split(",")]; TEACH = int(args.get("teach", 120)); TEST = int(args.get("test", 80)); tag = args.get("tag", "run")
dump = args.get("dump")
tr = pd.read_parquet("data/school/sciq/data/train-00000-of-00001.parquet").dropna(subset=["support"])
def hit(o, g):
    a = ((o.get("text") or "") + " " + str(o.get("answer") or "")).lower(); return str(g).lower() in a
def rec(m, o):
    adv = getattr(m, "_advice", None) or {}
    return dict(how=o.get("how"), ans=str(o.get("answer"))[:60], adv_n=adv.get("n", 0), adv_top=round(adv.get("top", 0), 2), bold=adv.get("bold"), soft=sorted(adv.get("soft", [])), board=[(b[0], b[1], b[4]) for b in (o.get("arbiter") or [])][:5])
deltas = []; t0 = time.time(); D = []
for lo in slices:
    rows = list(tr.iloc[lo:lo + TEACH + TEST].iterrows()); h1, h2 = rows[:TEACH], rows[TEACH:]
    sd = f"/tmp/gel_{tag}_{lo}"
    shutil.rmtree(sd + "0", ignore_errors=True); m0 = ExperimentalMind(school=False, wiki=False, state_dir=sd + "0")
    for _, r in rows: m0.kb.read_text(str(r["support"]), "учебник")
    cold_r = []
    for _, r in h2:
        o = m0.ask(r["question"], internet=False); cold_r.append(dict(rec(m0, o), hit=hit(o, r["correct_answer"])))   # rec сразу: _advice живёт до следующего ask
    cold = sum(c["hit"] for c in cold_r) / len(h2); del m0; gc.collect()
    shutil.rmtree(sd + "1", ignore_errors=True); m1 = ExperimentalMind(school=False, wiki=False, state_dir=sd + "1")
    for _, r in rows: m1.kb.read_text(str(r["support"]), "учебник")
    nc = 0
    for _, r in h1:
        o = m1.ask(r["question"], internet=False)
        if hit(o, r["correct_answer"]): m1.ask("yes", internet=False)
        elif o.get("kind") in ("answer", "clarify", "unknown"): m1.ask(f"no, it's {r['correct_answer']}", internet=False); nc += 1
    gh = 0
    for c, (_, r) in zip(cold_r, h2):
        o = m1.ask(r["question"], internet=False); g = rec(m1, o); h = hit(o, r["correct_answer"]); gh += h
        D.append(dict(slice=lo, q=r["question"], gold=r["correct_answer"], cold=c, grown=g, cold_hit=c["hit"], grown_hit=h))
    grown = gh / len(h2)
    deltas.append(grown - cold)
    extra = ""
    if hasattr(m1, "cases"): extra += f" cases {len(m1.cases.items)}"
    print(f"[{lo}] cold {cold:.3f} → grown {grown:.3f} ({grown - cold:+.3f}) corrections {nc}{extra}", flush=True)
    del m1; gc.collect()
if dump: json.dump(D, open(dump, "w"), ensure_ascii=False, indent=0)
print(f"GROWTH mean {sum(deltas) / len(deltas):+.3f} over {len(deltas)} slices ({int(time.time() - t0)}s) tag={tag}", flush=True)
