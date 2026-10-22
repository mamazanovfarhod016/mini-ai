"""probe: miss anatomy for one slice (cold mind). For each miss: gold sentence index rank in _span cnt, cov, rare_missing, picked piece."""
import sys, shutil, gc, json, collections; sys.path.insert(0, ".")
import pandas as pd
from mini_ai.experimental import ExperimentalMind
from mini_ai.memory_v2 import toks, content_words
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
lo = int(args.get("lo", 2000)); N = int(args.get("n", 200))
tr = pd.read_parquet("data/school/sciq/data/train-00000-of-00001.parquet").dropna(subset=["support"])
rows = list(tr.iloc[lo:lo + N].iterrows())
sd = f"/tmp/gel_probe_{lo}"; shutil.rmtree(sd, ignore_errors=True)
m = ExperimentalMind(school=False, wiki=False, state_dir=sd)
for _, r in rows: m.kb.read_text(str(r["support"]), "учебник")
def hit(o, g): return str(g).lower() in ((o.get("text") or "") + " " + str(o.get("answer") or "")).lower()
cls = collections.Counter(); out = []
for _, r in rows:
    o = m.ask(r["question"], internet=False); g = r["correct_answer"].lower()
    if hit(o, g): cls["hit"] += 1; continue
    dbg = getattr(m, "_span_dbg", None) or {}
    gold_si = [si for si in range(len(m.kb.sents)) if g in m.kb.sents[si][0].lower()]
    top = dbg.get("top", [])
    rank = next((i for i, t in enumerate(top) if t[0] in gold_si), None)
    if not gold_si: c = "gold-not-in-kb"
    elif rank is None: c = "gold-sent-not-in-top12"
    else:
        t = top[rank]
        if t[1] < dbg["thr"]: c = "gold-sent-below-thr"
        elif t[3] is None and t[2]: c = "gold-sent-rare-missing"
        elif t[3] is None: c = "gold-sent-no-piece"
        elif g not in t[3].lower(): c = "gold-sent-wrong-piece"
        else: c = "gold-piece-lost-to-other"
    cls[c] += 1
    out.append(dict(q=r["question"], gold=g, how=o.get("how"), ans=str(o.get("answer"))[:40], cls=c, thr=dbg.get("thr"), rank=rank, top=top[:4], gold_sent=m.kb.sents[gold_si[0]][0][:150] if gold_si else None))
print(cls)
json.dump(out, open(f"/tmp/gel_logs/probe_{lo}.json", "w"), ensure_ascii=False, indent=0)
for d in out[:25]:
    print(f"{d['cls']:24} thr={d['thr']} rank={d['rank']} | {d['q'][:60]} | gold={d['gold']} | {str(d['how'])[:14]}:{d['ans'][:20]} | top0={d['top'][0] if d['top'] else None}")
