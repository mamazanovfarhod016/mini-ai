"""probe_reform.py — диагностика для reform: на срезе SciQ (200 вопросов) холодный ответ + предложение с золотом +
слова вопроса, которых нет в предложении (лишние/перефраз) + префиксные фразы-уточнители. Пишет JSON.
    tools/light.sh tools/probe_reform.py lo=2000 n=200 out=/tmp/wtb/probe_2000.json
"""
import sys, re, json, shutil, gc; sys.path.insert(0, ".")
import pandas as pd
from mini_ai.experimental import ExperimentalMind
from mini_ai.memory_v2 import toks, content_words
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
lo = int(args.get("lo", 2000)); n = int(args.get("n", 200)); out = args.get("out", f"/tmp/wtb/probe_{lo}.json")
tr = pd.read_parquet("data/school/sciq/data/train-00000-of-00001.parquet").dropna(subset=["support"])
rows = list(tr.iloc[lo:lo + n].iterrows())
sd = f"/tmp/wtb/probe_{lo}"; shutil.rmtree(sd, ignore_errors=True)
m = ExperimentalMind(school=False, wiki=False, state_dir=sd)
for _, r in rows: m.kb.read_text(str(r["support"]), "учебник")
def hit(o, g):
    a = ((o.get("text") or "") + " " + str(o.get("answer") or "")).lower(); return str(g).lower() in a
res = []
for i, (_, r) in enumerate(rows):
    q, g = r["question"], str(r["correct_answer"])
    o = m.ask(q, internet=False)
    rl = g.lower(); qw = [w for w in content_words(toks(q)) if w not in m.META]
    best = None
    for si, (sent, src) in enumerate(m.kb.sents):
        if rl not in sent.lower(): continue
        sw = set(content_words(toks(sent))); sf = set().union(*[m._forms(x) for x in sw]) if sw else set()
        cov = sum(1 for w in qw if m._forms(w) & sf)
        if not best or cov > best[0]: best = (cov, si, sent, sf, sw)
    d = dict(i=lo + i, q=q, gold=g, ans=str(o.get("answer")), how=o.get("how"), hit=hit(o, g), kind=o.get("kind"), conf=o.get("confidence"),
             board=[(b[0], b[1], b[4]) for b in (o.get("arbiter") or [])][:5])
    if best:
        cov, si, sent, sf, sw = best
        d["sent"] = sent; d["extra"] = [w for w in qw if not (m._forms(w) & sf)]; d["needed"] = [w for w in qw if m._forms(w) & sf]
        d["s_only"] = [w for w in sw if not any(m._forms(w) & m._forms(x) for x in qw) and w not in rl]
    res.append(d)
    if i % 40 == 39: print(i + 1, sum(x["hit"] for x in res), flush=True)
json.dump(res, open(out, "w"), ensure_ascii=False, indent=0)
print("hits", sum(x["hit"] for x in res), "/", len(res), "→", out)
