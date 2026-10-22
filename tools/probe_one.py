"""probe_one.py — один вопрос на выращенном уме: доска арбитра + кандидаты _span (cov, qual, absent) для предложений с золотом.
    tools/light.sh tools/probe_one.py tag=b5 lo=8000 i=<индекс в срезе teach+test> 
"""
import sys, json, re; sys.path.insert(0, ".")
import pandas as pd
from mini_ai.experimental import ExperimentalMind
from mini_ai.memory_v2 import toks, content_words
from mini_ai.question import parse_question
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
tag = args.get("tag", "b5"); lo = int(args.get("lo", 8000)); qsub = args.get("q", "")
tr = pd.read_parquet("data/school/sciq/data/train-00000-of-00001.parquet").dropna(subset=["support"])
rows = list(tr.iloc[lo:lo + 200].iterrows())
m = ExperimentalMind(school=False, wiki=False, state_dir=f"/tmp/gel_{tag}_{lo}1")
for _, r in rows: m.kb.read_text(str(r["support"]), "учебник")
RF = m.reform
for _, r in rows[120:]:
    q, g = r["question"], str(r["correct_answer"])
    if qsub.lower() not in q.lower(): continue
    o = m.ask(q, internet=False)
    print("Q:", q, "| gold:", g); print("  ans:", o.get("answer"), "|", o.get("how"), "| conf", o.get("confidence"))
    for b in (o.get("arbiter") or [])[:6]: print("   board:", b[:6])
    fr = parse_question(q); m._cur_q = q
    known0 = set(fr["Y"] if fr["slot"] == "X" else fr["X"]) | set(fr["Y"] if fr["slot"] == "Y" else fr["X"])
    known = {w for w in known0 if w not in m.META} - RF.type_words(q)
    hard = {w for w in known if not m._soft(w)} or known
    idf = {w: m._widf(w) for w in known}; kidf = sum(idf[w] for w in hard) + 0.3 * sum(idf[w] for w in known - hard)
    print("  known:", sorted(known), "hard:", sorted(hard), "type:", RF.type_words(q), "thr≈", round(0.75 - 0.4 * m.growth.boldness(q), 2))
    for si, (sent, _) in enumerate(m.kb.sents):
        sw = set(content_words(toks(sent))); sf = set().union(*[m._forms(x) for x in sw]) if sw else set()
        absent = {w for w in known if not (m._forms(w) & sf)}
        cov = sum((idf[w] if w in hard else 0.3 * idf[w]) for w in known - absent) / (kidf or 1)
        if g.lower() in sent.lower() or cov >= 0.5:
            qual = RF.qualifiers(q, absent)
            print(f"   {'GOLD' if g.lower() in sent.lower() else '    '} S[{si}] cov {cov:.2f} absent {sorted(absent)} qual {sorted(qual)} :: {sent[:110]}")
