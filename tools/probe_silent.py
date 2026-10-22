"""probe_silent.py — почему молчит на свежих вопросах после обучения: для каждого «не знаю» — лучшие предложения _span по покрытию,
их cov, absent-слова, qualifiers reform и причина отказа. Использует состояние выращенного ума из /tmp/gel_<tag>_<lo>1.
    tools/light.sh tools/probe_silent.py tag=b3 lo=2000
"""
import sys, json, re, collections; sys.path.insert(0, ".")
import pandas as pd
from mini_ai.experimental import ExperimentalMind
from mini_ai.memory_v2 import toks, content_words
from mini_ai.question import parse_question
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
tag = args.get("tag", "b3"); lo = int(args.get("lo", 2000)); TEACH, TEST = 120, 80
tr = pd.read_parquet("data/school/sciq/data/train-00000-of-00001.parquet").dropna(subset=["support"])
rows = list(tr.iloc[lo:lo + TEACH + TEST].iterrows()); h2 = rows[TEACH:]
m = ExperimentalMind(school=False, wiki=False, state_dir=f"/tmp/gel_{tag}_{lo}1")
for _, r in rows: m.kb.read_text(str(r["support"]), "учебник")
RF = m.reform
for _, r in h2:
    q, g = r["question"], str(r["correct_answer"]); o = m.ask(q, internet=False)
    a = ((o.get("text") or "") + " " + str(o.get("answer") or "")).lower()
    if g.lower() in a or not (o.get("how") or "").startswith("не знаю"): continue
    fr = parse_question(q); m._cur_q = q
    known0 = set(fr["Y"] if fr["slot"] == "X" else fr["X"]) | set(fr["Y"] if fr["slot"] == "Y" else fr["X"])
    known = {w for w in known0 if w not in m.META}
    hard = {w for w in known if not m._soft(w)} or known
    idf = {w: m._widf(w) for w in known}; kidf = sum(idf[w] for w in hard) + 0.3 * sum(idf[w] for w in known - hard)
    gold_si = [si for si, (s, _) in enumerate(m.kb.sents) if g.lower() in s.lower()]
    print("\n###", q, "| gold:", g, "| known:", sorted(known), "| type:", RF.typeword(q), "| thr≈", round(0.75 - 0.4 * m.growth.boldness(q), 2))
    for si in gold_si[:2]:
        sent = m.kb.sents[si][0]; sw = set(content_words(toks(sent))); sf = set().union(*[m._forms(x) for x in sw]) if sw else set()
        absent = {w for w in known if not (m._forms(w) & sf)}
        cov = sum((idf[w] if w in hard else 0.3 * idf[w]) for w in known - absent) / (kidf or 1)
        qual = RF.qualifiers(q, absent)
        rare = [w for w in absent if idf[w] >= 2.5 and not m._soft(w)]
        print(f"   S[{si}] cov {cov:.2f} absent {sorted(absent)} qual {sorted(qual)} rare_missing {rare} :: {sent[:120]}")
        print("      runs:", [(i, ws, ab) for i, ws, ab in RF.runs(q, lambda w: w not in absent) if ab])
