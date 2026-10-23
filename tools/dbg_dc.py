import sys, os, re, shutil; sys.path.insert(0, ".")
import pandas as pd
from mini_ai.experimental import ExperimentalMind
from mini_ai.question import parse_question
from mini_ai.memory_v2 import toks, content_words
DATA = os.path.expanduser("~/Developer/mini-ai/data")
tr = pd.read_parquet(f"{DATA}/school/sciq/data/train-00000-of-00001.parquet").dropna(subset=["support"])
B = []
for _, r in tr.iloc[5000:].iterrows():
    q, g, sup = r["question"], str(r["correct_answer"]).lower(), str(r["support"])
    sents = re.split(r"(?<=[.!?])\s+", sup); qa = {w for w in content_words(toks(q)) if len(w) > 3}
    with_ans = [s_ for s_ in sents if g in s_.lower()]
    if not with_ans or len(sents) < 3: continue
    if all(len(qa & set(content_words(toks(s_)))) <= 1 for s_ in with_ans): B.append(r)
    if len(B) >= 120: break
shutil.rmtree("/tmp/dbg_dc", ignore_errors=True); m = ExperimentalMind(school=False, wiki=False, state_dir="/tmp/dbg_dc")
for r in B: m.kb.read_text(str(r["support"]), "учебник")
for key in sys.argv[1:]:
    r = next(x for x in B if key.lower() in x["question"].lower()); q = r["question"]; fr = parse_question(q); m._cur_q = q
    print("\nQ:", q, "| gold:", r["correct_answer"]); print("SUPPORT:", str(r["support"])[:500])
    D = m.discourse; known = set(fr.get("X") or []) | set(fr.get("Y") or [])
    qw = {w for w in known if w not in m.META}; print("qw:", sorted(qw), "atype:", m.deep._goal(q))
    for bi, cov in D.bridges(qw, k=4):
        print(f"  bridge #{bi} cov={cov:.2f}: {m.kb.sents[bi][0][:100]}")
        for nj in (bi + 1, bi - 1, bi + 2, bi - 2):
            if 0 <= nj < len(m.kb.sents):
                lk, nm = D.link(bi, nj)
                if lk: print(f"     -> #{nj} link={lk} ({nm}): {m.kb.sents[nj][0][:80]}"); print("        cands:", D.candidates(m.kb.sents[nj][0], D._forms_of(D._words(q)) | D._words(q), set())[:4])
    print("RESULT:", {k: v for k, v in (D.reason(q, fr) or {}).items() if k in ("answer", "confidence", "why", "link", "cov")})
