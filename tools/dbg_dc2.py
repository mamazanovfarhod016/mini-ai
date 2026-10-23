import sys, os, re, shutil; sys.path.insert(0, ".")
import pandas as pd
from mini_ai.experimental import ExperimentalMind
from mini_ai.question import parse_question
from mini_ai.memory_v2 import toks, content_words
import mini_ai.discourse as DC
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
D = m.discourse
orig_c = D.candidates
def traced(sent, ex, qm, antecedent=False):
    c = orig_c(sent, ex, qm, antecedent)
    print(f"      cands(ant={antecedent}, subj={D._subject(sent) if antecedent else None}) {sent[:60]!r}: {c[:3]}")
    return c
D.candidates = traced
orig_l = D.link
def tl(bi, nj):
    r = orig_l(bi, nj)
    if r[0]: print(f"   link #{bi}->#{nj} = {r}")
    return r
D.link = tl
for key in sys.argv[1:]:
    r = next(x for x in B if key.lower() in x["question"].lower()); q = r["question"]; fr = parse_question(q); m._cur_q = q
    print("\nQ:", q, "| gold:", r["correct_answer"])
    res = D.reason(q, fr)
    print("RESULT:", {k: v for k, v in (res or {}).items() if k in ("answer", "confidence", "why", "link", "cov", "bridge")})
