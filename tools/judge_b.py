"""judge B only (multi-hop 120) with per-question dump; args: tag=x [dump=path]. Light-runnable (school=False → only B supports in memory,
like probe_hop) OR school=True like the judge (heavy). Default school=0."""
import sys, os, re, json, shutil, gc, collections; sys.path.insert(0, ".")
import pandas as pd
from mini_ai.experimental import ExperimentalMind
from mini_ai.memory_v2 import toks, content_words
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a); tag = args.get("tag", "x"); school = args.get("school", "0") == "1"
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
sd = f"/tmp/judgeB_{tag}"; shutil.rmtree(sd, ignore_errors=True)
m = ExperimentalMind(school=school, wiki=False, state_dir=sd)
for r in B: m.kb.read_text(str(r["support"]), "учебник")
def hit(o, g): return str(g).lower() in ((o.get("text") or "") + " " + str(o.get("answer") or "")).lower()
D = []; hk = lambda h: (h or "?").split("«")[0].split(":")[0].strip()[:24]
for r in B:
    o = m.ask(r["question"], internet=False); h = hit(o, r["correct_answer"])
    D.append(dict(q=r["question"], gold=r["correct_answer"], hit=h, how=o.get("how"), ans=str(o.get("answer"))[:60], board=[(b[0][:22], str(b[1])[:30], b[4]) for b in (o.get("arbiter") or [])][:5]))
acc = sum(d["hit"] for d in D) / len(D)
print(f"B multi-hop {len(D)}  {acc:.3f}  tag={tag} school={school}")
print("how (hits):", collections.Counter(hk(d["how"]) for d in D if d["hit"]).most_common())
print("how (miss):", collections.Counter(hk(d["how"]) for d in D if not d["hit"]).most_common())
gob = sum(1 for d in D if not d["hit"] and any(str(d["gold"]).lower() in b[1].lower() for b in d["board"]))
print("gold on board not chosen:", gob)
if args.get("dump"): json.dump(D, open(args["dump"], "w"), ensure_ascii=False, indent=0)
