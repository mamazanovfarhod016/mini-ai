"""probe_hop.py — анатомия промахов multi-hop (набор B судьи: ответное предложение делит ≤1 слово с вопросом).
Для каждого промаха: где лежит золото (предложение), какое предложение по словам ближе к вопросу (мост), через какие слова
мост связан с золотым предложением (общие термины), и что ответила. Классы: no-bridge / bridge-found-wrong-piece / silent."""
import sys, os, re, json, shutil, gc, collections; sys.path.insert(0, ".")
import pandas as pd
from mini_ai.experimental import ExperimentalMind
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
sd = "/tmp/probe_hop"; shutil.rmtree(sd, ignore_errors=True)
m = ExperimentalMind(school=False, wiki=False, state_dir=sd)
for r in B: m.kb.read_text(str(r["support"]), "учебник")
def hit(o, g): return str(g).lower() in ((o.get("text") or "") + " " + str(o.get("answer") or "")).lower()
cls = collections.Counter(); out = []
STOP = set("the a an of in on at to for by with from and or is are was were be been do does did what which who how why when where that this these those it its their there as into than then so such also can may most some many more other".split())
for r in B:
    q, g, sup = r["question"], str(r["correct_answer"]).lower(), str(r["support"])
    o = m.ask(q, internet=False); h = hit(o, g)
    sents = re.split(r"(?<=[.!?])\s+", sup)
    qw = {w for w in content_words(toks(q)) if w not in STOP and len(w) > 2}
    gold = [i for i, s in enumerate(sents) if g in s.lower()]
    # мост: предложение абзаца с макс. пересечением с вопросом
    ov = [(len(qw & {w for w in content_words(toks(s)) if w not in STOP}), i) for i, s in enumerate(sents)]
    bov, bi = max(ov)
    gw = {w for w in content_words(toks(sents[gold[0]])) if w not in STOP and len(w) > 3} if gold else set()
    bw = {w for w in content_words(toks(sents[bi])) if w not in STOP and len(w) > 3}
    link = sorted((gw & bw) - qw)
    adjacent = gold and abs(gold[0] - bi) == 1
    if h: c = "hit"
    elif not o.get("answer"): c = "silent"
    else: c = "wrong"
    cls[c] += 1
    out.append(dict(q=q, gold=g, hit=h, cls=c, how=o.get("how"), ans=str(o.get("answer"))[:50], bridge_ov=bov, bridge_idx=bi, gold_idx=gold[:1], adjacent=bool(adjacent), link=link[:6],
                    bridge=sents[bi][:140], gold_sent=(sents[gold[0]][:140] if gold else None)))
print(cls)
print("misses: adjacent gold/bridge", sum(1 for d in out if not d["hit"] and d["adjacent"]), "| link words>0", sum(1 for d in out if not d["hit"] and d["link"]), "| bridge_ov>=2", sum(1 for d in out if not d["hit"] and d["bridge_ov"] >= 2))
print("miss how:", collections.Counter((d["how"] or "?")[:22] for d in out if not d["hit"]).most_common())
json.dump(out, open("/tmp/probe_hop.json", "w"), ensure_ascii=False, indent=0)
for d in [x for x in out if not x["hit"]][:30]:
    print(f"\n[{d['cls']}] {d['q'][:80]} | gold={d['gold']} | ans={d['how'][:14] if d['how'] else ''}:{d['ans'][:30]}\n   bridge(ov{d['bridge_ov']},#{d['bridge_idx']}): {d['bridge'][:110]}\n   gold(#{d['gold_idx']}, adj={d['adjacent']}, link={d['link'][:4]}): {d['gold_sent'][:110] if d['gold_sent'] else None}")
