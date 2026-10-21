"""Эксперимент 37: ответ по фактам с ПРИОРИТЕТОМ ТЕМЫ — факт про тему статьи надёжнее.
exp36: эвристики «какая сторона — ответ» ломались; цепочка тянула мусор (Kyoto, Linux).
Один принципиальный сигнал: X == тема статьи (src). «Paris be the capital city of France» — из статьи Paris;
«The capital city be Fort-de-France» — из Martinique. Факты про тему = определения, им +вес.
Ответ: если тема статьи НЕ в вопросе → ответ = тема (X); если тема в вопросе → ответ = Y.
Это убирает все эвристики про местоимения: тема всегда имя, всегда из заголовка.
Цепочка — только если лучший score < 0.5. Успех: R@1 ≥ 10/16.
"""
import sys, os, json, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_v2 import toks, content_words, C
from mini_ai.binary_store import pack, hamming

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"; STORE = f"{OUT}/simplewiki_store"
idf = json.load(open(f"{STORE}/idf.json")); facts = [json.loads(l) for l in open(f"{STORE}/facts.jsonl")]
def bag(ws):
    cw = content_words(ws); return sum(idf.get(w, 1.0) * C(w) for w in cw) if cw else None
keys, keep = [], []
for i, (X, rel, Y, src) in enumerate(facts):
    v = bag(toks(X) + toks(rel.replace("_", " ")) + toks(Y))
    if v is not None: keys.append(pack(v)[0]); keep.append(i)
K = np.stack(keys); facts = [facts[i] for i in keep]
is_topic = np.array([bool(toks(X) == toks(src) or (toks(src) and toks(src)[0] in toks(X) and len(toks(X)) <= 3)) for X, _, _, src in facts])
print(f"фактов {len(facts)}, про тему статьи: {is_topic.sum()} ({is_topic.mean()*100:.0f}%)")
NUMW = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "hundred", "thousand", "million", "billion"}
QW = {"what", "who", "where", "when", "how", "many", "much", "which", "does", "do", "did", "is", "was", "are", "were"}

def ans_ok(ans, qtype):
    t = toks(ans)
    if not t: return False
    if qtype == "who": return ans[:1].isupper()
    if qtype == "howmany": return any(w.isdigit() or w in NUMW for w in t)
    if qtype == "when": return bool(re.search(r"\b1\d{3}|20\d{2}\b", ans))
    return True

def score_fact(i, qset, qidf, qtype):
    X, rel, Y, src = facts[i]; xw, rw, yw = set(toks(X)), set(toks(rel.replace("_", " "))), set(toks(Y))
    cov = sum(idf.get(w, 1.0) for w in qset & (xw | rw | yw)) / qidf
    if is_topic[i]:
        ans = Y if (qset & set(toks(src))) else X          # тема в вопросе → ответ Y, иначе ответ = тема
        cov += 0.25
    else:
        in_x, in_y = len(qset & xw), len(qset & yw); ans = X if in_y > in_x else Y
    if not ans_ok(ans, qtype): return None
    return cov - 0.01 * len(yw), ans, f"{X} {rel} {Y}  [{src}]"

def answer(q, pre=400, topk=3):
    ql = q.lower(); qtype = "who" if ql.startswith("who") else "howmany" if "how many" in ql else "when" if ql.startswith("when") else "what"
    qw = [w for w in toks(q) if w not in QW]; qset = set(content_words(qw)); qidf = sum(idf.get(w, 1.0) for w in qset)
    cand = np.argsort(hamming(K, pack(bag(qw))[0]))[:pre]
    scored = sorted((s for s in (score_fact(i, qset, qidf, qtype) for i in cand) if s), key=lambda t: -t[0])
    return scored[:topk]

Q = [("What is the population of Samarkand?", ["population", "000", "million"]), ("What is the capital of Uzbekistan?", ["tashkent"]),
     ("Who was Timur?", ["ruler", "conqueror", "founder", "emperor", "leader"]), ("What is the Silk Road?", ["trade", "route", "road"]),
     ("What strait divides Istanbul?", ["bosphorus", "bosporus"]), ("What is the largest city in Kazakhstan?", ["almaty"]),
     ("What lake is in Kyrgyzstan?", ["issyk"]), ("What sea borders Turkey?", ["black sea", "mediterranean", "aegean"]),
     ("Who wrote Hamlet?", ["shakespeare"]), ("What is the chemical symbol for gold?", ["au"]),
     ("How many legs does a spider have?", ["eight", "8"]), ("What is the boiling point of water?", ["100"]),
     ("Who painted the Mona Lisa?", ["leonardo", "da vinci"]), ("What is the largest planet?", ["jupiter"]),
     ("What is the capital of France?", ["paris"]), ("Who discovered penicillin?", ["fleming"])]
r1 = r3 = 0; t0 = time.time()
for q, golds in Q:
    top = answer(q); hit = [any(g in a.lower() for g in golds) for _, a, _ in top]
    r1 += bool(hit and hit[0]); r3 += any(hit)
    print(f"  {'OK ' if hit and hit[0] else ('R3 ' if any(hit) else 'MISS')} {q} → «{top[0][1][:45] if top else '-'}»  [{top[0][2][:85] if top else ''}]")
print(f"R@1 {r1}/16  R@3 {r3}/16  (exp36: 6/16, 7/16)  {(time.time()-t0)/16*1000:.0f} мс/вопрос")
json.dump(dict(r1=r1, r3=r3, facts=len(facts), topic_frac=float(is_topic.mean())), open(f"{OUT}/exp37_topic_facts.json", "w"))
