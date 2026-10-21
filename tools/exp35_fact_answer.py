"""Эксперимент 35: ОТВЕТ ПО ФАКТАМ, не по предложениям (на 142k фактов из дампа).
exp33: поиск предложения R@1 3/16 — «It is also the capital city of France» найдено, но «Paris» там нет:
субъект — «It». Читатель уже заменил «It» на заголовок статьи → факт (Paris, be, also the capital city of France).
Значит, искать нужно по ФАКТАМ: вопрос → мешок слов → ближайший факт по (rel + Y) → ответ = X (субъект),
либо по X → ответ = Y. Два направления:
  «What is the capital of France?» → слова {capital, france} ∈ Y → X = Paris
  «Who was Timur?» → X = Timur → Y
Ключ факта = IDF-мешок(X ∪ Y) (бинарно); ответ = X если слова вопроса в Y, иначе Y.
Успех: R@1 ≥ 10/16 на тех же вопросах (exp33: 3/16), ответ-слово (не предложение) в top-1.
"""
import sys, os, json, time, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_v2 import toks, content_words, C, unit, D
from mini_ai.binary_store import BinaryStore, pack, hamming

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"; STORE = f"{OUT}/simplewiki_store"
idf = json.load(open(f"{STORE}/idf.json"))
facts = [json.loads(l) for l in open(f"{STORE}/facts.jsonl")]
print(f"фактов {len(facts)}")

def bag(ws):
    cw = content_words(ws); return sum(idf.get(w, 1.0) * C(w) for w in cw) if cw else None

t0 = time.time(); keys = []; keep = []
for i, (X, rel, Y) in enumerate(facts):
    v = bag(toks(X) + toks(Y))
    if v is not None: keys.append(pack(v)[0]); keep.append(i)
K = np.stack(keys); facts = [facts[i] for i in keep]
print(f"ключи фактов: {K.nbytes/1e6:.0f} МБ, {time.time()-t0:.0f}s")

Q = [("What is the population of Samarkand?", ["population"]), ("What is the capital of Uzbekistan?", ["tashkent"]),
     ("Who was Timur?", ["timur", "tamerlane", "ruler", "conqueror"]), ("What is the Silk Road?", ["trade", "route", "road"]),
     ("What strait divides Istanbul?", ["bosphorus", "bosporus"]), ("What is the largest city in Kazakhstan?", ["almaty"]),
     ("What lake is in Kyrgyzstan?", ["issyk"]), ("What sea borders Turkey?", ["black sea", "mediterranean", "aegean"]),
     ("Who wrote Hamlet?", ["shakespeare"]), ("What is the chemical symbol for gold?", ["au"]),
     ("How many legs does a spider have?", ["eight", "8"]), ("What is the boiling point of water?", ["100"]),
     ("Who painted the Mona Lisa?", ["leonardo", "da vinci"]), ("What is the largest planet?", ["jupiter"]),
     ("What is the capital of France?", ["paris"]), ("Who discovered penicillin?", ["fleming"])]

def answer(q, topk=3):
    qw = set(content_words(toks(q))); qv = bag(toks(q)); h = hamming(K, pack(qv)[0]); order = np.argsort(h)[:topk]
    out = []
    for i in order:
        X, rel, Y = facts[i]
        yw = set(toks(Y)); xw = set(toks(X))
        ans = X if len(qw & yw) >= len(qw & xw) else Y       # слова вопроса в Y → ответ X, иначе Y
        out.append((ans, f"{X} {rel} {Y}"))
    return out

r1 = r3 = 0; t0 = time.time()
for q, golds in Q:
    top = answer(q); hit = [any(g in a.lower() for g in golds) for a, _ in top]
    r1 += hit[0]; r3 += any(hit)
    print(f"  {'OK ' if hit[0] else ('R3 ' if any(hit) else 'MISS')} {q} → «{top[0][0][:50]}»  [{top[0][1][:80]}]")
print(f"R@1 {r1}/16  R@3 {r3}/16  (exp33 по предложениям: 3/16, 6/16)  {(time.time()-t0)/16*1000:.0f} мс/вопрос")
json.dump(dict(facts=len(facts), r1=r1, r3=r3), open(f"{OUT}/exp35_fact_answer.json", "w"), indent=1)
