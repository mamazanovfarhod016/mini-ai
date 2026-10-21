"""Эксперимент 35b: ответ по фактам — ТОЧНОЕ покрытие слов, не мешок (внутри той же системы).
exp35 (мешок X∪Y бинарно): 1/16 — побеждают короткие факты с общим словом («The planet be also 1»).
Факты есть: (Jupiter, be, the largest planet in the Solar System). Нужен счёт, который требует ВСЕ
содержательные слова вопроса: score = |Q ∩ words(Y)| / |Q| (покрытие) с IDF-весом, штраф за местоимённый
субъект (These/It/That), ответ = X. Для «who was X» (X в вопросе) — ответ Y.
Бинарный поиск остаётся как ПРЕДФИЛЬТР (top-200 по Хэммингу), затем точный счёт — это и есть
«вектор = индекс, текст рядом» (как в обсуждении RAG). Успех: R@1 ≥ 10/16.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_v2 import toks, content_words, C
from mini_ai.binary_store import pack, hamming

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"; STORE = f"{OUT}/simplewiki_store"
idf = json.load(open(f"{STORE}/idf.json")); facts = [json.loads(l) for l in open(f"{STORE}/facts.jsonl")]
def bag(ws):
    cw = content_words(ws); return sum(idf.get(w, 1.0) * C(w) for w in cw) if cw else None
keys, keep = [], []
for i, (X, rel, Y) in enumerate(facts):
    v = bag(toks(X) + toks(Y))
    if v is not None: keys.append(pack(v)[0]); keep.append(i)
K = np.stack(keys); facts = [facts[i] for i in keep]
PRON = {"it", "this", "that", "these", "those", "they", "he", "she", "there", "the", "a"}

Q = [("What is the population of Samarkand?", ["population"]), ("What is the capital of Uzbekistan?", ["tashkent"]),
     ("Who was Timur?", ["timur", "tamerlane", "ruler", "conqueror"]), ("What is the Silk Road?", ["trade", "route", "road"]),
     ("What strait divides Istanbul?", ["bosphorus", "bosporus"]), ("What is the largest city in Kazakhstan?", ["almaty"]),
     ("What lake is in Kyrgyzstan?", ["issyk"]), ("What sea borders Turkey?", ["black sea", "mediterranean", "aegean"]),
     ("Who wrote Hamlet?", ["shakespeare"]), ("What is the chemical symbol for gold?", ["au"]),
     ("How many legs does a spider have?", ["eight", "8"]), ("What is the boiling point of water?", ["100"]),
     ("Who painted the Mona Lisa?", ["leonardo", "da vinci"]), ("What is the largest planet?", ["jupiter"]),
     ("What is the capital of France?", ["paris"]), ("Who discovered penicillin?", ["fleming"])]

def answer(q, pre=300, topk=3):
    qw = content_words(toks(q)); qset = set(qw); qidf = sum(idf.get(w, 1.0) for w in qset)
    h = hamming(K, pack(bag(toks(q)))[0]); cand = np.argsort(h)[:pre]
    scored = []
    for i in cand:
        X, rel, Y = facts[i]; xw, yw = set(toks(X)), set(toks(Y))
        cov_y = sum(idf.get(w, 1.0) for w in qset & yw) / qidf       # покрытие вопроса словами Y
        cov_x = sum(idf.get(w, 1.0) for w in qset & xw) / qidf       # … словами X
        if cov_y >= cov_x: ans, score = X, cov_y - (0.5 if xw & PRON and len(xw) <= 2 else 0) - 0.01 * len(yw)
        else: ans, score = Y, cov_x - 0.01 * len(yw)
        scored.append((score, ans, f"{X} {rel} {Y}"))
    scored.sort(key=lambda t: -t[0]); return scored[:topk]

r1 = r3 = 0; t0 = time.time()
for q, golds in Q:
    top = answer(q); hit = [any(g in a.lower() for g in golds) for _, a, _ in top]
    r1 += hit[0]; r3 += any(hit)
    print(f"  {'OK ' if hit[0] else ('R3 ' if any(hit) else 'MISS')} {q} → «{top[0][1][:45]}»  [{top[0][2][:75]}]")
print(f"R@1 {r1}/16  R@3 {r3}/16  (exp35 мешок: 1/16; exp33 предложения: 3/16)  {(time.time()-t0)/16*1000:.0f} мс/вопрос")
json.dump(dict(r1=r1, r3=r3), open(f"{OUT}/exp35b_fact_answer.json", "w"))
