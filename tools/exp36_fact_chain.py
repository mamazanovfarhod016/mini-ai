"""Эксперимент 36: ответ по фактам — покрытие по X∪Y + ответ = непокрытая часть; тип ответа от вопросного слова.
exp35b: покрытие считалось отдельно по Y или по X → факт (Gold, symbol, Au) не находился для «chemical symbol
for gold» (слово «gold» в X, «symbol» в rel). Теперь: покрытие вопроса словами X∪rel∪Y; ответ = та часть,
где слов вопроса меньше. Тип ответа: «who» → X должен быть именем (заглавная), «how many» → в ответе цифра
или числительное, «when» → год. Цепочка (2 шага): если лучший факт даёт ответ-сущность Z, а вопрос «who/what»
ещё не покрыт — ищем факты Z. Успех: R@1 ≥ 10/16.
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
for i, (X, rel, Y) in enumerate(facts):
    v = bag(toks(X) + toks(rel.replace("_", " ")) + toks(Y))
    if v is not None: keys.append(pack(v)[0]); keep.append(i)
K = np.stack(keys); facts = [facts[i] for i in keep]
PRON = {"it", "this", "that", "these", "those", "they", "he", "she", "there", "the", "a", "its", "his", "her", "their", "but", "when",
        "in", "on", "at", "past", "some", "many", "most", "all", "one", "other", "capital", "city", "population"}
NUMW = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "hundred", "thousand", "million", "billion"}
QW = {"what", "who", "where", "when", "how", "many", "much", "which", "does", "do", "did", "is", "was", "are", "were"}
BAD_START = re.compile(r"^(?:When|If|But|And|Or|Because|While|Although|In the|At the|On the|Daddy|Some|Many|Most)\b")

def ans_ok(ans, qtype):
    t = toks(ans)
    if not t or set(t) <= PRON or BAD_START.match(ans): return False
    if qtype == "who": return ans[:1].isupper() and not any(w in ("museum", "play", "film", "book") for w in t)
    if qtype == "howmany": return any(w.isdigit() or w in NUMW for w in t)
    if qtype == "when": return bool(re.search(r"\b1\d{3}|20\d{2}\b", ans))
    return True

def score_fact(i, qset, qidf, qtype):
    X, rel, Y = facts[i]; xw, rw, yw = set(toks(X)), set(toks(rel.replace("_", " "))), set(toks(Y))
    cov = sum(idf.get(w, 1.0) for w in qset & (xw | rw | yw)) / qidf
    in_x, in_y = len(qset & xw), len(qset & yw)
    # ответ — сторона с МЕНЬШИМ числом слов вопроса; при равенстве — та, где нет местоимений/служебных
    if in_y > in_x: ans = X
    elif in_x > in_y: ans = Y
    else: ans = X if not (xw & PRON) and len(xw) <= 3 else Y
    if BAD_START.match(X) or (xw & PRON and len(xw) <= 2): cov -= 0.3
    if not ans_ok(ans, qtype): return None
    return cov - 0.01 * len(yw), ans, f"{X} {rel} {Y}"

def answer(q, pre=400, topk=3):
    ql = q.lower(); qtype = "who" if ql.startswith("who") else "howmany" if "how many" in ql else "when" if ql.startswith("when") else "what"
    qw = [w for w in toks(q) if w not in QW]; qset = set(content_words(qw)); qidf = sum(idf.get(w, 1.0) for w in qset)
    h = hamming(K, pack(bag(qw))[0]); cand = np.argsort(h)[:pre]
    scored = [s for s in (score_fact(i, qset, qidf, qtype) for i in cand) if s]
    scored.sort(key=lambda t: -t[0])
    if not scored: return []
    best = scored[0]
    # цепочка: лучший факт покрывает вопрос слабо (<0.6) и даёт сущность-ответ → факты про неё
    if best[0] < 0.6 and best[1][:1].isupper():
        z = best[1]; zset = set(toks(z)); extra = []
        for j in np.argsort(hamming(K, pack(bag(toks(z) + qw))[0]))[:200]:
            X, rel, Y = facts[j]
            if set(toks(X)) & zset or set(toks(Y)) & zset:
                s = score_fact(j, qset | zset, qidf + sum(idf.get(w, 1.0) for w in zset), qtype)
                if s and s[1].lower() != z.lower(): extra.append((s[0], s[1], s[2] + f"  ← через «{z}»"))
        scored = sorted(scored + extra, key=lambda t: -t[0])
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
    print(f"  {'OK ' if hit and hit[0] else ('R3 ' if any(hit) else 'MISS')} {q} → «{top[0][1][:45] if top else '-'}»  [{top[0][2][:80] if top else ''}]")
print(f"R@1 {r1}/16  R@3 {r3}/16  (exp35b: 6/16)  {(time.time()-t0)/16*1000:.0f} мс/вопрос, фактов {len(facts)}")
json.dump(dict(r1=r1, r3=r3, facts=len(facts)), open(f"{OUT}/exp36_fact_answer.json", "w"))
