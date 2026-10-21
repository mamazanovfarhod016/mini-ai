"""Эксперимент 32: БИНАРНЫЕ КОДЫ — та же архитектура, другая арифметика (для дампа Википедии на устройстве).
Сейчас: вектор float64 D=2048 = 16 КБ на предложение → 1 млн предложений = 16 ГБ. Не влезает.
Бинарно: знак вектора → D бит = 256 байт → 1 млн = 256 МБ. Поиск = XOR + popcount (нет FFT, нет float).
Проверяем на одном и том же (память exp31: 4728 предл., 12 статей, 1730 фактов):
  1. Поиск предложения по вопросу (мешок IDF): R@1/R@3 float vs бинарно — те же 8 вопросов exp24 + 8 новых.
  2. Соседи по связям (exp31 A, 35/36 float) — бинарно.
  3. Скорость и память: 4728 × (16 КБ vs 256 Б); время 100 запросов.
  4. Экстраполяция: 1 млн предложений — время запроса бинарно (попкаунт на numpy uint8).
Бинаризация: b = sign(v) → упаковка np.packbits. Сходство = D − 2·popcount(xor) (= скалярное произведение ±1).
"""
import sys, os, json, time, re, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_v2 import Memory, toks, content_words, C, bind, unit, D

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
mem = Memory(path=f"{OUT}/exp31_memory"); raw, meta = mem.L.stext, mem.L.smeta
V = np.stack(mem.L.sk); print(f"предложений {len(raw)}, float: {V.nbytes/1e6:.1f} МБ")

def pack(M): return np.packbits((M > 0).astype(np.uint8), axis=-1)
B = pack(V); print(f"бинарно: {B.nbytes/1e6:.2f} МБ (×{V.nbytes/B.nbytes:.0f} меньше)")
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)
def ham_sim(Bm, q): return D - 2 * POP[np.bitwise_xor(Bm, q)].sum(axis=-1)

QUESTIONS = [
    ("What is the population of Samarkand?", ["population"]), ("Who built the Registan?", ["registan"]),
    ("What river flows through Tashkent?", ["chirchik", "river"]), ("What is the capital of Uzbekistan?", ["tashkent"]),
    ("Who was Timur?", ["timur", "tamerlane"]), ("What is the Silk Road?", ["silk road"]),
    ("When was Astana made the capital?", ["1997", "capital"]), ("What strait divides Istanbul?", ["bosphorus", "bosporus"]),
    ("What is the largest city in Kazakhstan?", ["almaty"]), ("What mountains are near Almaty?", ["tian shan", "trans-ili", "alatau"]),
    ("When was Ankara made the capital of Turkey?", ["1923"]), ("What is the old name of Bishkek?", ["frunze", "pishpek"]),
    ("What lake is in Kyrgyzstan?", ["issyk"]), ("What is Bukhara known for?", ["bukhara"]),
    ("What is the currency of Uzbekistan?", ["som", "soum"]), ("What sea borders Turkey?", ["black sea", "mediterranean", "aegean"]),
]
def evalq(search, label):
    r1 = r3 = 0; t0 = time.time()
    for q, golds in QUESTIONS:
        top = search(mem.sieve.bag(toks(q)))[:3]
        hit = [any(g in raw[i].lower() for g in golds) for i in top]
        r1 += hit[0]; r3 += any(hit)
    dt = (time.time() - t0) / len(QUESTIONS)
    print(f"  {label:10} R@1 {r1}/{len(QUESTIONS)}  R@3 {r3}/{len(QUESTIONS)}  {dt*1000:.2f} мс/запрос")
    return r1, r3, dt
print("--- 1. поиск предложения по вопросу ---")
f_res = evalq(lambda q: np.argsort(-(V @ q)), "float")
b_res = evalq(lambda q: np.argsort(-ham_sim(B, pack(q[None])[0])), "binary")

# --- 2. соседи по связям, бинарно (читатель exp31, теперь mini_ai/reader.py) ---
from mini_ai.reader import read_all
facts, _ = read_all(raw, [m.get("src", "") for m in meta])
facts = [(X, rel, Y) for X, rel, Y, _ in facts if "@" not in rel]
print(f"читатель: фактов {len(facts)}")
CITIES = ["Tashkent", "Samarkand", "Bukhara", "Astana", "Almaty", "Istanbul", "Ankara", "Bishkek"]
COUNTRIES = ["Uzbekistan", "Kazakhstan", "Turkey", "Kyrgyzstan"]
def yvec(Y):
    cw = content_words(toks(Y)) or toks(Y); return unit(sum(mem.sieve.idf(w) * C(w) for w in cw)) if cw else None
ent = collections.defaultdict(lambda: np.zeros(D))
for X, rel, Y in facts:
    yv = yvec(Y)
    if yv is not None: ent[X] += bind(rel, yv)
names = [n for n in CITIES + COUNTRIES if n in ent]; E = np.stack([unit(ent[n]) for n in names]); EB = pack(E)
def nb_eval(sim, label):
    good = 0
    for i, n in enumerate(names):
        nb = [names[j] for j in np.argsort(-sim(i))[1:4]]; kind = CITIES if n in CITIES else COUNTRIES
        good += sum(x in kind for x in nb)
    print(f"  {label:10} соседи того же типа {good}/{3*len(names)}"); return good
print("--- 2. смысл = связи ---")
nf = nb_eval(lambda i: E @ E[i], "float"); nbn = nb_eval(lambda i: ham_sim(EB, EB[i]), "binary")

# --- 3/4. масштаб: 1 млн предложений бинарно ---
print("--- 3. масштаб ---")
N = 1_000_000; rng = np.random.default_rng(0)
big = rng.integers(0, 256, size=(N, D // 8), dtype=np.uint8)
q = big[123]; t0 = time.time(); s = ham_sim(big, q); i = int(np.argmax(s)); dt = time.time() - t0
print(f"  1 000 000 предложений бинарно: {big.nbytes/1e6:.0f} МБ, запрос {dt*1000:.0f} мс, нашёл себя: {i == 123}")
print(f"  (float64 было бы {N*D*8/1e9:.0f} ГБ)")
json.dump(dict(n=len(raw), float_mb=V.nbytes/1e6, bin_mb=B.nbytes/1e6, search_float=f_res, search_bin=b_res,
               nb_float=nf, nb_bin=nbn, million_ms=dt*1000, million_mb=big.nbytes/1e6),
          open(f"{OUT}/exp32_binary.json", "w"), indent=1)
