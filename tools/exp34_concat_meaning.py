"""Эксперимент 34: ДВА ВИДА СМЫСЛА В ОДНОМ ВЕКТОРЕ — [связи | мешок] (внутри той же системы, exp31 →).
exp31: связи дают ТИП (соседи 35/36, аналогии 2/4), мешок даёт ПРИНАДЛЕЖНОСТЬ (соседи 21/36, аналогии 4/4).
Гипотеза: concat [связи | мешок] (как exp12: [тип | содержание]) даёт оба. Успех: соседи ≥33/36 И аналогии 4/4.
Также проверяем вес α: [α·связи | (1−α)·мешок], α ∈ {0.3, 0.5, 0.7}.
"""
import sys, os, json, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_v2 import Memory, toks, content_words, C, bind, unit, D
from mini_ai.reader import read_all

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
mem = Memory(path=f"{OUT}/exp31_memory"); raw, meta = mem.L.stext, mem.L.smeta
CITIES = ["Tashkent", "Samarkand", "Bukhara", "Astana", "Almaty", "Istanbul", "Ankara", "Bishkek"]
COUNTRIES = ["Uzbekistan", "Kazakhstan", "Turkey", "Kyrgyzstan"]
CAPITAL = {"Uzbekistan": "Tashkent", "Kazakhstan": "Astana", "Turkey": "Ankara", "Kyrgyzstan": "Bishkek"}
NAMES = CITIES + COUNTRIES

facts, _ = read_all(raw, [m.get("src", "") for m in meta])
link = collections.defaultdict(lambda: np.zeros(D))
for X, rel, Y, _ in facts:
    if "@" in rel: continue
    cw = content_words(toks(Y)) or toks(Y)
    if cw: link[X] += bind(rel, unit(sum(mem.sieve.idf(w) * C(w) for w in cw)))
bag = collections.defaultdict(lambda: np.zeros(D))
for si, s in enumerate(raw): bag[meta[si]["src"]] += mem.sieve.bag(toks(s))

def evaluate(vecs, label, verbose=False):
    M = np.stack([unit(vecs[n]) for n in NAMES]); good = 0
    for i, n in enumerate(NAMES):
        nb = [NAMES[j] for j in np.argsort(-(M @ M[i]))[1:4]]; kind = CITIES if n in CITIES else COUNTRIES
        good += sum(x in kind for x in nb)
        if verbose: print(f"    {n:11} ~ {nb}")
    an = 0; pairs = (("Uzbekistan", "Kazakhstan"), ("Turkey", "Uzbekistan"), ("Kazakhstan", "Kyrgyzstan"), ("Uzbekistan", "Turkey"))
    for a, c in pairs:
        v = unit(M[NAMES.index(CAPITAL[a])] - M[NAMES.index(a)] + M[NAMES.index(c)])
        r = [NAMES[j] for j in np.argsort(-(M @ v)) if NAMES[j] not in (a, CAPITAL[a], c)][:3]; an += CAPITAL[c] in r
        if verbose: print(f"    {a}:{CAPITAL[a]} :: {c}:? → {r}")
    print(f"  {label:28} соседи {good}/36  аналогии {an}/4"); return good, an

print("--- по отдельности (exp31) ---")
evaluate(link, "связи"); evaluate(bag, "мешок")
print("--- concat [α·связи | (1−α)·мешок] ---")
res = {}
for a in (0.3, 0.5, 0.7):
    cat = {n: np.concatenate([a * unit(link[n]), (1 - a) * unit(bag[n])]) for n in NAMES}
    res[a] = evaluate(cat, f"concat α={a}", verbose=(a == 0.5))
print("--- сумма в одном пространстве (контроль: без concat) ---")
evaluate({n: unit(link[n]) + unit(bag[n]) for n in NAMES}, "сумма")
json.dump({str(k): v for k, v in res.items()}, open(f"{OUT}/exp34_concat_meaning.json", "w"), indent=1)
