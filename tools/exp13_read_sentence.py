"""Эксперимент 13: чтение предложения без кортежей и без типов. «Не через железо».
Предложение режется на слова; каждое слово связывается (bind) с кодом своей ПОЗИЦИИ относительно
соседей: p_{-1} (слово перед), p_{+1} (слово после). ДНК предложения = Σ_i code(w_i) ⊛ code(w_{i-1})⊛p_prev
+ code(w_i)⊛code(w_{i+1})⊛p_next … — т.е. память хранит, КТО С КЕМ РЯДОМ. Это всё, что она знает о грамматике.
Вопрос «сколько olma у Бека?» тоже режется, и ищем предложение, у которого больше всего общих пар-соседей.
Ответ: слово из найденного предложения, которого НЕТ в вопросе и которое стоит в той же позиции,
где в вопросе стоит слово-вопрос (nechta / сколько / how many). Никаких типов, никаких шаблонов.
Факты из facts_unique (uz/ru/en перемешаны), 100/300/1000, плюс шум: 2000 случайных реальных предложений из data/corpus/ru.txt.
"""
import sys, os, json, random, re, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_core import unit
from tools.exp3_wordcode import letter_code
from tools.facts_unique import make_unique, OUT

D = 2048; rng_code = np.random.default_rng(0); _c = {}
QWORDS = {"nechta", "сколько", "how", "many", "does", "have", "bor", "у", "ta", "has"}


def code(w):
    if w not in _c:
        v = letter_code(w)
        if "P" not in _c: _c["P"] = rng_code.standard_normal((len(v), D)) / np.sqrt(len(v))
        _c[w] = unit(v @ _c["P"])
    return _c[w]


F = np.fft.fft
def bind(a, b): return unit(np.real(np.fft.ifft(F(a) * F(b))))
P_PREV, P_NEXT = unit(rng_code.standard_normal(D)), unit(rng_code.standard_normal(D))
def words(s): return [w for w in re.findall(r"\w+", s.lower()) if w]


def dna(ws, skip=()):
    """ДНК предложения: сумма bind(слово, сосед⊛направление). Слова из skip (вопросительные) не входят."""
    v = np.zeros(D); keep = [w for w in ws if w not in skip]
    for i, w in enumerate(keep):
        v += code(w)                                           # само слово (мешок слов)
        if i > 0: v += bind(code(w), bind(code(keep[i - 1]), P_PREV))
        if i < len(keep) - 1: v += bind(code(w), bind(code(keep[i + 1]), P_NEXT))
    return unit(v)


res = {}
noise = [l.strip() for l in open("data/corpus/ru.txt", encoding="utf-8") if 20 < len(l.strip()) < 120][:2000]
for N in (100, 300, 1000):
    facts = make_unique(N)
    sents = [f["fact"] for f in facts] + noise
    t0 = time.time()
    K = np.stack([dna(words(s)) for s in sents]); t_write = time.time() - t0
    r1 = r1_bag = ans_ok = 0
    Kbag = np.stack([unit(sum(code(w) for w in words(s))) for s in sents])
    for i, f in enumerate(facts):
        qw = words(f["q"]); q = dna(qw, skip=QWORDS)
        j = int((K @ q).argmax()); r1 += j == i
        jb = int((Kbag @ unit(sum(code(w) for w in qw if w not in QWORDS))).argmax()); r1_bag += jb == i
        # ответ: слово найденного предложения, которого нет в вопросе и которое — число
        cand = [w for w in words(sents[j]) if w not in qw]
        nums = [w for w in cand if w.isdigit()]
        ans_ok += bool(nums) and nums[0] == f["gold"]
    r = {"records": len(sents), "R1_dna": r1 / N, "R1_bag": r1_bag / N, "answer": ans_ok / N, "write_s": round(t_write, 1)}
    res[N] = r
    print(f"N={N:4} фактов + 2000 шум: нашла предложение (соседи) {r['R1_dna']*100:5.1f}%  (мешок слов) {r['R1_bag']*100:5.1f}%  "
          f"ответ-число {r['answer']*100:5.1f}%  запись {r['write_s']}s", flush=True)
json.dump(res, open(f"{OUT}/exp13_read_sentence.json", "w"), indent=1)
