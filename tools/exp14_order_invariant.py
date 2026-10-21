"""Эксперимент 14b (архитектор): порядок слов без порядка. Лечение exp13 (соседи 47% vs мешок 95%).
В exp13 ДНК предложения = «кто с кем РЯДОМ» (bind с позицией prev/next) — а в вопросе порядок слов другой
(«У Bazo 45 olma» vs «Сколько olma у Bazo?»), поэтому пары-соседи не совпадают и точность падает с ростом N.
Тезис: связи надо кодировать ПЕРЕСТАНОВОЧНО-ИНВАРИАНТНО — ДНК предложения = Σ_i code(w_i) + Σ_{i<j} code(w_i)⊛code(w_j)
(свёртка коммутативна: a⊛b = b⊛a, значит пара «Bazo–olma» одна и та же в любом порядке и на любом расстоянии).
Это даёт больше различимости, чем мешок (пары!), но не ломается от перестановки.
Второй тезис: слова надо прогонять через сито (буквы → узнать слово → КАНОНИЧЕСКИЙ код): буквенные коды имён
коррелируют (Bazo/Bazora 0.97), а канонические — нет (≤0.1). Сито растёт само: новое слово = новая запись.
Варианты (данные exp13 один в один: 1000 фактов + 2000 реальных ru-предложений шума):
  bag_letters   — мешок буквенных кодов (exp13, 95.4%)
  nbr_letters   — соседи с позицией (exp13, 46.9%)
  bag_canon     — мешок канонических кодов (сито)
  pairs_canon   — Σ слов + Σ неупорядоченных пар (канонические коды)
Метрики: R@1 (нужное предложение первое), ответ-число, и ЗАЗОР = средняя разность cos(top1) − cos(top2)
(чем больше, тем меньше зависимость от порога thr).
"""
import sys, os, json, re, time, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_core import unit
from mini_ai.transformer import clean_question
from tools.facts_unique import make_unique, OUT

D = 2048; rng = np.random.default_rng(0); P = rng.standard_normal((192, D)) / np.sqrt(192)
QWORDS = {"nechta", "сколько", "how", "many", "does", "have", "bor", "у", "ta", "has"}
F = np.fft.fft
def bind(a, b): return unit(np.real(np.fft.ifft(F(a) * F(b))))
P_PREV, P_NEXT = unit(rng.standard_normal(D)), unit(rng.standard_normal(D))
def words(s): return [w for w in re.findall(r"\w+", s.lower()) if w]


def letter_code(word, d=192, ns=(2, 3)):
    w = f"<{clean_question(word)}>"; v = np.zeros(d)
    for n in ns:
        for i in range(len(w) - n + 1):
            h = zlib.crc32(w[i:i + n].encode()); v[h % d] += 1.0 if (h >> 16) & 1 else -1.0
    return v


_L, _C = {}, {}
def code_letters(w):
    if w not in _L: _L[w] = unit(letter_code(w) @ P)
    return _L[w]
def code_canon(w):
    """Сито: слово → канонический код. Новое слово = новая запись (один шаг), без обучения."""
    if w not in _C: _C[w] = unit(np.random.default_rng(zlib.crc32(w.encode())).standard_normal(D))
    return _C[w]


def dna_bag(ws, code): return unit(sum(code(w) for w in ws))
def dna_nbr(ws, code):                        # exp13
    v = np.zeros(D)
    for i, w in enumerate(ws):
        v += code(w)
        if i > 0: v += bind(code(w), bind(code(ws[i - 1]), P_PREV))
        if i < len(ws) - 1: v += bind(code(w), bind(code(ws[i + 1]), P_NEXT))
    return unit(v)
def dna_pairs(ws, code):                      # слова + неупорядоченные пары (⊛ коммутативна)
    v = sum(code(w) for w in ws) * 1.0
    cs = [code(w) for w in ws]
    for i in range(len(cs)):
        for j in range(i + 1, len(cs)): v = v + bind(cs[i], cs[j])
    return unit(v)


VARIANTS = {"bag_letters": (dna_bag, code_letters), "nbr_letters": (dna_nbr, code_letters),
            "bag_canon": (dna_bag, code_canon), "pairs_canon": (dna_pairs, code_canon)}

if __name__ == "__main__":
    noise = [l.strip() for l in open("data/corpus/ru.txt", encoding="utf-8") if 20 < len(l.strip()) < 120][:2000]
    res = {}
    for N in [int(x) for x in os.environ.get("NS", "100,300,1000").split(",")]:
        facts = make_unique(N); sents = [f["fact"] for f in facts] + noise; res[N] = {}
        for name, (dna, code) in VARIANTS.items():
            t0 = time.time(); K = np.stack([dna(words(s), code) for s in sents]); tw = time.time() - t0
            r1 = ans_ok = 0; gaps = []
            for i, f in enumerate(facts):
                qw = [w for w in words(f["q"]) if w not in QWORDS]; s = K @ dna(qw, code)
                top = np.argsort(-s)[:2]; j = int(top[0]); r1 += j == i; gaps.append(float(s[top[0]] - s[top[1]]))
                nums = [w for w in words(sents[j]) if w not in qw and w.isdigit()]
                ans_ok += bool(nums) and nums[0] == f["gold"]
            r = dict(records=len(sents), R1=r1 / N, answer=ans_ok / N, gap=float(np.mean(gaps)), write_s=round(tw, 1))
            res[N][name] = r
            print(f"N={N:4}+2000 шум  {name:12}: R@1 {r['R1']*100:5.1f}%  ответ {r['answer']*100:5.1f}%  зазор top1−top2 {r['gap']:.3f}  запись {tw:.1f}s", flush=True)
    json.dump(res, open(f"{OUT}/exp14_order_invariant.json", "w"), indent=1, ensure_ascii=False)
