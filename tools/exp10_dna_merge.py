"""Эксперимент 10: ДНК-слияние. Предложение = одна ДНК, собранная слиянием (bind) частей.
Память хранит ДНК → значение. Запрос «сколько olma у друга Бека» = ДНК(Бек)⊛ДНК(dost)⊛ДНК(olma) — ОДНО слияние.
Если такой ДНК нет (сходство < thr): раскручиваем — отделяем часть (unbind по каждой известной части),
ищем остаток в памяти, подставляем найденное и сливаем снова. Найденные составные ДНК («друг Бека» = Зора)
ЗАПИСЫВАЕМ как новые слова — второй раз это один шаг. Типы не задаём руками: ограничение только одно —
unbind пробуем по частям запроса, выбираем ту, после которой остаток лучше всего совпал с памятью.
"""
import sys, os, json, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_core import unit
from tools.exp3_wordcode import letter_code
from tools.facts_unique import make_unique, OUT

OBJ = ["olma", "kitob", "qalam"]
D = int(os.environ.get("D", 2048))
rng_code = np.random.default_rng(0)
_cache = {}


def code(w):
    if w not in _cache:
        v = letter_code(w)
        if "P" not in _cache: _cache["P"] = rng_code.standard_normal((len(v), D)) / np.sqrt(len(v))
        _cache[w] = unit(v @ _cache["P"])
    return _cache[w]


F = np.fft.fft; iF = lambda z: unit(np.real(np.fft.ifft(z)))
def bind(*vs):
    z = np.ones(D, dtype=complex)
    for v in vs: z = z * F(v)
    return iF(z)
def unbind(c, a): return iF(F(c) * np.conj(F(a)))


class DNAMemory:
    def __init__(self): self.K, self.V, self.S = [], [], []
    def write(self, dna, answer):
        self.K.append(dna); self.V.append(code(answer)); self.S.append(answer)
    def lookup(self, dna):
        s = np.stack(self.K) @ dna; i = int(s.argmax()); return float(s[i]), self.S[i]


def solve(mem, parts, thr=0.35, depth=0, max_depth=3, trace=None):
    """parts — список слов (строк). Сначала одно слияние всех частей. Если нет в памяти — раскрутка."""
    trace = [] if trace is None else trace
    dna = bind(*[code(w) for w in parts])
    s, ans = mem.lookup(dna)
    if s >= thr: trace.append(("hit", tuple(parts), ans, round(s, 2))); return ans, trace
    if depth >= max_depth or len(parts) < 3: return None, trace
    # раскрутка: убираем одну часть; остаток (слияние остальных) ищем в памяти; лучший — подставляем
    best = None
    for i in range(len(parts)):
        rest = parts[:i] + parts[i + 1:]
        # остаток тоже может быть составным — ищем его как слияние ПАРЫ (минимальная единица)
        for j in range(len(rest)):
            for k in range(j + 1, len(rest)):
                s2, a2 = mem.lookup(bind(code(rest[j]), code(rest[k])))
                if best is None or s2 > best[0]: best = (s2, i, j, k, a2, rest)
    s2, i, j, k, a2, rest = best
    if s2 < thr: return None, trace
    trace.append(("unroll", (rest[j], rest[k]), a2, round(s2, 2)))
    new_parts = [a2] + [w for t, w in enumerate(rest) if t not in (j, k)] + [parts[i]]
    # дедуп: parts[i] был «убран» только чтобы найти пару; возвращаем его
    new_parts = list(dict.fromkeys(new_parts))
    ans, trace = solve(mem, new_parts, thr, depth + 1, max_depth, trace)
    if ans is not None:
        mem.write(dna, ans)                                         # познанное: составная ДНК → ответ
        mem.write(bind(code(rest[j]), code(rest[k])), a2)           # (уже есть, но безвредно)
    return ans, trace


res = {}
for N in (100, 300, 1000):
    facts = make_unique(N); names = [f["name"] for f in facts]
    rng = random.Random(7); perm = names[:]; rng.shuffle(perm); friend = dict(zip(names, perm))
    count = {(f["name"], o): str(rng.randrange(2, 99)) for f in facts for o in OBJ}
    mem = DNAMemory()
    for a, b in friend.items(): mem.write(bind(code(a), code("dost")), b)
    for (a, o), n in count.items(): mem.write(bind(code(a), code(o)), n)
    base = len(mem.S); t0 = time.time()
    ok2 = ok3 = 0; qs = []
    for a in names:
        o = rng.choice(OBJ); qs.append((a, o))
        ans, _ = solve(mem, [a, "dost", o]); ok2 += ans == count[(friend[a], o)]
    n_after2 = len(mem.S)
    ok2_again = sum(solve(mem, [a, "dost", o])[0] == count[(friend[a], o)] for a, o in qs)   # второй раз — один шаг?
    hits1 = sum(solve(mem, [a, "dost", o])[1][0][0] == "hit" for a, o in qs)
    for a, o in qs:
        ans3, _ = solve(mem, [a, "dost", "dost", o]); ok3 += ans3 == count[(friend[friend[a]], o)]
    old = np.mean([mem.lookup(bind(code(a), code(o)))[1] == count[(a, o)] for a in names for o in OBJ])
    r = {"D": D, "records": base, "hop2": ok2 / N, "hop2_again": ok2_again / N, "hop2_onehit": hits1 / N, "hop3": ok3 / N,
         "new_dna": len(mem.S) - base, "old_intact": float(old), "sec": round(time.time() - t0, 1)}
    res[N] = r
    print(f"N={N:4} ({base} зап., D={D}): 2 шага {r['hop2']*100:5.1f}%  → повторно {r['hop2_again']*100:5.1f}% (из них за 1 шаг {r['hop2_onehit']*100:5.1f}%)  "
          f"3 шага {r['hop3']*100:5.1f}%  | новых ДНК {r['new_dna']}, старые целы {old*100:5.1f}%  {r['sec']}s", flush=True)
json.dump(res, open(f"{OUT}/exp10_dna_D{D}.json", "w"), indent=1)
