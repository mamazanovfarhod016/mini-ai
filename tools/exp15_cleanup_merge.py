"""Эксперимент 15: ДНК⊛ДНК с ОЧИСТКОЙ через сито после каждого слияния (против exp10, где всё тонуло).
Та же задача, что exp10 (без плана, без типов): [X, dost, obj] и [X, dost, dost, obj].
Разница: после каждого слияния пары результат ПРОГОНЯЕТСЯ ЧЕРЕЗ СИТО — заменяется на чистый код ближайшей
известной записи/слова (clean-up memory, Plate 1995), и только чистый код сливается дальше.
Плюс правило выбора пары без типов: берём ту пару, после подстановки ответа которой ОСТАТОК снова
совпадает с какой-то записью (просмотр вперёд, но по памяти, а не по таблице типов).
Выведенные составные ДНК записываются как новые слова (самообучение, как в exp10).
"""
import sys, os, json, random, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_core import unit
from tools.exp3_wordcode import letter_code
from tools.facts_unique import make_unique, OUT

OBJ = ["olma", "kitob", "qalam"]; D = 2048
rng_code = np.random.default_rng(0); _c = {}


def code(w):
    if w not in _c:
        v = letter_code(w)
        if "P" not in _c: _c["P"] = rng_code.standard_normal((len(v), D)) / np.sqrt(len(v))
        _c[w] = unit(v @ _c["P"])
    return _c[w]


F = np.fft.fft
def bind(a, b): return unit(np.real(np.fft.ifft(F(a) * F(b))))


class Memory:
    def __init__(self): self.K, self.V, self.S, self.probes = [], [], [], 0
    def write(self, k, ans): self.K.append(k); self.V.append(code(ans)); self.S.append(ans)
    def lookup(self, k):
        self.probes += 1; s = np.stack(self.K) @ k; i = int(s.argmax()); return float(s[i]), self.S[i]


def solve(mem, parts, thr=0.5, depth=0, trace=None):
    """Без типов и без плана. На каждом шаге: для каждой пары (i,j) — ищем её в памяти; если есть (s≥thr),
    ОЧИЩАЕМ: ответ-строка → чистый код; остаток = [ответ] + прочие; оцениваем остаток ПРОСМОТРОМ ВПЕРЁД:
    если в остатке 1 слово — готово; если 2 — есть ли такая запись; если ≥3 — рекурсия. Выбираем первую пару, где путь доходит до конца."""
    trace = [] if trace is None else trace
    if len(parts) == 1: return parts[0], trace
    if len(parts) == 2:
        s, a = mem.lookup(bind(code(parts[0]), code(parts[1])))
        return (a if s >= thr else None), trace
    if depth > 4: return None, trace
    cands = []
    for i in range(len(parts)):
        for j in range(len(parts)):
            if i == j: continue
            s, a = mem.lookup(bind(code(parts[i]), code(parts[j])))
            if s >= thr: cands.append((s, i, j, a))
    cands.sort(reverse=True)
    for s, i, j, a in cands:
        rest = [a] + [w for t, w in enumerate(parts) if t not in (i, j)]        # a — ЧИСТАЯ строка (сито наоборот уже сделано в lookup)
        ans, trace = solve(mem, rest, thr, depth + 1, trace)
        if ans is not None:
            trace.append(((parts[i], parts[j]), a)); return ans, trace
    return None, trace


res = {}
for N in (100, 300, 1000):
    facts = make_unique(N); names = [f["name"] for f in facts]
    rng = random.Random(7); perm = names[:]; rng.shuffle(perm); friend = dict(zip(names, perm))
    count = {(f["name"], o): str(rng.randrange(2, 99)) for f in facts for o in OBJ}
    mem = Memory()
    for a, b in friend.items(): mem.write(bind(code(a), code("dost")), b)
    for (a, o), n in count.items(): mem.write(bind(code(a), code(o)), n)
    base = len(mem.S); t0 = time.time(); mem.probes = 0
    ok2 = ok3 = 0; qs = []
    for a in names:
        o = rng.choice(OBJ); qs.append((a, o))
        ans, tr = solve(mem, [a, "dost", o]); ok2 += ans == count[(friend[a], o)]
        if ans is not None: mem.write(bind(bind(code(a), code("dost")), code(o)), ans)      # составная ДНК → слово
    p2 = mem.probes / N; mem.probes = 0
    for a, o in qs:
        ans3, _ = solve(mem, [a, "dost", "dost", o]); ok3 += ans3 == count[(friend[friend[a]], o)]
    p3 = mem.probes / N; mem.probes = 0
    again = sum(mem.lookup(bind(bind(code(a), code("dost")), code(o)))[1] == count[(friend[a], o)] for a, o in qs)   # повтор — 1 проба
    old = np.mean([mem.lookup(bind(code(a), code(o)))[1] == count[(a, o)] for a in names for o in OBJ])
    r = {"records": base, "hop2": ok2 / N, "probes2": p2, "hop3": ok3 / N, "probes3": p3, "again_1probe": again / N, "old_intact": float(old), "sec": round(time.time() - t0, 1)}
    res[N] = r
    print(f"N={N:4} ({base} зап.): 2 шага {r['hop2']*100:5.1f}% ({p2:.1f} проб)  3 шага {r['hop3']*100:5.1f}% ({p3:.1f} проб)  "
          f"повтор за 1 пробу {r['again_1probe']*100:5.1f}%  старые целы {old*100:5.1f}%  {r['sec']}s", flush=True)
json.dump(res, open(f"{OUT}/exp15_cleanup_merge.json", "w"), indent=1)
