"""Эксперимент 14a (архитектор): почему ДНК-слияние (exp10) тонет в шуме и что его спасает.
Гипотеза: шум exp10 — НЕ шум свёртки (HRR), а корреляция буквенных кодов имён: имена из 20 слогов
(«Bazo», «Bazora») дают косинус до 0.97, среднее по ближайшему соседу 0.74; свёртка переносит эту
корреляцию на составные ДНК (X⊛dost⊛olma vs X'⊛dost⊛olma ≈ 0.68) → ложные «hit» при thr=0.35.
Лечение = «очистка через сито»: буквенный код нужен только чтобы УЗНАТЬ слово (ближайшее известное),
а для слияния берётся КАНОНИЧЕСКИЙ код слова (случайный фиксированный вектор, cos ≈ 0.1 между разными).
Второй дефект exp10 — монетка при раскрутке (как в exp8) → лечится типами + просмотром вперёд (exp9).
Варианты (данные exp10 один в один, D=2048):
  letters        — exp10 как есть (буквенные коды, раскрутка без типов, thr=0.35)
  letters+types  — буквенные коды, раскрутка по типам (просмотр вперёд)       → ложные hit остаются?
  canon          — канонические коды (очистка ситом), раскрутка без типов      → монетка остаётся?
  canon+types    — очистка ситом + типы                                         → ожидаем ~100%
"""
import sys, os, json, random, time, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_core import unit
from mini_ai.transformer import clean_question
from tools.facts_unique import make_unique, OUT


def letter_code(word, d=192, ns=(2, 3)):
    """Копия tools/exp3_wordcode.letter_code (без загрузки модели): хэш n-грамм букв → ±1 в d измерений."""
    w = f"<{clean_question(word)}>"; v = np.zeros(d)
    for n in ns:
        for i in range(len(w) - n + 1):
            h = zlib.crc32(w[i:i + n].encode()); v[h % d] += 1.0 if (h >> 16) & 1 else -1.0
    return v

OBJ = ["olma", "kitob", "qalam"]
D = int(os.environ.get("D", 2048)); THR = 0.35
rng_code = np.random.default_rng(0)
P = rng_code.standard_normal((192, D)) / np.sqrt(192)
_lc = {}


def code_letters(w):
    if w not in _lc: _lc[w] = unit(letter_code(w) @ P)
    return _lc[w]


class Sieve:
    """Сито в обе стороны: буква→вектор (узнать слово) и слово→канонический код (чистый, почти ортогональный).
    cleanup(v): ближайшее известное слово по буквенному коду → его канонический код."""
    def __init__(self): self.words, self.L, self.C, self.canon, self._LM = [], [], [], {}, None
    def add(self, w):
        if w not in self.canon:
            self.canon[w] = unit(np.random.default_rng(zlib.crc32(w.encode())).standard_normal(D))
            self.words.append(w); self.L.append(code_letters(w)); self.C.append(self.canon[w]); self._LM = None
    def cleanup(self, letters_vec):
        if self._LM is None: self._LM = np.stack(self.L)
        return self.C[int((self._LM @ letters_vec).argmax())]


F = np.fft.fft; iF = lambda z: unit(np.real(np.fft.ifft(z)))
def bind(*vs):
    z = np.ones(D, dtype=complex)
    for v in vs: z = z * F(v)
    return iF(z)


class DNAMemory:
    def __init__(self): self.K, self.S, self._KM, self.probes = [], [], None, 0
    def write(self, dna, answer): self.K.append(dna); self.S.append(answer); self._KM = None
    def lookup(self, dna):
        if self._KM is None: self._KM = np.stack(self.K)
        self.probes += 1; s = self._KM @ dna; i = int(s.argmax()); return float(s[i]), self.S[i]


GRAMMAR = {("name", "rel"): "name", ("name", "obj"): "num"}
def allowed(a, b): return GRAMMAR.get((a, b)) or GRAMMAR.get((b, a))
def reducible(ts):
    if len(ts) <= 1: return True
    for i in range(len(ts)):
        for j in range(i + 1, len(ts)):
            r = allowed(ts[i], ts[j])
            if r and reducible([r] + [t for k, t in enumerate(ts) if k not in (i, j)]): return True
    return False


def solve(mem, code, parts, types, typed, depth=0, max_depth=4):
    """parts — слова, types — их типы (используются только при typed=True).
    1) слить всё в одну ДНК и спросить память; 2) если нет — раскрутить: слить ПАРУ, подставить ответ, повторить."""
    dna = bind(*[code(w) for w in parts])
    s, ans = mem.lookup(dna)
    if s >= THR: return ans, "hit"
    if depth >= max_depth or len(parts) < 3: return None, "dead"
    cands = []
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            if typed:
                r = allowed(types[i], types[j])
                if not r: continue
                rest_t = [t for k, t in enumerate(types) if k not in (i, j)]
                if not reducible([r] + rest_t): continue
            else:
                r = "?"
            cands.append((i, j, r))
    best = None
    for i, j, r in cands:
        s2, a2 = mem.lookup(bind(code(parts[i]), code(parts[j])))
        if best is None or s2 > best[0]: best = (s2, i, j, r, a2)
    if best is None or best[0] < THR: return None, "dead"
    s2, i, j, r, a2 = best
    np_ = [a2] + [w for k, w in enumerate(parts) if k not in (i, j)]
    nt_ = [r] + [t for k, t in enumerate(types) if k not in (i, j)]
    ans, how = solve(mem, code, np_, nt_, typed, depth + 1, max_depth)
    if ans is not None: mem.write(dna, ans)       # познанное: составная ДНК → ответ (самообучение)
    return ans, how


def run(N, variant):
    facts = make_unique(N); names = [f["name"] for f in facts]
    rng = random.Random(7); perm = names[:]; rng.shuffle(perm); friend = dict(zip(names, perm))
    count = {(f["name"], o): str(rng.randrange(2, 99)) for f in facts for o in OBJ}
    sieve = Sieve()
    for w in names + ["dost"] + OBJ + [str(n) for n in range(2, 99)]: sieve.add(w)
    if variant.startswith("canon"): code = lambda w: sieve.cleanup(code_letters(w))   # буквы → узнать → чистый код
    else: code = code_letters
    typed = variant.endswith("types")
    mem = DNAMemory()
    for a, b in friend.items(): mem.write(bind(code(a), code("dost")), b)
    for (a, o), n in count.items(): mem.write(bind(code(a), code(o)), n)
    base = len(mem.S); t0 = time.time()
    qs = [(a, rng.choice(OBJ)) for a in names]
    T = ["name", "rel", "obj"]
    ok2 = sum(solve(mem, code, [a, "dost", o], T, typed)[0] == count[(friend[a], o)] for a, o in qs)
    p0 = mem.probes
    again = [solve(mem, code, [a, "dost", o], T, typed) for a, o in qs]
    ok2_again = sum(r[0] == count[(friend[a], o)] for r, (a, o) in zip(again, qs))
    onehit = sum(r[1] == "hit" for r in again); probes_again = (mem.probes - p0) / N
    ok3 = sum(solve(mem, code, [a, "dost", "dost", o], ["name", "rel", "rel", "obj"], typed)[0] == count[(friend[friend[a]], o)] for a, o in qs)
    old = np.mean([mem.lookup(bind(code(a), code(o)))[1] == count[(a, o)] for a in names for o in OBJ])
    r = dict(records=base, hop2=ok2 / N, hop2_again=ok2_again / N, hop2_onehit=onehit / N, hop3=ok3 / N,
             new_dna=len(mem.S) - base, old_intact=float(old), probes_again=probes_again, sec=round(time.time() - t0, 1))
    print(f"  {variant:14} N={N:4} ({base}): 2 шага {r['hop2']*100:5.1f}%  повторно {r['hop2_again']*100:5.1f}% (за 1 hit {r['hop2_onehit']*100:5.1f}%)  "
          f"3 шага {r['hop3']*100:5.1f}%  | новых ДНК {r['new_dna']}, старые целы {old*100:5.1f}%, проб/вопрос {r['probes_again']:.1f}  {r['sec']}s", flush=True)
    return r


if __name__ == "__main__":
    Ns = [int(x) for x in os.environ.get("NS", "100,300,1000").split(",")]
    variants = os.environ.get("VARIANTS", "letters,letters+types,canon,canon+types").split(",")
    tag = os.environ.get("TAG", "")
    res = {}
    for N in Ns:
        print(f"N={N}", flush=True); res[N] = {}
        for v in variants:
            res[N][v] = run(N, v)
    json.dump(res, open(f"{OUT}/exp14_merge_cleanup{tag}.json", "w"), indent=1, ensure_ascii=False)
