"""Эксперимент 8: слияние с ролями (binding) вместо суммы.
Сумма кодов теряет «кто кому кто». Связывание: bind(a,b) = круговая свёртка (Plate 1995, HRR) —
даёт новый уникальный код; unbind(c, a) ≈ b. Запись «X dost → Y» становится X⊛dost ; «X obj → N» = X⊛obj.
Запрос [X, dost, obj] без плана: память сама ищет, какая пара слов связывается в известный ключ,
подставляет ответ и продолжает. Плюс новая способность: ОБРАТНЫЙ вопрос «чей друг Y?» = unbind(ключ, dost)
— записи с такой стороны не было, но из связанного кода её можно вытащить.
"""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_core import unit
from tools.exp3_wordcode import letter_code
from tools.facts_unique import make_unique, OUT

OBJ = ["olma", "kitob", "qalam"]
D = 1024
rng_code = np.random.default_rng(0)
_cache = {}


def code(w):
    """Код слова: буквенный хэш → расширяем до D случайной фиксированной проекцией (сито, не учится)."""
    if w not in _cache:
        v = letter_code(w)
        if "P" not in _cache: _cache["P"] = rng_code.standard_normal((len(v), D)) / np.sqrt(len(v))
        _cache[w] = unit(v @ _cache["P"])
    return _cache[w]


def bind(a, b): return unit(np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b))))
def unbind(c, a): return unit(np.real(np.fft.ifft(np.fft.fft(c) * np.conj(np.fft.fft(a)))))


class Memory:
    def __init__(self): self.K, self.V, self.S, self.KW = [], [], [], []
    def write(self, words, answer):
        self.K.append(bind(code(words[0]), code(words[1]))); self.V.append(code(answer)); self.S.append(answer); self.KW.append(tuple(words))
    def best_key(self, k):
        s = np.stack(self.K) @ k; i = int(s.argmax()); return float(s[i]), self.KW[i], self.S[i]
    def best(self, words): return self.best_key(bind(code(words[0]), code(words[1])))


def solve(mem, words, thr=0.5, max_steps=5):
    words = list(words); trace = []
    for _ in range(max_steps):
        if len(words) == 2:
            s, kw, ans = mem.best(words); trace.append((tuple(words), ans)); return (ans if s >= thr else None), trace
        cand = [(mem.best([words[i], words[j]])[0], i, j) for i in range(len(words)) for j in range(len(words)) if i != j]
        s2, i, j = max(cand)
        if s2 < thr: return None, trace
        a2 = mem.best([words[i], words[j]])[2]; trace.append(((words[i], words[j]), a2))
        words = [a2] + [w for k, w in enumerate(words) if k not in (i, j)]
    return None, trace


res = {}
for N in (100, 300, 1000):
    facts = make_unique(N); names = [f["name"] for f in facts]
    rng = random.Random(7); perm = names[:]; rng.shuffle(perm); friend = dict(zip(names, perm))
    count = {(f["name"], o): str(rng.randrange(2, 99)) for f in facts for o in OBJ}
    mem = Memory()
    for a, b in friend.items(): mem.write([a, "dost"], b)
    for (a, o), n in count.items(): mem.write([a, o], n)
    base = len(mem.S)
    T_names = np.stack([code(n) for n in names])
    ok1 = ok2 = ok3 = rev = 0; derived = []
    for a in names:
        o = rng.choice(OBJ)
        ok1 += mem.best([a, o])[2] == count[(a, o)]
        ans, tr = solve(mem, [a, "dost", o]); ok2 += ans == count[(friend[a], o)]
        if ans is not None: derived.append(([a, "dost", o], ans, count[(friend[a], o)]))
        ans3, _ = solve(mem, [a, "dost", "dost", o]); ok3 += ans3 == count[(friend[friend[a]], o)]
        # обратный вопрос: чей друг friend[a]? — ищем запись со значением friend[a], из её ключа вытаскиваем имя
        y = code(friend[a]); vi = int((np.stack(mem.V) @ y).argmax()); who = unbind(mem.K[vi], code("dost"))
        rev += names[int((T_names @ who).argmax())] == a
    r = {"records": base, "hop1": ok1 / N, "hop2_noplan": ok2 / N, "hop3_noplan": ok3 / N, "reverse": rev / N}
    res[N] = r
    print(f"N={N:4} ({base} записей): 1 шаг {r['hop1']*100:5.1f}%  2 шага без плана {r['hop2_noplan']*100:5.1f}%  "
          f"3 шага {r['hop3_noplan']*100:5.1f}%  | обратный вопрос (чей друг?) {r['reverse']*100:5.1f}%", flush=True)
json.dump(res, open(f"{OUT}/exp8_binding.json", "w"), indent=1)
