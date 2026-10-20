"""Эксперимент 9: «ДНК типов» (идея Фархода 20.10).
Код слова = [код ТИПА | код СОДЕРЖАНИЯ] (или bind(тип, содержание)). У каждого типа есть «грамматика связей»:
с каким типом он связывается и какой тип даёт:  name⊛rel → name ;  name⊛obj → num.  num ни с чем не связывается.
Решатель без плана: рассматривает только ДОПУСТИМЫЕ по типам пары, подставляет результат С ЕГО ТИПОМ (тип берётся
из таблицы связи, не из словаря) и — ключевое — смотрит ПО ТИПАМ (не трогая память), можно ли остаток запроса
довести до одного слова. Это «план из грамматики», а не заданный план.

Варианты:
  exp8            — контроль, как в exp8 (без типов, без просмотра вперёд)
  exp8_backtrack  — контроль без типов: перебор с откатом через ПРОБЫ ПАМЯТИ (дорого)
  bind_nolook     — тип вбит в код (bind), фильтр по таблице, но без просмотра вперёд
  bind_look       — тип вбит в код (bind) + фильтр + просмотр вперёд по типам
  concat_look     — код = [тип | содержание] (конкатенация) + фильтр + просмотр вперёд
  field_learned   — тип как ОТДЕЛЬНОЕ поле записи (косинус по полю); таблица связей НЕ задана руками,
                    а выводится из записей памяти (какие пары типов встречались и что давали)
Метрики на тех же фактах, что exp7/8: 1 шаг, 2 шага без плана, 3 шага, обратный вопрос «чей друг Y?»,
самозапись выводов (читаются в 1 шаг?) и целостность старых фактов; плюс число обращений к памяти на вопрос.
"""
import sys, os, json, random, time, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_core import unit
from tools.exp3_wordcode import letter_code
from tools.facts_unique import make_unique, OUT

OBJ = ["olma", "kitob", "qalam"]
D = 1024
DT = 128                      # для варианта concat: длина части «тип», содержание = D-DT
LOG = None


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True)
    if LOG: LOG.write(s + "\n"); LOG.flush()


# ---------- коды (сито: ничему не учится) ----------
_rng = np.random.default_rng(0)
_P = {}


def proj(dim):
    if dim not in _P: _P[dim] = _rng.standard_normal((192, dim)) / np.sqrt(192)
    return _P[dim]


_cc = {}


def content(w, dim=D):
    """Код содержания: буквенный хэш → случайная фиксированная проекция в dim."""
    k = (w, dim)
    if k not in _cc: _cc[k] = unit(letter_code(w) @ proj(dim))
    return _cc[k]


_tc = {}


def tcode(t, dim=D):
    """Код типа — случайный фиксированный вектор («ДНК» типа)."""
    k = (t, dim)
    if k not in _tc: _tc[k] = unit(np.random.default_rng(zlib.crc32(t.encode())).standard_normal(dim))
    return _tc[k]


def bind(a, b): return unit(np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b))))
def unbind(c, a): return unit(np.real(np.fft.ifft(np.fft.fft(c) * np.conj(np.fft.fft(a)))))


# ---------- типы и грамматика связей ----------
# Словарь типов нужен только для слов ЗАПРОСА и фактов; тип ответа решатель берёт из таблицы.
GRAMMAR = {("name", "rel"): "name", ("name", "obj"): "num"}      # что с чем связывается и что даёт


def allowed(ti, tj):
    """Таблица связей симметрична по порядку слов (⊛ коммутативна)."""
    return GRAMMAR.get((ti, tj)) or GRAMMAR.get((tj, ti))


def reducible(types, gram=allowed):
    """Можно ли ПО ТИПАМ свести список к одному слову? (без обращения к памяти)"""
    if len(types) <= 1: return True
    for i in range(len(types)):
        for j in range(i + 1, len(types)):
            r = gram(types[i], types[j])
            if r and reducible([r] + [t for k, t in enumerate(types) if k not in (i, j)], gram): return True
    return False


class TW:
    """Типизированное слово: (тип, строка)."""
    __slots__ = ("t", "w")
    def __init__(self, t, w): self.t, self.w = t, w
    def __repr__(self): return f"{self.w}:{self.t}"


# ---------- кодировщики слова (4 способа) ----------
def enc_plain(x):  return content(x.w)                                     # exp8: только содержание
def enc_bind(x):   return bind(tcode(x.t), content(x.w))                    # ДНК вбита в код
def enc_concat(x): return unit(np.concatenate([tcode(x.t, DT), content(x.w, D - DT)]))   # [тип | содержание]


class Memory:
    """Список записей. Ключ = bind(код w1, код w2) (для 3+ слов — свёртка по цепочке).
    Дополнительно хранит поле типов (для варианта field) и тип ответа."""
    def __init__(self, enc):
        self.enc = enc; self.K, self.V, self.S, self.KW, self.T, self.TA = [], [], [], [], [], []
        self.probes = 0

    def key(self, words):
        k = self.enc(words[0])
        for w in words[1:]: k = bind(k, self.enc(w))
        return k

    def tfield(self, words): return unit(np.concatenate([tcode(w.t, DT) for w in words]))

    def write(self, words, answer):
        self.K.append(self.key(words)); self.V.append(self.enc(answer)); self.S.append(answer.w)
        self.KW.append(tuple(w.w for w in words)); self.T.append(tuple(w.t for w in words)); self.TA.append(answer.t)
        self._KM = None

    def KM(self):
        if getattr(self, "_KM", None) is None: self._KM = np.stack(self.K)
        return self._KM

    def best_key(self, k):
        self.probes += 1
        s = self.KM() @ k; i = int(s.argmax()); return float(s[i]), i

    def best(self, words):
        s, i = self.best_key(self.key(words)); return s, self.S[i], self.TA[i]


# ---------- решатели ----------
def solve_exp8(mem, words, thr=0.5, max_steps=5):
    """Как в exp8: любая пара, лучшая по памяти; тип результата — не известен (берём из записи только строку)."""
    words = list(words)
    for _ in range(max_steps):
        if len(words) == 2:
            s, ans, ta = mem.best(words); return ans if s >= thr else None
        cand = [(mem.best([words[i], words[j]])[0], i, j) for i in range(len(words)) for j in range(len(words)) if i != j]
        s2, i, j = max(cand)
        if s2 < thr: return None
        a2 = mem.best([words[i], words[j]])[1]
        words = [TW("?", a2)] + [w for k, w in enumerate(words) if k not in (i, j)]
    return None


def solve_backtrack(mem, words, thr=0.5):
    """Контроль без типов: перебор пар с откатом, проверка — только пробой памяти (дорого)."""
    words = list(words)
    if len(words) == 2:
        s, ans, ta = mem.best(words); return ans if s >= thr else None
    cand = []
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            s, a2, ta = mem.best([words[i], words[j]])
            if s >= thr: cand.append((s, i, j, a2))
    for s, i, j, a2 in sorted(cand, reverse=True):
        r = solve_backtrack(mem, [TW("?", a2)] + [w for k, w in enumerate(words) if k not in (i, j)], thr)
        if r is not None: return r
    return None


def solve_typed(mem, words, gram=allowed, look=True, thr=0.5, max_steps=5):
    """ДНК типов: пары только из допустимых по таблице; результат получает тип ИЗ ТАБЛИЦЫ;
    при look=True сначала отбрасываем пары, после которых остаток по типам не сводится (просмотр вперёд без памяти)."""
    words = list(words)
    for _ in range(max_steps):
        pairs = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                r = gram(words[i].t, words[j].t)
                if not r: continue
                rest = [w.t for k, w in enumerate(words) if k not in (i, j)]
                if look and not reducible([r] + rest, gram): continue
                pairs.append((i, j, r))
        if not pairs: return None
        best = None
        for i, j, r in pairs:
            s, a2, ta = mem.best([words[i], words[j]])
            if best is None or s > best[0]: best = (s, i, j, r, a2)
        s, i, j, r, a2 = best
        if s < thr: return None
        if len(words) == 2: return a2
        words = [TW(r, a2)] + [w for k, w in enumerate(words) if k not in (i, j)]
    return None


def make_field_solver(mem):
    """Вариант «тип как отдельное поле»: грамматика НЕ задана руками — выводится из записей памяти:
    пара типов допустима, если косинус поля типов с каким-то записанным полем ≈ 1; тип результата — тип ответа той записи."""
    TF = np.stack([mem.tfield([TW(t, "") for t in ts]) for ts in mem.T])
    def gram(ti, tj):
        for a, b in ((ti, tj), (tj, ti)):
            f = mem.tfield([TW(a, ""), TW(b, "")]); s = TF @ f; i = int(s.argmax())
            if s[i] > 0.99: return mem.TA[i]
        return None
    return lambda words: solve_typed(mem, words, gram=gram, look=True)


# ---------- данные (те же, что exp7/exp8) ----------
def build(N):
    facts = make_unique(N); names = [f["name"] for f in facts]
    rng = random.Random(7); perm = names[:]; rng.shuffle(perm); friend = dict(zip(names, perm))
    count = {(f["name"], o): str(rng.randrange(2, 99)) for f in facts for o in OBJ}
    qs = [(a, rng.choice(OBJ)) for a in names]           # тот же порядок случайностей, что в exp8
    return names, friend, count, qs


def run_variant(name, enc, solver_kind, N, names, friend, count, qs):
    t0 = time.time()
    mem = Memory(enc)
    nm = lambda s: TW("name", s); rel = TW("rel", "dost"); ob = lambda o: TW("obj", o); num = lambda n: TW("num", n)
    for a, b in friend.items(): mem.write([nm(a), rel], nm(b))
    for (a, o), n in count.items(): mem.write([nm(a), ob(o)], num(n))
    base = len(mem.S)
    if solver_kind == "exp8": solve = lambda ws: solve_exp8(mem, ws)
    elif solver_kind == "backtrack": solve = lambda ws: solve_backtrack(mem, ws)
    elif solver_kind == "typed_nolook": solve = lambda ws: solve_typed(mem, ws, look=False)
    elif solver_kind == "typed_look": solve = lambda ws: solve_typed(mem, ws, look=True)
    elif solver_kind == "field": solve = make_field_solver(mem)
    T_names = np.stack([enc(nm(n)) for n in names]); VM = np.stack(mem.V)
    ok1 = ok2 = ok3 = rev = 0; derived = []; probes2 = []
    for a, o in qs:
        ok1 += mem.best([nm(a), ob(o)])[1] == count[(a, o)]
        p0 = mem.probes; ans = solve([nm(a), rel, ob(o)]); probes2.append(mem.probes - p0)
        ok2 += ans == count[(friend[a], o)]
        if ans is not None: derived.append(([nm(a), rel, ob(o)], ans, count[(friend[a], o)]))
        ans3 = solve([nm(a), rel, rel, ob(o)]); ok3 += ans3 == count[(friend[friend[a]], o)]
        # обратный вопрос: чей друг friend[a]? запись «? dost → friend[a]» ищем по ЗНАЧЕНИЮ, из ключа вытаскиваем имя
        vi = int((VM @ enc(nm(friend[a]))).argmax()); who = unbind(mem.K[vi], enc(rel))
        rev += names[int((T_names @ who).argmax())] == a
    # самозапись: выведенное → в память как новый факт (ключ из трёх слов), читается ли в 1 шаг? старые целы?
    for ws, ans, gold in derived: mem.write(ws, num(ans))
    new_ok = 0
    for ws, ans, gold in derived:
        s, a1, _ = mem.best(ws); new_ok += (a1 == gold and s >= 0.9)
    old = float(np.mean([mem.best([nm(a), ob(o)])[1] == count[(a, o)] for a in names for o in OBJ]))
    r = {"records": base, "hop1": ok1 / N, "hop2_noplan": ok2 / N, "hop3_noplan": ok3 / N, "reverse": rev / N,
         "derived_written": len(derived), "derived_1step": new_ok / max(1, len(derived)), "old_intact": old,
         "probes_per_hop2": float(np.mean(probes2)), "sec": round(time.time() - t0, 1)}
    log(f"  {name:15} N={N:4} ({base}): 1 шаг {r['hop1']*100:5.1f}%  2 шага {r['hop2_noplan']*100:5.1f}%  3 шага {r['hop3_noplan']*100:5.1f}%  "
        f"обратный {r['reverse']*100:5.1f}%  | новых {len(derived)}, в 1 шаг {r['derived_1step']*100:5.1f}%, старые целы {old*100:5.1f}%  "
        f"| проб памяти/вопрос {r['probes_per_hop2']:.1f}  ({r['sec']} s)")
    return r


VARIANTS = [("exp8", enc_plain, "exp8"), ("exp8_backtrack", enc_plain, "backtrack"),
            ("bind_nolook", enc_bind, "typed_nolook"), ("bind_look", enc_bind, "typed_look"),
            ("concat_look", enc_concat, "typed_look"), ("field_learned", enc_plain, "field")]

if __name__ == "__main__":
    LOG = open(f"{OUT}/exp9.log", "w")
    res = {}
    for N in (100, 300, 1000):
        names, friend, count, qs = build(N); log(f"N={N}")
        res[N] = {}
        for name, enc, kind in VARIANTS:
            res[N][name] = run_variant(name, enc, kind, N, names, friend, count, qs)
    json.dump(res, open(f"{OUT}/exp9_dna_types.json", "w"), indent=1, ensure_ascii=False)
    LOG.close()
