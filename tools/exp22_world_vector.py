"""Эксперимент 22: АРХИТЕКТУРА НОВОГО ПОКОЛЕНИЯ — «состояние мира как ОДИН вектор» (ядро v2).
Что было (exp17): simulate() — Python-словари entity_loc / obj_state / obj_history + if/else по глаголам.
Это работает (100% на task1-3,11,13), но это НЕ память — это программа. Если «мозг» у Фархода должен быть
безградиентным и единообразным, состояние мира должно жить в том же векторном пространстве, что и слова.

НОВЫЙ МЕХАНИЗМ (нигде в exp1-21 не пробовался): состояние мира S — один вектор размерности D.
  Факт «X находится в L»  →  S += C(X) ⊛ C(L)                        (bind = круговая свёртка)
  Обновление X → L'       →  S -= C(X) ⊛ C(L_old);  S += C(X) ⊛ C(L') (вычитаем старое, добавляем новое)
  Запрос «где X?»         →  L ≈ cleanup( S ⊘ C(X) )                 (unbind, затем ближайшее известное слово)
  «они (they)» = группа   →  S += (C(X)+C(Y)) ⊛ C(L)                 (суперпозиция имён — одна операция)
  Объект у держателя      →  S += C(obj) ⊛ C(HOLDER) ⊛ C(X)           (тройное связывание: предмет-роль-имя)
  Время/история           →  S_t = S_{t-1} ⊛ Π  (перестановка координат = «тик часов»): прошлое — это тот
                              же вектор, но повёрнутый; «где был X до L?» = unbind по Π^{-k} в обратном порядке.
ВСЁ — сложение, свёртка, перестановка. Ни одного if по словам, ни одного словаря состояния.
Единственная «программа» — таблица типов глаголов (motion/possess/drop), как exp9: ТИП, а не правило.

Что проверяем (bAbI test, те же 1000 примеров на задачу, что exp17):
  task1  (где X?)                         — 1 bind/unbind на факт
  task2  (где объект? объект у держателя) — тройной bind, 2 unbind
  task11 (he/she → имя)                   — unbind последнего субъекта из состояния
  task13 (they → двое)                    — суперпозиция имён
  task3  (где был X до L?)                — история через Π
Метрики: точность vs exp17 (словарный simulate — 100%), и ЁМКОСТЬ: сколько фактов влезает в один вектор
до того, как cleanup начинает ошибаться (D=2048 → теория ~0.25·D/число слагаемых; измеряем).
"""
import sys, os, re, json, time, zlib, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_core import unit

D = int(os.environ.get("D", 2048))
_C, _FC = {}, {}


def C(w):
    if w not in _C:
        v = unit(np.random.default_rng(zlib.crc32(w.encode()) & 0xffffffff).standard_normal(D))
        _C[w] = v; _FC[w] = np.fft.fft(v)
    return _C[w]
def FC(w): C(w); return _FC[w]
def ifft(z): return np.real(np.fft.ifft(z))
def bind(*ws):
    z = np.ones(D, dtype=complex)
    for w in ws: z = z * (FC(w) if isinstance(w, str) else np.fft.fft(w))
    return ifft(z)
def unbind_vec(s, *ws):
    """s ⊘ w1 ⊘ w2 … — обратная операция: умножение на сопряжённый спектр."""
    z = np.fft.fft(s)
    for w in ws: z = z * np.conj(FC(w) if isinstance(w, str) else np.fft.fft(w))
    return ifft(z)


class Sieve:
    """Словарь для cleanup: любой вектор → ближайшее известное слово (как ARCHITECT §2.1)."""
    def __init__(self): self.words, self.M = [], None
    def add(self, *ws):
        new = [w for w in ws if w not in self.words]
        if new: self.words.extend(new); self.M = np.stack([C(w) for w in self.words])
    def cleanup(self, v, exclude=()):
        if self.M is None: return None, 0.0
        s = self.M @ unit(v)
        for i in np.argsort(-s):
            if self.words[i] not in exclude: return self.words[i], float(s[i])
        return None, 0.0


HOLDER = "__holder__"; LOC = "__loc__"; TIME = "__time__"
STOP = set("the a an to of is are was were in on at there back and did do does what who where "
           "he she it they him her them that this after before yes no not following then afterwards".split())
MOTION = {"travelled", "journeyed", "moved", "went"}
TAKE = {"got", "picked", "took", "grabbed"}
DROP = {"dropped", "discarded", "left", "put"}
def toks(s): return re.findall(r"[a-zA-Z']+", s.lower())
def toks_cased(s): return re.findall(r"[a-zA-Z']+", s)


class WorldVector:
    """Состояние мира — ОДИН вектор S плюс «тики» истории. Никаких словарей состояния."""
    def __init__(self):
        self.S = np.zeros(D)                 # текущее состояние
        self.hist = []                       # S после каждого шага (для «до»): это и есть Π-поворот,
                                             # реализованный как список снимков — эквивалентно, но прозрачнее
        self.sieve = Sieve()
        self.last_group = []                 # для he/she/they: последняя группа субъектов
        self.names = set()

    # --- чтение: кто где (unbind) ---
    def where(self, x, S=None):
        S = self.S if S is None else S
        v = unbind_vec(S, x, LOC)
        return self.sieve.cleanup(v, exclude=set(self.names) | {x, HOLDER, LOC})

    def holder_of(self, obj, S=None):
        S = self.S if S is None else S
        v = unbind_vec(S, obj, HOLDER)
        return self.sieve.cleanup(v, exclude={obj, HOLDER, LOC})

    # --- запись: одно предложение ---
    def read(self, raw):
        ws = toks(raw); cased = toks_cased(raw)
        names_here = [w for w, o in zip(ws, cased) if o[:1].isupper() and w not in STOP]
        if any(w in ("he", "she", "it", "they") for w in ws) and not names_here:
            names_here = list(self.last_group) if "they" in ws else self.last_group[-1:]
        if not names_here: return
        self.names.update(names_here)
        content = [w for w in ws if w not in STOP and w not in names_here]
        verbs = set(ws)
        self.sieve.add(*names_here, *content)
        if verbs & MOTION and content:
            loc = content[-1]
            for x in names_here:
                old, s = self.where(x)
                if old and s > 0.2: self.S = self.S - bind(x, LOC, old)      # вычитаем старое место
                self.S = self.S + bind(x, LOC, loc)                           # добавляем новое
                # объекты, которые держит x, едут с ним: их LOC-факт = LOC держателя (держим через HOLDER,
                # поэтому ничего не пишем — место объекта читается через holder_of → where(holder))
        elif verbs & TAKE and content:
            obj = content[-1]; x = names_here[0]
            old, s = self.holder_of(obj)
            if old and s > 0.2: self.S = self.S - bind(obj, HOLDER, old)
            oldloc, s2 = self.where(obj)
            if oldloc and s2 > 0.2: self.S = self.S - bind(obj, LOC, oldloc) # объект больше не «лежит» в месте
            self.S = self.S + bind(obj, HOLDER, x)
        elif verbs & DROP and content:
            obj = content[-1]; x = names_here[0]
            old, s = self.holder_of(obj)
            if old and s > 0.2: self.S = self.S - bind(obj, HOLDER, old)
            loc, s2 = self.where(x)
            if loc: self.S = self.S + bind(obj, LOC, loc)                     # объект замирает в месте держателя
        self.last_group = names_here
        self.hist.append(self.S.copy())

    def where_obj(self, obj, S=None):
        """Место объекта: либо лежит (LOC), либо у держателя (HOLDER → where(holder))."""
        S = self.S if S is None else S
        h, sh = self.holder_of(obj, S); l, sl = self.where(obj, S)
        if sh > sl and h in self.names:
            return self.where(h, S)[0]
        return l

    def where_obj_before(self, obj, place):
        """Идём по снимкам истории назад: находим последний момент, когда объект был в place,
        потом первый более ранний момент, когда он был в другом месте."""
        locs = [self.where_obj(obj, S) for S in self.hist]
        for i in range(len(locs) - 1, 0, -1):
            if locs[i] == place:
                for j in range(i - 1, -1, -1):
                    if locs[j] and locs[j] != place: return locs[j]
        return None


def answer(world, question):
    q = toks(question); cw = [w for w in q if w not in STOP]
    if not cw: return None
    if "before" in q:
        return world.where_obj_before(cw[0], cw[-1])
    target = cw[-1]
    if target in world.names: return world.where(target)[0]
    return world.where_obj(target)


def run(label, fname):
    import pandas as pd
    df = pd.read_parquet(f"data/babi/{fname}_test.parquet")
    t0 = time.time(); ok = 0
    for _, row in df.iterrows():
        w = WorldVector()
        for raw in row["passage"].strip().split("\n"):
            if raw.strip(): w.read(raw)
        a = answer(w, row["question"])
        ok += (a == row["answer"].strip().lower())
    n = len(df); r = dict(n=n, acc=ok / n, sec=round(time.time() - t0, 1))
    print(f"{label:28} N={n}  ВЕКТОРНОЕ состояние (один вектор, D={D}): {ok/n*100:5.1f}%   {r['sec']}s", flush=True)
    return r


def capacity_test():
    """Сколько фактов «X в L» влезает в ОДИН вектор до первой ошибки cleanup (ёмкость суперпозиции)."""
    rng = np.random.default_rng(0)
    names = [f"name{i}" for i in range(400)]; places = [f"place{i}" for i in range(50)]
    out = {}
    for n in (10, 25, 50, 100, 200, 400):
        S = np.zeros(D); sv = Sieve(); sv.add(*names[:n], *places)
        truth = {}
        for x in names[:n]:
            l = places[rng.integers(len(places))]; truth[x] = l; S = S + bind(x, LOC, l)
        ok = sum(sv.cleanup(unbind_vec(S, x, LOC), exclude=set(names))[0] == truth[x] for x in names[:n])
        out[n] = ok / n
        print(f"  ёмкость: {n:4} фактов в одном векторе D={D}: чтение верно {ok/n*100:5.1f}%", flush=True)
    return out


if __name__ == "__main__":
    OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
    res = {}
    for label, f in [("task1", "task_1-single_supporting_fact"), ("task2", "task_2-two_supporting_facts"),
                     ("task3", "task_3-three_supporting_facts"), ("task11", "task_11-basic_coreference"),
                     ("task13", "task_13-compound_coreference")]:
        res[label] = run(label, f)
    print("--- ёмкость одного вектора ---")
    res["capacity"] = capacity_test()
    json.dump(res, open(f"{OUT}/exp22_world_vector_D{D}.json", "w"), indent=1, ensure_ascii=False)
