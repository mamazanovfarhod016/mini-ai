"""Эксперимент 23: ДВЕ ПАМЯТИ — «не забывает» при любом объёме.
Проблема v2 (exp22): один вектор S держит ~25-50 фактов без ошибок, 400 → 28%. Это рабочая память.
Решение (как у человека): рабочий вектор S (быстро, суперпозиция, десятки фактов) + долговременный
СПИСОК L (exp4: 97.7% на 1000; ключ=bind(X,ρ), значение=C(L)). Правило: каждый факт пишется в ОБА.
Читаем сначала из S (одна операция); если уверенность (зазор s1−s2) мала — идём в L (поиск по списку).
Забывания нет: список только растёт, старые записи никто не трогает (exp7: «старые целы 100%»).
Обновление факта (X переехал): в S — вычитание старого; в L — новая запись ПОВЕРХ (recency: при
равных ключах выигрывает последняя — как exp17 best_recent).

Проверяем ёмкость: N фактов «X в L» = 25, 100, 400, 1000, 4000 — точность чтения S-only / L-only / S+L,
и доля ответов, взятых из S (быстрый путь) vs из L. Плюс «не забывает»: после 4000 новых фактов —
точность на первых 100 (они должны остаться 100%).
"""
import sys, os, json, time, zlib
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
def bind(*ws):
    z = np.ones(D, dtype=complex)
    for w in ws: z = z * FC(w)
    return np.real(np.fft.ifft(z))
def unbind(s, *ws):
    z = np.fft.fft(s)
    for w in ws: z = z * np.conj(FC(w))
    return np.real(np.fft.ifft(z))

LOC = "__loc__"


class TwoMemory:
    def __init__(self, vocab_places):
        self.S = np.zeros(D)                                   # рабочая память (суперпозиция)
        self.LK, self.LV, self.Lkey_words = [], [], []         # долговременная: ключи, значения, имена
        self._LK = None
        self.places = list(vocab_places); self.PM = np.stack([C(p) for p in self.places])
        self.stats = dict(from_S=0, from_L=0)

    def write(self, x, loc):
        self.S = self.S + bind(x, LOC, loc)
        self.LK.append(bind(x, LOC)); self.LV.append(C(loc)); self.Lkey_words.append(x); self._LK = None

    def _cleanup(self, v):
        s = self.PM @ unit(v); o = np.argsort(-s)
        return self.places[o[0]], float(s[o[0]] - s[o[1]])    # слово и ЗАЗОР до второго

    def read_S(self, x):
        return self._cleanup(unbind(self.S, x, LOC))

    def read_L(self, x):
        if self._LK is None: self._LK = np.stack(self.LK)
        k = bind(x, LOC); s = self._LK @ k
        best = int(np.argmax(s))
        # recency при равных ключах: среди ключей с s ≥ max−1e-6 берём последний
        top = s[best]; cands = np.where(s >= top - 1e-6)[0]
        i = int(cands[-1])
        return self._cleanup(self.LV[i])[0]

    def read(self, x, gap_thr=0.05):
        w, gap = self.read_S(x)
        if gap >= gap_thr: self.stats["from_S"] += 1; return w
        self.stats["from_L"] += 1; return self.read_L(x)


def run():
    OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
    rng = np.random.default_rng(0)
    places = [f"place{i}" for i in range(50)]
    res = {}
    for N in (25, 100, 400, 1000, 4000):
        names = [f"name{i}" for i in range(N)]
        truth = {x: places[rng.integers(50)] for x in names}
        m = TwoMemory(places); t0 = time.time()
        for x in names: m.write(x, truth[x])
        t_w = time.time() - t0
        t0 = time.time()
        okS = sum(m.read_S(x)[0] == truth[x] for x in names)
        okL = sum(m.read_L(x) == truth[x] for x in names)
        m.stats = dict(from_S=0, from_L=0)
        okSL = sum(m.read(x) == truth[x] for x in names)
        from_S = m.stats["from_S"]
        t_r = time.time() - t0
        first100 = sum(m.read(x) == truth[x] for x in names[:min(100, N)]) / min(100, N)
        r = dict(N=N, S_only=okS / N, L_only=okL / N, S_plus_L=okSL / N, from_S=from_S / N,
                 first100_intact=first100, write_s=round(t_w, 2), read_s=round(t_r, 2))
        res[N] = r
        print(f"N={N:5}: только S {okS/N*100:5.1f}%  только L {okL/N*100:5.1f}%  S+L {okSL/N*100:5.1f}%  "
              f"(из S взято {from_S/N*100:4.0f}%)  первые 100 целы {first100*100:5.1f}%  "
              f"запись {t_w:.2f}s чтение {t_r:.2f}s", flush=True)
    json.dump(res, open(f"{OUT}/exp23_two_memory.json", "w"), indent=1)


if __name__ == "__main__":
    run()
