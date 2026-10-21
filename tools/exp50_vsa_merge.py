"""Эксперимент 50: «ДНК-слияние» (binding, VSA) с ОЧИСТКОЙ (cleanup-память) и нормировкой.

Гипотеза. exp10 провалился (~10% восстановления компонент после слияния) не потому, что binding не работает,
а потому что (1) не было cleanup-памяти — после каждого unbind шумный вектор шёл дальше в следующее
связывание, шум накапливался; (2) коды слов были буквенными (коррелированными, cos до 0.97), а не случайными
почти ортогональными; (3) не было нормировки (sign для биполярных / unit для HRR).
Если после КАЖДОГО unbind заменять результат на ближайший известный код (cleanup) — цепочки из 2–4
связываний через суперпозицию нескольких пар восстанавливаются ≥90%.

Задача (та же структура, что exp10 «сколько olma у друга Бека»): три памяти-суперпозиции
  M1 = Σ_i X_i⊛Y_i,  M2 = Σ_i Y_i⊛Z_i,  M3 = Σ_i Z_i⊛W_i   (k пар в каждой, один вектор на память).
Цепочка длины L: по X найти Y (L=1), Z (L=2), W (L=3), далее (L=4) через M4.
  шаг: v ← unbind(M, v); [cleanup: v ← ближайший известный код]; [norm].
Без cleanup: промежуточные векторы шумные, читаем ближайший код только в самом конце.
Контроли: случайный ответ (1/|словарь|); «буквенные коды без cleanup» = условия exp10.

Числа успеха (записаны ДО запуска):
  * цепочка L=3, k=5 пар в суперпозиции, D=2048..10000: с cleanup ≥90%, без cleanup — ожидаем ≪90% (как exp10 ~10%).
  * ёмкость: максимальное k, при котором одношаговое восстановление ≥90% (ожидаем ≥5 при D≥2048, рост с D).
Кодировки: bipolar MAP (±1, bind = поэлементное умножение = XOR), HRR (гауссовы, циркулярная свёртка).
Лог: ~/Developer/mini-ai-runs/learning-core/exp50_vsa_merge.{log,json}
"""
import sys, os, json, time, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

OUT = os.path.expanduser("~/Developer/mini-ai-runs/learning-core")
os.makedirs(OUT, exist_ok=True)
LOG = open(os.path.join(OUT, "exp50_vsa_merge.log"), "w")


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); LOG.write(s + "\n"); LOG.flush()


def unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True); return x / np.maximum(n, 1e-9)


# ---------------- кодировки ----------------
class Bipolar:
    name = "bipolar(MAP/XOR)"
    def __init__(self, D, rng): self.D, self.rng = D, rng
    def codes(self, n): return self.rng.choice([-1.0, 1.0], size=(n, self.D))
    def bind(self, a, b): return a * b
    def unbind(self, s, b): return s * b          # самообратное
    def bundle(self, vs): return np.sum(vs, axis=0)
    def norm(self, v):                            # биполяризация (sign), нули → случайно
        out = np.sign(v); z = out == 0
        if z.any(): out[z] = self.rng.choice([-1.0, 1.0], size=int(z.sum()))
        return out
    def sim(self, V, q): return V @ q


class HRR:
    name = "HRR(circ-conv)"
    def __init__(self, D, rng): self.D, self.rng = D, rng
    def codes(self, n): return unit(self.rng.standard_normal((n, self.D)))
    def bind(self, a, b): return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))
    def unbind(self, s, b): return np.real(np.fft.ifft(np.fft.fft(s) * np.conj(np.fft.fft(b))))
    def bundle(self, vs): return np.sum(vs, axis=0)
    def norm(self, v): return unit(v)
    def sim(self, V, q): return V @ q


def letter_code(word, d=192, ns=(2, 3)):
    """Буквенные n-граммы (как exp10/exp3) — коррелированные коды, условия провала exp10."""
    w = f"<{word}>"; v = np.zeros(d)
    for n in ns:
        for i in range(len(w) - n + 1):
            h = zlib.crc32(w[i:i + n].encode()); v[h % d] += 1.0 if (h >> 16) & 1 else -1.0
    return v


SYL = ["ba", "zo", "ra", "ki", "lo", "mu", "na", "sa", "ti", "ve", "do", "fe", "gu", "ha", "ji", "ko", "le", "mi", "no", "pu"]
def names(n, rng):
    out = set()
    while len(out) < n:
        out.add("".join(rng.choice(SYL) for _ in range(rng.integers(2, 4))).capitalize())
    return sorted(out)


# ---------------- задача ----------------
def run_chain(enc, V, k, L, n_trials, rng, cleanup, normalize):
    """V — (n, D) словарь кодов. k пар в каждой из L памятей. Возвращает точность по n_trials цепочкам."""
    n = len(V); correct = 0
    for _ in range(n_trials):
        # L+1 слоёв сущностей, по k в каждом, без повторов внутри слоя
        layers = [rng.choice(n, size=k, replace=False) for _ in range(L + 1)]
        mems = []
        for l in range(L):
            pairs = [enc.bind(V[layers[l][i]], V[layers[l + 1][i]]) for i in range(k)]
            m = enc.bundle(pairs)
            mems.append(enc.norm(m) if normalize else m)
        q = rng.integers(k)                       # какую цепочку спрашиваем
        v = V[layers[0][q]]
        for l in range(L):
            v = enc.unbind(mems[l], v)
            if cleanup:
                v = V[int(np.argmax(enc.sim(V, v)))]
            elif normalize:
                v = enc.norm(v)
        ans = int(np.argmax(enc.sim(V, v)))
        correct += ans == layers[L][q]
    return correct / n_trials


def main():
    t0 = time.time(); rng = np.random.default_rng(0)
    N_VOCAB = int(os.environ.get("N_VOCAB", 1000)); TRIALS = int(os.environ.get("TRIALS", 200))
    res = {"vocab": N_VOCAB, "trials": TRIALS, "chain": [], "capacity": [], "exp10_letters": []}
    log(f"exp50: словарь {N_VOCAB} кодов, {TRIALS} цепочек на точку; контроль случайный = {100 / N_VOCAB:.1f}%")

    # --- 1. цепочки: L=1..4, k=5, с/без cleanup, с/без нормировки ---
    log("\n== 1. Цепочки через суперпозицию k=5 пар (точность, %) ==")
    log(f"{'кодировка':>18} {'D':>6} {'L':>2} | {'без cleanup, без norm':>22} {'без cleanup, norm':>18} {'cleanup':>8}")
    for Enc in (Bipolar, HRR):
        for D in (2048, 4096, 10000):
            enc = Enc(D, rng); V = enc.codes(N_VOCAB)
            for L in (1, 2, 3, 4):
                a0 = run_chain(enc, V, 5, L, TRIALS, rng, cleanup=False, normalize=False)
                a1 = run_chain(enc, V, 5, L, TRIALS, rng, cleanup=False, normalize=True)
                a2 = run_chain(enc, V, 5, L, TRIALS, rng, cleanup=True, normalize=True)
                res["chain"].append(dict(enc=enc.name, D=D, k=5, L=L, raw=a0, norm=a1, cleanup=a2))
                log(f"{enc.name:>18} {D:>6} {L:>2} | {a0 * 100:>21.1f}% {a1 * 100:>17.1f}% {a2 * 100:>7.1f}%")

    # --- 2. ёмкость: k пар в одном векторе, одношаговое восстановление, с cleanup (единственный способ прочитать) ---
    log("\n== 2. Ёмкость: k пар в одной суперпозиции, L=1 (точность %), и L=3 с cleanup ==")
    KS = [2, 5, 10, 20, 40, 80, 160]
    for Enc in (Bipolar, HRR):
        for D in (2048, 4096, 10000):
            enc = Enc(D, rng); V = enc.codes(N_VOCAB)
            row1, row3 = [], []
            for k in KS:
                a1 = run_chain(enc, V, k, 1, TRIALS, rng, cleanup=True, normalize=True)
                a3 = run_chain(enc, V, k, 3, TRIALS, rng, cleanup=True, normalize=True)
                row1.append(a1); row3.append(a3)
                res["capacity"].append(dict(enc=enc.name, D=D, k=k, L1=a1, L3=a3))
            kmax1 = max([k for k, a in zip(KS, row1) if a >= 0.9], default=0)
            kmax3 = max([k for k, a in zip(KS, row3) if a >= 0.9], default=0)
            log(f"{enc.name:>18} D={D:>5}  L=1: " + " ".join(f"k={k}:{a * 100:.0f}%" for k, a in zip(KS, row1))
                + f"  → ёмкость≥90%: k={kmax1}")
            log(f"{'':>18} {'':>7}  L=3: " + " ".join(f"k={k}:{a * 100:.0f}%" for k, a in zip(KS, row3))
                + f"  → ёмкость≥90%: k={kmax3}")
            res["capacity"].append(dict(enc=enc.name, D=D, kmax_L1=kmax1, kmax_L3=kmax3))

    # --- 3. условия exp10: буквенные (коррелированные) коды имён, HRR, D=2048 ---
    log("\n== 3. Условия exp10: буквенные коды имён (коррелированные), HRR D=2048, k=5 ==")
    D = 2048; enc = HRR(D, rng)
    P = rng.standard_normal((192, D)) / np.sqrt(192)
    ws = names(N_VOCAB, rng); Vl = unit(np.stack([letter_code(w) for w in ws]) @ P)
    S = Vl @ Vl.T; np.fill_diagonal(S, -1)
    log(f"  буквенные коды: средний cos ближайшего соседа {S.max(1).mean():.2f} (случайные ≈ {1 / np.sqrt(D) * 2:.2f})")
    for L in (1, 2, 3):
        a0 = run_chain(enc, Vl, 5, L, TRIALS, rng, cleanup=False, normalize=False)
        a2 = run_chain(enc, Vl, 5, L, TRIALS, rng, cleanup=True, normalize=True)
        Vc = enc.codes(N_VOCAB)
        a3 = run_chain(enc, Vc, 5, L, TRIALS, rng, cleanup=True, normalize=True)
        res["exp10_letters"].append(dict(D=D, L=L, letters_raw=a0, letters_cleanup=a2, canon_cleanup=a3))
        log(f"  L={L}: буквенные без cleanup {a0 * 100:.1f}%  | буквенные + cleanup {a2 * 100:.1f}%  | случайные + cleanup {a3 * 100:.1f}%")

    # --- пример в консоль ---
    log("\n== Пример (bipolar, D=4096, k=5, L=3) ==")
    enc = Bipolar(4096, rng); ws = names(N_VOCAB, rng); V = enc.codes(N_VOCAB); idx = {w: i for i, w in enumerate(ws)}
    layers = [rng.choice(N_VOCAB, size=5, replace=False) for _ in range(4)]
    mems = [enc.norm(enc.bundle([enc.bind(V[layers[l][i]], V[layers[l + 1][i]]) for i in range(5)])) for l in range(3)]
    q = 0; chain = [ws[layers[l][q]] for l in range(4)]
    log(f"  факты (5 пар в каждой из 3 памятей); истинная цепочка: {' → '.join(chain)}")
    v = V[layers[0][q]]; got = []
    for l in range(3):
        v = enc.unbind(mems[l], v); sims = enc.sim(V, v); i = int(np.argmax(sims))
        got.append(f"{ws[i]}(cos={sims[i] / 4096:.2f})"); v = V[i]
    log(f"  восстановлено с cleanup: {chain[0]} → " + " → ".join(got))
    v = V[layers[0][q]]
    for l in range(3): v = enc.unbind(mems[l], v)
    i = int(np.argmax(enc.sim(V, v)))
    log(f"  без cleanup (3 шага подряд): {chain[0]} → ... → {ws[i]}  {'✓' if i == layers[3][q] else '✗'}")

    res["time_s"] = time.time() - t0
    json.dump(res, open(os.path.join(OUT, "exp50_vsa_merge.json"), "w"), indent=1, ensure_ascii=False)
    log(f"\nготово за {res['time_s']:.0f} c")


if __name__ == "__main__":
    main()
