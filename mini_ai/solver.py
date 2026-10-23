"""solver.py — ОБУЧАЕМЫЙ АРБИТР БЕЗ ГРАДИЕНТА (замена градиентам внутри ядра, exp62 → ядро).

Та же механика, что в tools/exp62_nograd_lm.py: фиксированные случайные признаки + ОДНО замкнутое решение
    W = (A + λI)⁻¹ B,   A = Σ φφᵀ,  B = Σ φ·y
вместо шагов градиента. Суммы A, B копятся от каждого исправления («no, it's X») и подтверждения («yes»),
решение пересчитывается за миллисекунды → учится с первого примера и НЕ забывает старое.

Что учит: для каждого кандидата на доске арбитра (способ, ответ, уверенность, доверие, самопроверка, согласие,
смысл вопроса, смысл ответа и их взаимодействие) — вероятность, что именно этот кандидат верен.
Как применяется: в _arbiter после подсчёта score, когда примеров ≥ MIN_N: score *= (0.5 + p).
Отключить для абляции: MINI_AI_SOLVER=0. Состояние: <state_dir>/solver.npz (A, B, n).
"""
import os, re, json, zlib
import numpy as np

MIN_N = int(os.environ.get("MINI_AI_SOLVER_MIN", "12"))      # с какого числа размеченных кандидатов начинать вмешиваться
HOWS = ["1 шаг", "2 шага", "отрывок предложения", "отрывок предложения (смело", "отрывок: другое предложение", "комбинация", "нейронный читатель", "шагов (deep)", "?"]
D_SEM = 96      # проекция смысла (4096 → 96)
N_RES = 512     # случайные нелинейные признаки
LAM = 2.0


def _toks(s): return re.findall(r"[a-zа-яё]+", (s or "").lower())


class Solver:
    def __init__(self, mind, state_dir=None):
        self.mind = mind; self.rng = np.random.default_rng(62)
        self.P = (self.rng.normal(0, 1, (4096, D_SEM)) / np.sqrt(4096)).astype(np.float32)   # фиксированная проекция смысла
        self.n_raw = len(HOWS) + 16 + 12 + 3 * D_SEM + 64 + D_SEM        # + признаки ПРЕДЛОЖЕНИЯ-источника (покрытие, редкие слова, смысл q↔sent)                                        # способ, числа, смыслы q/a/q⊙a, форма вопроса (хэш-код)
        self.Win = (self.rng.normal(0, 1, (self.n_raw, N_RES)) / np.sqrt(self.n_raw) * 1.5).astype(np.float32)
        self.F = N_RES + self.n_raw + 1
        self.A = np.zeros((self.F, self.F), np.float64); self.B = np.zeros(self.F, np.float64); self.n = 0; self.n_pos = 0
        self.W = None; self.path = os.path.join(state_dir, "solver.npz") if state_dir else None
        self.last_feats = None; self.last_q = None; self.stats = {"applied": 0, "learned": 0}
        if self.path and os.path.exists(self.path):
            z = np.load(self.path); self.A, self.B, self.n, self.n_pos = z["A"], z["B"], int(z["n"]), int(z["n_pos"]); self._solve()

    # ---------- признаки ----------
    def _sem(self, words):
        M = getattr(self.mind, "meaning", None)
        v = M.sent_vec(words) if M else None
        if v is None: return np.zeros(D_SEM, np.float32)
        p = v.astype(np.float32) @ self.P; return p / (np.linalg.norm(p) + 1e-9)

    def _src_sentence(self, c):
        f = (c.get("facts") or [""])[-1]
        return f.split("—span→", 1)[1] if "—span→" in f else f.replace("—", " ").replace("→", " ")

    def _sent_sem(self, c):
        return self._sem(_toks(self._src_sentence(c)))

    def _sent_feats(self, q, qw, c, aw):
        """о предложении-источнике: насколько оно про ТО ЖЕ, что вопрос (а не просто с теми же словами)"""
        m = self.mind; kb = getattr(m, "kb", None); sent = self._src_sentence(c); sw = set(_toks(sent))
        try: forms = m._forms; meta = m.META
        except Exception: forms = lambda w: {w}; meta = set()
        qc = [w for w in qw if len(w) > 2 and w not in meta and not (hasattr(m, "_soft") and False)]
        try:
            from mini_ai.experimental import content_words; qc = [w for w in content_words(qw) if w not in meta]
        except Exception: pass
        if not qc: return np.zeros(12, np.float32)
        idf = {w: (kb.idf(w) if kb else 1.0) for w in qc}; tot = sum(idf.values()) or 1.0
        sforms = set().union(*[forms(x) for x in sw]) if sw else set()
        present = [w for w in qc if forms(w) & sforms]; missing = [w for w in qc if w not in present]
        rare = [w for w in qc if idf[w] >= 2.5]; rare_miss = [w for w in rare if w in missing]
        cov = sum(idf[w] for w in present) / tot
        maxidf_missing = max([idf[w] for w in missing], default=0.0) / 6.0
        M = getattr(m, "meaning", None); sim = M.phrase_sim(qc, list(sw - set(qc))) if M else None
        ans_idf = max([kb.idf(x) for x in aw], default=0.0) / 6.0 if kb and aw else 0.0
        called = 1.0 if re.search(r"\b(?:called|known as|termed|named|referred to as)\b", sent, re.I) else 0.0
        ans_pos = (sent.lower().find(" ".join(aw)) / max(1, len(sent))) if aw else 0.5
        return np.array([cov, len(present) / len(qc), len(missing) / len(qc), len(rare_miss) / max(1, len(rare)), 1.0 if rare_miss else 0.0,
                         maxidf_missing, (sim or 0.0), 1.0 if sim is None else 0.0, ans_idf, called, min(1.0, max(0.0, ans_pos)), min(len(sw), 40) / 40.0], np.float32)

    def _qcode(self, q):
        """грубая форма вопроса (как в growth.qcoarse) → детерминированный ±1 код 64"""
        try:
            from mini_ai.growth import qcoarse; f = qcoarse(q)
        except Exception: f = " ".join(_toks(q)[:2])
        r = np.random.default_rng(zlib.crc32(f.encode()) % (2**32)); return r.choice([-1.0, 1.0], size=64).astype(np.float32) / 8

    def features(self, q, cands):
        """один вектор признаков на кандидата; вызывается в _arbiter после подсчёта score/trust/check"""
        qw = _toks(q); qs = self._sem(qw); qc = self._qcode(q); n = max(1, len(cands)); out = []
        scores = np.array([c.get("score", 0.0) for c in cands], np.float32); smax = float(scores.max()) if len(scores) else 1.0
        for i, c in enumerate(cands):
            how = c.get("how", "?"); hk = np.zeros(len(HOWS), np.float32)
            j = next((k for k, h in enumerate(HOWS) if how.startswith(h)), len(HOWS) - 1); hk[j] = 1.0
            aw = _toks(str(c.get("answer", ""))); as_ = self._sem(aw)
            chk = str(c.get("check") or "")
            nums = np.array([c.get("confidence", 0.0), c.get("trust", 1.0) - 1.0, c.get("score", 0.0), c.get("score", 0.0) / (smax + 1e-9),
                             1.0 if "+согласие" in how else 0.0, 1.0 / n, i / n, min(len(aw), 6) / 6.0,
                             1.0 if ("type" in chk or "mismatch" in chk) else 0.0, 1.0 if set(aw) & set(qw) else 0.0,
                             c.get("alt_rank", 0) / 12.0, c.get("alt_rel", 1.0), 1.0 if c.get("alt_rank") is not None else 0.0,
                             1.0 if c.get("alt_passed", True) else 0.0, c.get("alt_rare", 0) / 3.0, c.get("alt_neural", 0.0)], np.float32)
            raw = np.concatenate([hk, nums, self._sent_feats(q, qw, c, aw), qs, as_, qs * as_ * 8.0, qc, self._sent_sem(c) * qs * 8.0])
            out.append(np.concatenate([np.tanh(raw @ self.Win), raw, [1.0]]).astype(np.float32))
        self.last_feats = out; self.last_q = q; return out

    # ---------- решение и применение ----------
    def _solve(self):
        if self.n < MIN_N or self.n_pos == 0 or self.n_pos == self.n: self.W = None; return
        self.W = np.linalg.solve(self.A + LAM * np.eye(self.F), self.B).astype(np.float32)

    @property
    def ready(self): return self.W is not None and os.environ.get("MINI_AI_SOLVER", "1") != "0"

    def predict(self, feats):
        if self.W is None: return [0.5] * len(feats)
        return [float(np.clip(f @ self.W, 0.0, 1.0)) for f in feats]

    def apply(self, cands, feats):
        """score *= 0.5 + p — p учтён как мнение ещё одного свидетеля, не как приговор"""
        if not self.ready or not feats or len(feats) != len(cands): return False
        for c, p in zip(cands, self.predict(feats)):
            c["solver"] = round(p, 2); c["score"] = round(c["score"] * (0.5 + p), 3)
        self.stats["applied"] += 1; return True

    # ---------- учёба ----------
    def learn(self, q, cands_board, right, feats=None):
        """right — верный ответ (из «no, it's X» или подтверждённый). Каждый кандидат доски получает метку 1/0."""
        feats = feats if feats is not None else (self.last_feats if self.last_q == q else None)
        if not feats or len(feats) != len(cands_board): return 0
        rw = set(_toks(right)); k = 0
        for f, c in zip(feats, cands_board):
            aw = set(_toks(str(c.get("answer", ""))))
            y = 1.0 if (rw and aw and (rw <= aw or aw <= rw)) else 0.0
            f64 = f.astype(np.float64); self.A += np.outer(f64, f64); self.B += f64 * y; self.n += 1; self.n_pos += int(y); k += 1
        if k: self._solve(); self.stats["learned"] += k; self.save()
        return k

    def save(self):
        if self.path: np.savez(self.path, A=self.A, B=self.B, n=self.n, n_pos=self.n_pos)

    def report(self):
        return f"solver: {self.n} labeled candidates ({self.n_pos} right), {'active' if self.ready else 'warming up'}, applied {self.stats['applied']}×"
