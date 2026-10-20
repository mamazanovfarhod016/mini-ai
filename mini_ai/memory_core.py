"""Ядро памяти без градиента (эксперимент 20.10).

Идея: знание = запись (ключ → значение), а не сдвиг весов. Запись — один проход, без backprop.

T1  «ассоциативная память по hidden-state»
    ключ  k = нормированный hidden-state факта из слоя L базовой сети (что сеть «думает» о фразе)
    значение v = ответ (строка). Чтение: hidden-state вопроса → косинус с ключами → лучшая запись.
    Это НЕ генерация: ответ достаётся напрямую. Измеряем, может ли сама сеть (без дообучения)
    отличать факты друг от друга своими внутренними векторами.

T1h «Hebbian-матрица» — та же память, но хранится не списком, а ОДНОЙ матрицей
    W = Σ v_i k_iᵀ   (outer product, Hebb 1949 / Kohonen 1972),  v_i — one-hot индекс записи.
    Чтение: W·k_q → argmax. Ёмкость ограничена crosstalk'ом (взаимными помехами ключей).

T1w «whitened» — ключи центрируем и отбеливаем (PCA-whitening) перед записью: hidden-state
    трансформера анизотропны (все похожи друг на друга), отбеливание раздвигает их.
"""
import numpy as np
from mini_ai.transformer import layernorm_forward, gelu_forward, softmax, apply_rope, rope_cos_sin
import math


class Encoder:
    """Hidden-state текста из слоя L базовой сети (forward без градиента, по одному токену)."""

    def __init__(self, p, cfg, tok, layer):
        self.p = {k: np.asarray(v, dtype=np.float64) for k, v in p.items()}
        self.cfg, self.tok, self.layer = cfg, tok, layer

    def states(self, ids):
        """Возвращает (T, d) hidden-state ПОСЛЕ слоя self.layer для каждого токена."""
        p, cfg = self.p, self.cfg
        H, d = cfg["n_head"], cfg["d"]; dh = d // H
        use_rope = cfg.get("pos", "learned") == "rope"
        cache = [{"k": [], "v": []} for _ in range(cfg["n_layer"])]
        out = []
        for pos, t in enumerate(ids):
            x = p["wte"][t] if use_rope else p["wte"][t] + p["wpe"][pos]
            if use_rope:
                cos, sin = rope_cos_sin([pos], dh, cfg.get("rope_base", 10000.0), np); cos, sin = cos[0], sin[0]
            for l in range(self.layer + 1):
                h, _ = layernorm_forward(x, p[f"{l}.ln1.g"], p[f"{l}.ln1.b"])
                qkv = h @ p[f"{l}.attn.wqkv"] + p[f"{l}.attn.bqkv"]
                q, k, v = qkv[:d].reshape(H, dh), qkv[d:2 * d].reshape(H, dh), qkv[2 * d:].reshape(H, dh)
                if use_rope:
                    q = apply_rope(q[:, None, :], cos[None, :], sin[None, :])[:, 0, :]
                    k = apply_rope(k[:, None, :], cos[None, :], sin[None, :])[:, 0, :]
                cache[l]["k"].append(k); cache[l]["v"].append(v)
                K = np.stack(cache[l]["k"], 1); Vv = np.stack(cache[l]["v"], 1)
                a = softmax(np.einsum("hd,htd->ht", q, K) / math.sqrt(dh), -1)
                x = x + np.einsum("ht,htd->hd", a, Vv).reshape(d) @ p[f"{l}.attn.wo"] + p[f"{l}.attn.bo"]
                h2, _ = layernorm_forward(x, p[f"{l}.ln2.g"], p[f"{l}.ln2.b"])
                g, _ = gelu_forward(h2 @ p[f"{l}.mlp.w1"] + p[f"{l}.mlp.b1"])
                x = x + g @ p[f"{l}.mlp.w2"] + p[f"{l}.mlp.b2"]
            out.append(x.copy())
        return np.stack(out)

    def key(self, text, pool="mean"):
        from train_transformer import clean_question
        ids = self.tok.encode(clean_question(text))
        s = self.states(ids)
        k = s.mean(0) if pool == "mean" else s[-1]
        return k


def unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


class AssocMemory:
    """T1: список ключей. write = append (один проход). read = argmax cos."""

    def __init__(self):
        self.K, self.V = [], []

    def write(self, k, v):
        self.K.append(np.asarray(k, dtype=np.float64)); self.V.append(v)

    def read(self, q, top=1):
        K = unit(np.stack(self.K)); s = K @ unit(q)
        idx = np.argsort(-s)[:top]
        return [(self.V[i], float(s[i])) for i in idx]


class HebbMemory:
    """T1h: одна матрица W (N_max × d). write: W[i] += k_i (= outer product с one-hot v_i).
    read: scores = W·q. Ёмкость падает от crosstalk — это и измеряем."""

    def __init__(self, d, n_max=4096):
        self.W = np.zeros((n_max, d)); self.n = 0; self.V = []

    def write(self, k, v):
        self.W[self.n] += unit(np.asarray(k, dtype=np.float64)); self.V.append(v); self.n += 1

    def read(self, q, top=1):
        s = self.W[:self.n] @ unit(q)
        idx = np.argsort(-s)[:top]
        return [(self.V[i], float(s[i])) for i in idx]


class Whitener:
    """Центрирование + PCA-whitening по набору «фоновых» векторов (без обучения; одна SVD)."""

    def __init__(self, X, eps=1e-3, keep=None):
        X = np.asarray(X, dtype=np.float64)
        self.mu = X.mean(0)
        Xc = X - self.mu
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        if keep:
            Vt, S = Vt[:keep], S[:keep]
        self.P = Vt.T / (S / math.sqrt(max(1, len(X) - 1)) + eps)

    def __call__(self, x):
        return (np.asarray(x) - self.mu) @ self.P
