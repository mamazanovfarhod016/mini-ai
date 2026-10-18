"""
Mini AI v0.2 — маленький трансформер (как GPT, только крошечный), написанный С НУЛЯ на numpy.

Никаких PyTorch / TensorFlow / JAX: прямой проход, обратный проход (backprop) для
каждого слоя, оптимизатор Adam — всё руками, с формулами в комментариях.

Что делает модель
-----------------
Она читает текст ПО СИМВОЛАМ и каждый раз угадывает СЛЕДУЮЩИЙ символ:

    <bos> к а к   д е л а ? <sep> Н о р м а л ь н о ! <eos>
                               └──── модель пишет это сама, буква за буквой ────┘

Архитектура (decoder-only, «GPT-подобная»):

    символы → эмбеддинг символа + эмбеддинг позиции
            → [ LayerNorm → многоголовое causal self-attention → + (residual)
                LayerNorm → MLP (GELU)                        → + (residual) ] × n_layer
            → LayerNorm → умножение на матрицу эмбеддингов (связанные веса) → logits по словарю
            → softmax → вероятность каждого следующего символа

Обозначения размеров:  B — батч (сколько фраз сразу), T — длина последовательности,
d — d_model (длина вектора-«смысла» каждого символа), H — число голов, dh = d / H,
V — размер словаря (число разных символов + спецтокены).
"""
import base64
import json
import math

import re
import numpy as np

# ======================================================================================
# 1. Токенизатор: символ ↔ номер
# ======================================================================================
SPECIALS = ["<pad>", "<bos>", "<sep>", "<eos>"]
PAD, BOS, SEP, EOS = 0, 1, 2, 3
APOSTROPHES = "`’‘ʻʼ´"   # все виды узбекского апострофа (o‘, g‘) приводим к обычному '
MAX_Q = 40                 # вопрос длиннее 40 символов обрезаем (контекст ограничен)

# Плейсхолдеры даты/времени: раньше при обучении {time}/{date} заменялись на
# СЛУЧАЙНЫЕ конкретные значения ("08:15", "17.10.2026", ...) — модель должна была
# «угадывать» произвольные цифры и выдумывала их (баг: «сегодня 0.0.2026»).
# Теперь модель учится писать один и тот же символ-сентинел (их не бывает в
# обычном тексте), а дата/время с устройства подставляются ПОСЛЕ генерации —
# и в Python (fill_placeholders_runtime), и в JS (app.js fillPlaceholders) по
# ЭТИМ ЖЕ символам.
TIME_TOK = "\u2460"   # ①
DATE_TOK = "\u2461"   # ②


def fill_placeholders_runtime(text, now=None):
    """Заменяет сентинелы TIME_TOK/DATE_TOK на реальные час/дату (для демо в Python)."""
    import datetime as _dt
    now = now or _dt.datetime.now()
    return text.replace(TIME_TOK, now.strftime("%H:%M")).replace(DATE_TOK, now.strftime("%d.%m.%Y"))


def clean_question(text):
    """Вопрос пользователя → «чистая» строка: нижний регистр, ё→е, один вид апострофа,
    одиночные пробелы. Ровно то же делает cleanQuestion() в app/www/transformer.js."""
    t = str(text).lower().replace("ё", "е")
    for a in APOSTROPHES:
        t = t.replace(a, "'")
    # В обучающих вопросах нет знаков препинания — убираем их и здесь, иначе «Привет!»
    # для модели незнакомая строка и она пишет «кашу» (баг найден на скриншоте 17.10).
    t = re.sub(r"[!?.,;:()\"«»…—–-]+", " ", t)
    return " ".join(t.split())


class CharTokenizer:
    """Словарь = спецтокены + все символы, которые встретились в данных."""

    def __init__(self, chars):
        self.itos = SPECIALS + sorted(set(chars) - set(SPECIALS))
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, s):
        # Незнакомые символы (например редкий эмодзи) просто пропускаем.
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if i >= len(SPECIALS))

    def prompt(self, question):
        """<bos> вопрос <sep> — с этого модель начинает писать ответ."""
        q = self.encode(clean_question(question))[:MAX_Q]
        return [BOS] + q + [SEP]


# ======================================================================================
# 2. Инициализация весов
# ======================================================================================
def init_params(V, d=128, n_layer=3, n_head=4, block=160, d_ff=None, seed=0, dtype=np.float32):
    """Маленькие случайные веса ~ N(0, 0.02), как в GPT-2. Выходные проекции residual-веток
    уменьшаем в 1/sqrt(2·n_layer), чтобы сумма многих residual не «раздувалась» в начале обучения."""
    rng = np.random.default_rng(seed)
    d_ff = d_ff or 4 * d
    n = lambda *s, std=0.02: (rng.normal(0, std, s)).astype(dtype)
    z = lambda *s: np.zeros(s, dtype)
    o = lambda *s: np.ones(s, dtype)
    p = {"wte": n(V, d), "wpe": n(block, d, std=0.01)}
    proj_std = 0.02 / math.sqrt(2 * n_layer)
    for l in range(n_layer):
        p.update({
            f"{l}.ln1.g": o(d), f"{l}.ln1.b": z(d),
            f"{l}.attn.wqkv": n(d, 3 * d), f"{l}.attn.bqkv": z(3 * d),
            f"{l}.attn.wo": n(d, d, std=proj_std), f"{l}.attn.bo": z(d),
            f"{l}.ln2.g": o(d), f"{l}.ln2.b": z(d),
            f"{l}.mlp.w1": n(d, d_ff), f"{l}.mlp.b1": z(d_ff),
            f"{l}.mlp.w2": n(d_ff, d, std=proj_std), f"{l}.mlp.b2": z(d),
        })
    p["lnf.g"], p["lnf.b"] = o(d), z(d)
    p["bout"] = z(V)          # смещение выходного слоя (сама матрица связана с wte)
    cfg = {"V": V, "d": d, "n_layer": n_layer, "n_head": n_head, "block": block, "d_ff": d_ff}
    return p, cfg


def n_params(p):
    return int(sum(v.size for v in p.values()))


# ======================================================================================
# 3. Слои: прямой проход (forward) и обратный проход (backward)
#    Правило: forward возвращает (выход, cache); backward(dвыход, cache) → dвход (+ градиенты весов)
# ======================================================================================
LN_EPS = 1e-5


def layernorm_forward(x, g, b):
    """LayerNorm по последней оси (по d):
        μ = mean(x),  σ² = mean((x-μ)²),  x̂ = (x-μ)/sqrt(σ²+ε),  y = g·x̂ + b
    Делает так, чтобы каждый вектор имел среднее 0 и разброс 1 — обучение стабильнее."""
    mu = x.mean(-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + LN_EPS)
    xhat = xc * rstd
    return xhat * g + b, (xhat, rstd, g)


def layernorm_backward(dy, cache):
    """Производная LayerNorm (выводится через правило цепочки, учитывая что μ и σ зависят от x):
        dg = Σ dy·x̂,   db = Σ dy,   dx̂ = dy·g
        dx = (1/σ) · ( dx̂ − mean(dx̂) − x̂ · mean(dx̂·x̂) )"""
    xhat, rstd, g = cache
    dg = (dy * xhat).reshape(-1, xhat.shape[-1]).sum(0)
    db = dy.reshape(-1, dy.shape[-1]).sum(0)
    dxhat = dy * g
    dx = rstd * (dxhat - dxhat.mean(-1, keepdims=True) - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db


GELU_C = math.sqrt(2.0 / math.pi)


def gelu_forward(x):
    """GELU (как в GPT-2), tanh-приближение:  gelu(x) = ½·x·(1 + tanh(c·(x + 0.044715·x³))),  c = √(2/π).
    Похож на ReLU, но гладкий: маленькие отрицательные значения не обнуляются резко."""
    u = GELU_C * (x + 0.044715 * x ** 3)
    t = np.tanh(u)
    return 0.5 * x * (1.0 + t), (x, t)


def gelu_backward(dy, cache):
    """d gelu/dx = ½(1+t) + ½·x·(1−t²)·c·(1 + 3·0.044715·x²),  где t = tanh(u)."""
    x, t = cache
    return dy * (0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * GELU_C * (1.0 + 3 * 0.044715 * x * x))


def causal_mask(T):
    """Маска «нельзя смотреть в будущее»: символ i видит только символы 0..i.
    mask[i, j] = True, если j ≤ i (нижний треугольник)."""
    return np.tril(np.ones((T, T), dtype=bool))


def softmax(x, axis=-1):
    """softmax(z)_i = e^(z_i − max) / Σ_j e^(z_j − max). Вычитаем max — чтобы e^z не переполнилось."""
    e = np.exp(x - x.max(axis, keepdims=True))
    return e / e.sum(axis, keepdims=True)


def attention_forward(h, wqkv, bqkv, wo, bo, n_head):
    """Многоголовое causal self-attention.

        Q = h·Wq,  K = h·Wk,  V = h·Wv            (одна матрица wqkv делает все три сразу)
        S = Q·Kᵀ / √dh                             (насколько символ i «интересуется» символом j)
        S[i, j] = −∞  для j > i                    (causal-маска: будущее запрещено)
        A = softmax(S)  по j                       (веса внимания, в строке сумма = 1)
        Y = A·V                                    (смесь «значений» тех символов, куда смотрим)
        out = concat(головы)·Wo + bo

    Каждая голова (H штук) делает это в своём подпространстве размера dh = d/H — одна может
    следить за «какое было слово», другая — за «какую букву я сейчас пишу» и т.д."""
    B, T, d = h.shape
    dh = d // n_head
    qkv = h @ wqkv + bqkv                                        # (B,T,3d)
    qkv = qkv.reshape(B, T, 3, n_head, dh).transpose(2, 0, 3, 1, 4)  # (3,B,H,T,dh)
    q, k, v = qkv[0], qkv[1], qkv[2]
    scale = 1.0 / math.sqrt(dh)
    S = (q @ k.transpose(0, 1, 3, 2)) * scale                    # (B,H,T,T)
    mask = causal_mask(T)
    S = np.where(mask, S, -1e9)        # −∞ на будущих позициях → после softmax там ровно 0
    A = softmax(S, -1)
    Y = A @ v                                                    # (B,H,T,dh)
    Ym = Y.transpose(0, 2, 1, 3).reshape(B, T, d)                # склеиваем головы
    out = Ym @ wo + bo
    return out, (h, q, k, v, A, Ym, wqkv, wo, scale, n_head)


def attention_backward(dout, cache):
    """Обратный проход внимания, шаг за шагом в обратном порядке:
        out = Ym·Wo + bo     →  dWo = Ymᵀ·dout,  dbo = Σ dout,  dYm = dout·Woᵀ
        Y = A·V              →  dA = dY·Vᵀ,      dV = Aᵀ·dY
        A = softmax(S)       →  dS = A ⊙ (dA − Σ_j dA⊙A)        (якобиан softmax)
           (на замаскированных местах A = 0, значит и dS = 0: градиент в будущее не течёт)
        S = Q·Kᵀ·scale       →  dQ = dS·K·scale,  dK = dSᵀ·Q·scale
        qkv = h·Wqkv + b     →  dWqkv = hᵀ·dqkv,  dh = dqkv·Wqkvᵀ"""
    h, q, k, v, A, Ym, wqkv, wo, scale, n_head = cache
    B, T, d = h.shape
    dh_ = d // n_head
    dwo = Ym.reshape(-1, d).T @ dout.reshape(-1, d)
    dbo = dout.reshape(-1, d).sum(0)
    dYm = dout @ wo.T
    dY = dYm.reshape(B, T, n_head, dh_).transpose(0, 2, 1, 3)    # (B,H,T,dh)
    dA = dY @ v.transpose(0, 1, 3, 2)
    dv = A.transpose(0, 1, 3, 2) @ dY
    dS = A * (dA - (dA * A).sum(-1, keepdims=True))
    dS *= scale
    dq = dS @ k
    dk = dS.transpose(0, 1, 3, 2) @ q
    dqkv = np.stack([dq, dk, dv]).transpose(1, 3, 0, 2, 4).reshape(B, T, 3 * d)
    dwqkv = h.reshape(-1, d).T @ dqkv.reshape(-1, 3 * d)
    dbqkv = dqkv.reshape(-1, 3 * d).sum(0)
    dh = dqkv @ wqkv.T
    return dh, dwqkv, dbqkv, dwo, dbo


def dropout_mask(shape, p, rng, dtype):
    """Dropout: случайно «выключаем» долю p нейронов и делим остальные на (1−p).
    Мешает модели зазубрить обучающие фразы. При ответе (инференсе) не используется."""
    if p <= 0 or rng is None:
        return None
    return (rng.random(shape) >= p).astype(dtype) / (1.0 - p)


def block_forward(x, p, l, n_head, drop=0.0, rng=None):
    """Один блок трансформера (pre-LN, как GPT-2):
        x = x + Attn(LN1(x))
        x = x + MLP(LN2(x)),   MLP(z) = GELU(z·W1 + b1)·W2 + b2
    «+ x» — residual-связь: градиент свободно течёт насквозь, глубокую сеть легко учить."""
    h1, c_ln1 = layernorm_forward(x, p[f"{l}.ln1.g"], p[f"{l}.ln1.b"])
    a, c_att = attention_forward(h1, p[f"{l}.attn.wqkv"], p[f"{l}.attn.bqkv"], p[f"{l}.attn.wo"], p[f"{l}.attn.bo"], n_head)
    m1 = dropout_mask(a.shape, drop, rng, x.dtype)
    if m1 is not None:
        a = a * m1
    x2 = x + a
    h2, c_ln2 = layernorm_forward(x2, p[f"{l}.ln2.g"], p[f"{l}.ln2.b"])
    z = h2 @ p[f"{l}.mlp.w1"] + p[f"{l}.mlp.b1"]
    gz, c_gelu = gelu_forward(z)
    f = gz @ p[f"{l}.mlp.w2"] + p[f"{l}.mlp.b2"]
    m2 = dropout_mask(f.shape, drop, rng, x.dtype)
    if m2 is not None:
        f = f * m2
    return x2 + f, (c_ln1, c_att, m1, c_ln2, h2, c_gelu, gz, m2)


def block_backward(dy, cache, p, l, grads):
    """Обратный проход блока. Residual: y = x + F(x) ⇒ dx = dy + dF/dx·dy (градиент «раздваивается»)."""
    c_ln1, c_att, m1, c_ln2, h2, c_gelu, gz, m2 = cache
    d = dy.shape[-1]
    # --- MLP-ветка ---
    df = dy if m2 is None else dy * m2
    grads[f"{l}.mlp.w2"] += gz.reshape(-1, gz.shape[-1]).T @ df.reshape(-1, d)
    grads[f"{l}.mlp.b2"] += df.reshape(-1, d).sum(0)
    dgz = df @ p[f"{l}.mlp.w2"].T
    dz = gelu_backward(dgz, c_gelu)
    grads[f"{l}.mlp.w1"] += h2.reshape(-1, d).T @ dz.reshape(-1, dz.shape[-1])
    grads[f"{l}.mlp.b1"] += dz.reshape(-1, dz.shape[-1]).sum(0)
    dh2 = dz @ p[f"{l}.mlp.w1"].T
    dx2, dg, db = layernorm_backward(dh2, c_ln2)
    grads[f"{l}.ln2.g"] += dg
    grads[f"{l}.ln2.b"] += db
    dx2 = dx2 + dy                                   # residual
    # --- ветка внимания ---
    da = dx2 if m1 is None else dx2 * m1
    dh1, dwqkv, dbqkv, dwo, dbo = attention_backward(da, c_att)
    grads[f"{l}.attn.wqkv"] += dwqkv
    grads[f"{l}.attn.bqkv"] += dbqkv
    grads[f"{l}.attn.wo"] += dwo
    grads[f"{l}.attn.bo"] += dbo
    dx, dg, db = layernorm_backward(dh1, c_ln1)
    grads[f"{l}.ln1.g"] += dg
    grads[f"{l}.ln1.b"] += db
    return dx + dx2                                  # residual


# ======================================================================================
# 4. Вся модель
# ======================================================================================
def forward(p, cfg, idx, drop=0.0, rng=None):
    """idx (B,T) номера символов → logits (B,T,V): «оценки» каждого возможного следующего символа."""
    B, T = idx.shape
    x = p["wte"][idx] + p["wpe"][:T]              # эмбеддинг символа + эмбеддинг позиции
    m0 = dropout_mask(x.shape, drop, rng, x.dtype)
    if m0 is not None:
        x = x * m0
    caches = []
    for l in range(cfg["n_layer"]):
        x, c = block_forward(x, p, l, cfg["n_head"], drop, rng)
        caches.append(c)
    hf, c_lnf = layernorm_forward(x, p["lnf.g"], p["lnf.b"])
    logits = hf @ p["wte"].T + p["bout"]          # связанные веса: выход = эмбеддинг «наоборот»
    return logits, (idx, m0, caches, hf, c_lnf)


def cross_entropy(logits, targets, mask):
    """Cross-entropy только по отмеченным позициям (токены ОТВЕТА):
        L = −(1/N) Σ log p(правильный символ),   p = softmax(logits)
    Градиент по logits — знаменитая простая формула:  dL/dz = (p − onehot(y)) / N."""
    V = logits.shape[-1]
    P = softmax(logits.reshape(-1, V).astype(np.float64), -1)
    t = targets.reshape(-1)
    m = mask.reshape(-1).astype(np.float64)
    N = max(m.sum(), 1.0)
    logp = np.log(P[np.arange(len(t)), t] + 1e-30)
    loss = -(logp * m).sum() / N
    dlogits = P
    dlogits[np.arange(len(t)), t] -= 1.0
    dlogits *= (m / N)[:, None]
    return loss, dlogits.reshape(logits.shape).astype(logits.dtype)


def backward(p, cfg, dlogits, cache):
    """Полный обратный проход: от dL/dlogits до градиентов ВСЕХ весов."""
    idx, m0, caches, hf, c_lnf = cache
    grads = {k: np.zeros_like(v) for k, v in p.items()}
    V, d = p["wte"].shape
    dl = dlogits.reshape(-1, V)
    grads["bout"] += dl.sum(0)
    grads["wte"] += dl.T @ hf.reshape(-1, d)       # logits = hf·wteᵀ  ⇒  dwte = dlᵀ·hf
    dhf = dlogits @ p["wte"]
    dx, dg, db = layernorm_backward(dhf, c_lnf)
    grads["lnf.g"] += dg
    grads["lnf.b"] += db
    for l in reversed(range(cfg["n_layer"])):
        dx = block_backward(dx, caches[l], p, l, grads)
    if m0 is not None:
        dx = dx * m0
    # эмбеддинги: x = wte[idx] + wpe[t]  ⇒  градиент «складываем» в строки, которые использовались
    T = idx.shape[1]
    grads["wpe"][:T] += dx.sum(0)
    np.add.at(grads["wte"], idx.reshape(-1), dx.reshape(-1, d))
    return grads


def loss_and_grads(p, cfg, idx, targets, mask, drop=0.0, rng=None):
    logits, cache = forward(p, cfg, idx, drop, rng)
    loss, dlogits = cross_entropy(logits, targets, mask)
    return loss, backward(p, cfg, dlogits, cache)


# ======================================================================================
# 5. Оптимизатор: AdamW + warmup/cosine + gradient clipping
# ======================================================================================
class AdamW:
    """Adam:  m = β1·m + (1−β1)·g,   v = β2·v + (1−β2)·g²
              m̂ = m/(1−β1ᵗ),  v̂ = v/(1−β2ᵗ),   w −= lr·( m̂/(√v̂+ε) + wd·w )
    Weight decay (wd) только для матриц — LayerNorm, смещения и позиции не «сжимаем»."""

    def __init__(self, params, lr=3e-3, betas=(0.9, 0.98), eps=1e-8, wd=0.01):
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, betas[0], betas[1], eps, wd
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads, lr):
        self.t += 1
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k, g in grads.items():
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            upd = (self.m[k] / c1) / (np.sqrt(self.v[k] / c2) + self.eps)
            if params[k].ndim == 2 and k != "wpe":
                upd = upd + self.wd * params[k]
            params[k] -= (lr * upd).astype(params[k].dtype)


def lr_schedule(step, total, base, warmup):
    """Сначала линейный «разогрев» (warmup), потом плавное косинусное затухание до 10% от base."""
    if step < warmup:
        return base * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def clip_grads(grads, max_norm=1.0):
    """Gradient clipping: если общая длина градиента ‖g‖ > max_norm, уменьшаем его до max_norm.
    Защищает от редких «взрывов» градиента."""
    norm = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads.values()))
    if norm > max_norm:
        s = max_norm / (norm + 1e-6)
        for g in grads.values():
            g *= s
    return norm


# ======================================================================================
# 6. Генерация с KV-cache (так же работает app/www/transformer.js)
# ======================================================================================
class KVModel:
    """Инференс по одному символу за шаг. KV-cache: ключи K и значения V прошлых символов
    сохраняем, чтобы не пересчитывать их на каждом шаге — новая буква считает только свои q, k, v."""

    def __init__(self, p, cfg):
        self.p = {k: v.astype(np.float64) for k, v in p.items()}
        self.cfg = cfg

    def new_cache(self):
        return [{"k": [], "v": []} for _ in range(self.cfg["n_layer"])]

    def step(self, tok, pos, cache, want_attn=False):
        p, cfg = self.p, self.cfg
        H = cfg["n_head"]
        d = cfg["d"]
        dh = d // H
        x = p["wte"][tok] + p["wpe"][pos]
        attn = []
        for l in range(cfg["n_layer"]):
            h, _ = layernorm_forward(x, p[f"{l}.ln1.g"], p[f"{l}.ln1.b"])
            qkv = h @ p[f"{l}.attn.wqkv"] + p[f"{l}.attn.bqkv"]
            q, k, v = qkv[:d].reshape(H, dh), qkv[d:2 * d].reshape(H, dh), qkv[2 * d:].reshape(H, dh)
            cache[l]["k"].append(k)
            cache[l]["v"].append(v)
            K = np.stack(cache[l]["k"], 1)       # (H, t, dh) — все прошлые ключи + текущий
            Vv = np.stack(cache[l]["v"], 1)
            s = np.einsum("hd,htd->ht", q, K) / math.sqrt(dh)   # будущего в кэше просто нет → маска не нужна
            a = softmax(s, -1)
            y = np.einsum("ht,htd->hd", a, Vv).reshape(d)
            x = x + y @ p[f"{l}.attn.wo"] + p[f"{l}.attn.bo"]
            h2, _ = layernorm_forward(x, p[f"{l}.ln2.g"], p[f"{l}.ln2.b"])
            g, _ = gelu_forward(h2 @ p[f"{l}.mlp.w1"] + p[f"{l}.mlp.b1"])
            x = x + g @ p[f"{l}.mlp.w2"] + p[f"{l}.mlp.b2"]
            if want_attn:
                attn.append(a)
        hf, _ = layernorm_forward(x, p["lnf.g"], p["lnf.b"])
        logits = hf @ p["wte"].T + p["bout"]
        return logits, attn

    def generate(self, prompt_ids, max_new=None, temperature=0.0, rng=None, want_attn=False, on_token=None):
        """Greedy (temperature=0: всегда самый вероятный символ) или сэмплинг с температурой."""
        block = self.cfg["block"]
        max_new = max_new or block - len(prompt_ids)
        cache = self.new_cache()
        logits, attn = np.zeros(self.cfg["V"]), []
        for i, t in enumerate(prompt_ids):
            logits, attn = self.step(t, i, cache, want_attn and i == len(prompt_ids) - 1)
        # attns[i] — внимание на том шаге, который ПРЕДСКАЗАЛ i-й символ ответа
        out, attns, steps_logits = [], [], []
        pos = len(prompt_ids)
        for _ in range(max_new):
            steps_logits.append(logits)
            if want_attn:
                attns.append(attn)
            if temperature <= 0:
                nxt = int(np.argmax(logits))
            else:
                pr = softmax(logits / temperature)
                nxt = int((rng or np.random.default_rng()).choice(len(pr), p=pr))
            if nxt == EOS or pos >= block:
                break
            out.append(nxt)
            if on_token:
                on_token(nxt)
            logits, attn = self.step(nxt, pos, cache, want_attn)
            pos += 1
        return out, {"logits": steps_logits, "attn": attns}


# ======================================================================================
# 7. Сохранение весов: float16 → base64 (компактно для телефона)
# ======================================================================================
def to_f16_b64(a):
    return base64.b64encode(np.asarray(a, dtype="<f2").tobytes()).decode("ascii")


def from_f16_b64(s, shape):
    return np.frombuffer(base64.b64decode(s), dtype="<f2").astype(np.float64).reshape(shape)


def export_model(p, cfg, tok, meta, tokenizer_type="char", max_q=None, merges=None):
    """tokenizer_type: "char" (старое поведение, по умолчанию — не ломаем старые вызовы)
    или "bpe" (тогда дополнительно сохраняем merges и max_q — в токенах, не в символах)."""
    order = list(p.keys())
    out = {
        "format": "mini-ai-transformer/1",
        "config": cfg,
        "vocab": tok.itos,
        "max_q": max_q if max_q is not None else MAX_Q,
        "tokenizer": tokenizer_type,
        "order": order,
        "shapes": {k: list(p[k].shape) for k in order},
        "weights": {k: to_f16_b64(p[k]) for k in order},
        **meta,
    }
    if tokenizer_type == "bpe":
        out["merges"] = merges if merges is not None else []
    return out


def load_model(path_or_dict):
    """Возвращает (p, cfg, tok, meta). tok — CharTokenizer или BPETokenizer
    (определяется по полю "tokenizer" в JSON; отсутствие поля = старый char-формат)."""
    m = path_or_dict
    if not isinstance(m, dict):
        with open(path_or_dict, encoding="utf-8") as f:
            m = json.load(f)
    p = {k: from_f16_b64(m["weights"][k], m["shapes"][k]) for k in m["order"]}
    tokenizer_type = m.get("tokenizer", "char")
    if tokenizer_type == "bpe":
        from mini_ai.bpe import BPETokenizer
        tok = BPETokenizer.from_dict({"vocab": m["vocab"], "merges": m.get("merges", [])})
    else:
        tok = CharTokenizer([])
        tok.itos = m["vocab"]
        tok.stoi = {c: i for i, c in enumerate(tok.itos)}
    return p, m["config"], tok, m
