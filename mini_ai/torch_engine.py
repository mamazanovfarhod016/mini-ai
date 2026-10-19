"""
Движок обучения на PyTorch — ТА ЖЕ САМАЯ сеть, что в transformer.py, только обратный проход
(backprop) считает PyTorch (autograd), а не наши формулы. Нужен для больших обучений на видеокарте.

Что здесь своё, а что из библиотеки:
  • устройство сети (эмбеддинги, внимание, LayerNorm, GELU, связанные веса) — наше, повторяет
    transformer.forward строка в строку;
  • оптимизатор AdamW, расписание lr и обрезка градиента — наши формулы (не torch.optim);
  • PyTorch даёт только: массивы на видеокарте и автоматические производные.

Эталон — наш numpy-код. tests/test_torch_engine.py проверяет, что loss и все градиенты
совпадают с transformer.loss_and_grads, и что шаги обучения дают те же веса.

    train_transformer.py --engine torch      # на видеокарте, если есть (cuda), иначе на процессоре
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

LN_EPS = 1e-5


def _ln(x, g, b):
    mu = x.mean(-1, keepdim=True)
    xc = x - mu
    var = (xc * xc).mean(-1, keepdim=True)
    return xc * torch.rsqrt(var + LN_EPS) * g + b


def _gelu(x):
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * x ** 3)))


def _drop(x, p, gen):
    if p <= 0 or gen is None:
        return x
    keep = (torch.rand(x.shape, generator=gen, device=x.device) >= p).to(x.dtype)
    return x * keep / (1.0 - p)


def _rope_cos_sin(positions, dh, base, device):
    """Как transformer.rope_cos_sin, но на torch (для forward/backward через autograd)."""
    i = torch.arange(0, dh, 2, dtype=torch.float64, device=device)
    theta = base ** (-i / dh)
    ang = positions.to(torch.float64)[:, None] * theta[None, :]
    return torch.cos(ang), torch.sin(ang)


def _apply_rope(x, cos, sin):
    """Поворот пар координат последней оси x (...,T,dh), как transformer.apply_rope."""
    *lead, T, dh = x.shape
    xr = x.reshape(*lead, T, dh // 2, 2)
    x0, x1 = xr[..., 0], xr[..., 1]
    c = cos.to(x.dtype).reshape((1,) * len(lead) + cos.shape)
    s = sin.to(x.dtype).reshape((1,) * len(lead) + sin.shape)
    o0 = x0 * c - x1 * s
    o1 = x0 * s + x1 * c
    return torch.stack([o0, o1], dim=-1).reshape(*lead, T, dh)


def forward(p, cfg, idx, drop=0.0, gen=None):
    """Как transformer.forward: idx (B,T) → logits (B,T,V)."""
    B, T = idx.shape
    d, H = cfg["d"], cfg["n_head"]
    dh = d // H
    use_rope = cfg.get("pos", "learned") == "rope"
    if use_rope:
        x = _drop(p["wte"][idx], drop, gen)
        cos, sin = _rope_cos_sin(torch.arange(T, device=idx.device), dh, cfg.get("rope_base", 10000.0), idx.device)
    else:
        x = _drop(p["wte"][idx] + p["wpe"][:T], drop, gen)
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=idx.device))
    use_sdpa = drop <= 0 and T > 64   # на коротких T точность не теряем, на длинных — экономим память
    for l in range(cfg["n_layer"]):
        h1 = _ln(x, p[f"{l}.ln1.g"], p[f"{l}.ln1.b"])
        qkv = (h1 @ p[f"{l}.attn.wqkv"] + p[f"{l}.attn.bqkv"]).reshape(B, T, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if use_rope:
            q = _apply_rope(q, cos, sin)
            k = _apply_rope(k, cos, sin)
        if use_sdpa:
            # scaled_dot_product_attention с is_causal=True — то же самое, что softmax(QK/√dh)·V
            # с causal-маской, но экономнее по памяти на длинных последовательностях (RoPE/10000)
            Y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            S = (q @ k.transpose(-1, -2)) / math.sqrt(dh)
            S = S.masked_fill(~mask, -1e9)
            Y = torch.softmax(S, -1) @ v
        a = Y.transpose(1, 2).reshape(B, T, d) @ p[f"{l}.attn.wo"] + p[f"{l}.attn.bo"]
        x2 = x + _drop(a, drop, gen)
        h2 = _ln(x2, p[f"{l}.ln2.g"], p[f"{l}.ln2.b"])
        f = _gelu(h2 @ p[f"{l}.mlp.w1"] + p[f"{l}.mlp.b1"]) @ p[f"{l}.mlp.w2"] + p[f"{l}.mlp.b2"]
        x = x2 + _drop(f, drop, gen)
    hf = _ln(x, p["lnf.g"], p["lnf.b"])
    return hf @ p["wte"].T + p["bout"]


def masked_ce(logits, targets, mask):
    """Как transformer.cross_entropy: среднее −log p(правильного токена) только по позициям ответа."""
    V = logits.shape[-1]
    logp = torch.log_softmax(logits.reshape(-1, V).double(), -1)
    nll = -logp.gather(1, targets.reshape(-1, 1)).squeeze(1)
    m = mask.reshape(-1).double()
    return (nll * m).sum() / torch.clamp(m.sum(), min=1.0)


class TorchEngine:
    """Веса на видеокарте + наш AdamW. step() = один шаг обучения, как в train_transformer.py."""

    def __init__(self, p_np, cfg, lr=3e-3, betas=(0.9, 0.98), eps=1e-8, wd=0.01, seed=0, device=None):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cfg = cfg
        self.p = {k: torch.tensor(np.asarray(v), dtype=torch.float32, device=self.dev, requires_grad=True)
                  for k, v in p_np.items()}
        self.m = {k: torch.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: torch.zeros_like(v) for k, v in self.p.items()}
        self.b1, self.b2, self.eps, self.wd, self.t = betas[0], betas[1], eps, wd, 0
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        import os
        self.amp = self.dev.type == "cuda" and os.environ.get("MINI_AI_AMP", "0") == "1"   # включить после проверки на Kaggle (сверка loss с fp32)
        self.scale, self._good = 2.0 ** 14, 0

    def _t(self, a, dtype):
        return torch.as_tensor(np.asarray(a), dtype=dtype, device=self.dev)

    def loss_and_grads(self, idx, targets, mask, drop=0.0):
        for v in self.p.values():
            v.grad = None
        # 19.10 (план, нед. 1): fp16 на видеокарте — веса и AdamW остаются float32 (master), считаем в fp16 →
        # в 2–3 раза быстрее на T4. Масштаб loss (динамический), чтобы мелкие градиенты не стали нулём.
        with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
            logits = forward(self.p, self.cfg, self._t(idx, torch.long), drop, self.gen if drop > 0 else None)
        loss = masked_ce(logits.float(), self._t(targets, torch.long), self._t(mask, torch.bool))
        (loss * self.scale if self.amp else loss).backward()
        return loss

    @torch.no_grad()
    def step(self, idx, targets, mask, lr, drop=0.0, max_norm=1.0):
        with torch.enable_grad():
            loss = self.loss_and_grads(idx, targets, mask, drop)
        grads = {k: v.grad for k, v in self.p.items()}
        if self.amp:
            for g in grads.values():
                g.div_(self.scale)
            if not all(torch.isfinite(g).all() for g in grads.values()):
                self.scale /= 2                                  # переполнение fp16 → шаг пропускаем
                return float(loss.detach()), float("nan")
            self._good += 1
            if self._good % 1000 == 0:
                self.scale = min(self.scale * 2, 2.0 ** 16)
        norm = math.sqrt(float(sum((g.double() ** 2).sum() for g in grads.values())))
        if norm > max_norm:                                     # обрезка градиента — как clip_grads
            s = max_norm / (norm + 1e-6)
            for g in grads.values():
                g.mul_(s)
        self.t += 1                                             # AdamW — как transformer.AdamW.step
        c1, c2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        for k, w in self.p.items():
            g = grads[k]
            self.m[k].mul_(self.b1).add_((1 - self.b1) * g)
            self.v[k].mul_(self.b2).add_((1 - self.b2) * g * g)
            upd = (self.m[k] / c1) / (torch.sqrt(self.v[k] / c2) + self.eps)
            if w.ndim == 2 and k != "wpe":
                upd = upd + self.wd * w
            w.sub_(lr * upd)
        return float(loss.detach()), norm

    def params_numpy(self):
        """Веса обратно в numpy — для экзамена, чекпойнтов и приложения (они не зависят от PyTorch)."""
        return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in self.p.items()}
