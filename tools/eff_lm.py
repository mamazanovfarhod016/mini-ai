"""eff_lm — «градиент, но в несколько раз эффективнее» (25.10.2026, его «поэкспериментировать, чтоб он в несколько раз эффективнее был»).

Вопрос: насколько быстрее (по минутам GPU до той же точности) можно учить НАШ трансформер, меняя только способ обучения.
Задача-линейка: язык TinyStories (следующее слово), проверочный loss на отложенных рассказах. Модель одинаковая по размеру.
Каждый вариант получает ОДИНАКОВЫЙ бюджет MIN минут на одной T4. Варианты накопительные:
  v0 база     — как сейчас наш движок: fp32, внимание вручную, AdamW lr 5e-4
  v1 скорость — + fp16 (autocast + GradScaler) + SDPA (ядро внимания PyTorch): та же математика, быстрее железо
  v2 шаг      — v1 + AdamW lr 2e-3, разогрев + косинус (правильный размер шага)
  v3 Muon     — v1 + Muon для матриц (ортогонализация шага Ньютоном–Шульцем, рекорды modded-nanogpt), AdamW для эмбеддингов
  v4 архит.   — v3 + RMSNorm + QK-norm + ReLU² в MLP (современные мелочи, стабильнее при большом шаге)
Число успеха (записано до запуска): «во сколько раз быстрее» = MIN / (время, когда вариант впервые достиг ФИНАЛЬНОГО loss базы v0).
≥ 2× — заметно, ≥ 3× — «в несколько раз». Честно: часть выигрыша (v1) — просто железо, не алгоритм; это будет видно отдельной строкой.
Локально: SMOKE=1 python3 tools/eff_lm.py (CPU, секунды, числа ничего не значат).
"""
import os, re, io, json, math, time, collections, urllib.request
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

E = os.environ.get; SMOKE = E("SMOKE") == "1"
OUT = E("OUT", "/kaggle/working" if os.path.isdir("/kaggle/working") else "/tmp/eff_lm"); os.makedirs(OUT, exist_ok=True)
MIN = float(E("MIN", 0.15 if SMOKE else 7)); D = int(E("D", 64 if SMOKE else 384)); NL = int(E("NL", 2 if SMOKE else 6)); NH = int(E("NH", 4 if SMOKE else 6))
T = int(E("T", 64 if SMOKE else 256)); BS = int(E("BS", 8 if SMOKE else 48)); V = int(E("V", 2000 if SMOKE else 8192))
VARS = (E("VARS", "v0,v1,v2,v3,v4")).split(","); DEV = "cuda" if torch.cuda.is_available() else "cpu"; T0 = time.time()
def log(*a):
    s = " ".join(str(x) for x in a); print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)
    open(os.path.join(OUT, "log.txt"), "a").write(s + "\n")

# ---------------- данные: TinyStories (HF parquet), словарь по словам — один для всех вариантов
def hf_parquet(repo, split):
    lst = json.loads(urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/api/datasets/{repo}/parquet", headers={"User-Agent": "mini-ai"}), timeout=60).read())
    url = lst["default"][split][0]; import pyarrow.parquet as pq
    return pq.read_table(io.BytesIO(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "mini-ai"}), timeout=900).read())).to_pandas()["text"].tolist()
val_txt = hf_parquet("roneneldan/TinyStories", "validation")
tr_txt = val_txt[:3000] if SMOKE else hf_parquet("roneneldan/TinyStories", "train")[: int(E("N_STORIES", 300000))]
if SMOKE: val_txt = val_txt[3000:3400]
else: val_txt = val_txt[:4000]
TOK = re.compile(r"[a-z]+(?:'[a-z]+)?|\d|[^\sa-z\d]")
tk = lambda s: TOK.findall(s.lower())
wc = collections.Counter(w for s in tr_txt[:100000] for w in tk(s))
itos = ["<unk>", "<eos>"] + [w for w, _ in wc.most_common(V - 2)]; stoi = {w: i for i, w in enumerate(itos)}; V = len(itos)
def stream(txts):
    out = []
    for s in txts: out += [stoi.get(w, 0) for w in tk(s)] + [1]
    return np.array(out, np.int64)
TR = stream(tr_txt); VA = stream(val_txt); del tr_txt
log(f"данные: train {len(TR)/1e6:.1f}M слов, val {len(VA)/1e3:.0f}k, словарь {V}, неизвестных {np.mean(TR==0):.1%}; устройство {DEV}")
g_val = np.random.default_rng(123); VAL_IDX = g_val.integers(0, len(VA) - T - 1, 40 if not SMOKE else 8)

# ---------------- модель (наш GPT: RoPE, pre-norm, связанные веса); переключатели архитектуры
class Block(nn.Module):
    def __init__(s, d, h, modern):
        super().__init__(); s.h = h; s.modern = modern
        s.n1 = nn.RMSNorm(d) if modern else nn.LayerNorm(d); s.n2 = nn.RMSNorm(d) if modern else nn.LayerNorm(d)
        s.qkv = nn.Linear(d, 3 * d); s.o = nn.Linear(d, d); s.f1 = nn.Linear(d, 4 * d); s.f2 = nn.Linear(4 * d, d)
        nn.init.normal_(s.o.weight, std=0.02 / math.sqrt(2 * NL)); nn.init.normal_(s.f2.weight, std=0.02 / math.sqrt(2 * NL))
    def forward(s, x, cos, sin, sdpa):
        B, L, d = x.shape; dh = d // s.h
        q, k, v = s.qkv(s.n1(x)).view(B, L, 3, s.h, dh).permute(2, 0, 3, 1, 4)
        if s.modern: q, k = F.rms_norm(q, (dh,)), F.rms_norm(k, (dh,))
        q, k = rope(q, cos, sin), rope(k, cos, sin)
        if sdpa: y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            S = (q @ k.transpose(-1, -2)) / math.sqrt(dh); S = S.masked_fill(~torch.ones(L, L, dtype=torch.bool, device=x.device).tril(), -1e4)
            y = torch.softmax(S.float(), -1).to(v.dtype) @ v
        x = x + s.o(y.transpose(1, 2).reshape(B, L, d))
        h = s.f1(s.n2(x)); h = F.relu(h).square() if s.modern else F.gelu(h, approximate="tanh")
        return x + s.f2(h)
def rope(x, cos, sin):
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], -1).flatten(-2)
class GPT(nn.Module):
    def __init__(s, modern):
        super().__init__(); s.wte = nn.Embedding(V, D); nn.init.normal_(s.wte.weight, std=0.02)
        s.blocks = nn.ModuleList(Block(D, NH, modern) for _ in range(NL)); s.nf = nn.RMSNorm(D) if modern else nn.LayerNorm(D)
        dh = D // NH; th = 10000 ** (-torch.arange(0, dh, 2).float() / dh); ang = torch.arange(T).float()[:, None] * th[None]
        s.register_buffer("cos", ang.cos(), persistent=False); s.register_buffer("sin", ang.sin(), persistent=False)
        for mod in s.modules():
            if isinstance(mod, nn.Linear): nn.init.normal_(mod.weight, std=0.02); nn.init.zeros_(mod.bias)
        for b in s.blocks:
            nn.init.normal_(b.o.weight, std=0.02 / math.sqrt(2 * NL)); nn.init.normal_(b.f2.weight, std=0.02 / math.sqrt(2 * NL))
    def forward(s, idx, sdpa):
        x = s.wte(idx); L = idx.shape[1]
        for b in s.blocks: x = b(x, s.cos[:L].to(x.dtype), s.sin[:L].to(x.dtype), sdpa)
        return s.nf(x) @ s.wte.weight.T

# ---------------- Muon: шаг для матриц = ортогонализованный импульс (Ньютон–Шульц, 5 итераций)
def ns5(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315; X = G.float() / (G.norm() + 1e-7); tr = X.shape[0] > X.shape[1]
    if tr: X = X.T
    for _ in range(steps): A = X @ X.T; X = a * X + (b * A + c * A @ A) @ X
    return X.T if tr else X
class Muon(torch.optim.Optimizer):
    def __init__(s, params, lr=0.02, mom=0.95): super().__init__(params, dict(lr=lr, mom=mom))
    @torch.no_grad()
    def step(s):
        for g in s.param_groups:
            for p in g["params"]:
                if p.grad is None: continue
                st = s.state[p]; buf = st.setdefault("m", torch.zeros_like(p.grad)); buf.mul_(g["mom"]).add_(p.grad)
                u = ns5(p.grad + g["mom"] * buf)                       # Нестеров
                p.add_(u.to(p.dtype), alpha=-g["lr"] * max(1, p.shape[0] / p.shape[1]) ** 0.5)

CFG = {"v0": dict(amp=False, sdpa=False, lr=5e-4, sched=False, muon=False, modern=False),
       "v1": dict(amp=True, sdpa=True, lr=5e-4, sched=False, muon=False, modern=False),
       "v2": dict(amp=True, sdpa=True, lr=2e-3, sched=True, muon=False, modern=False),
       "v3": dict(amp=True, sdpa=True, lr=2e-3, sched=True, muon=True, modern=False),
       "v4": dict(amp=True, sdpa=True, lr=2e-3, sched=True, muon=True, modern=True)}

@torch.no_grad()
def val_loss(m, c):
    m.eval(); tot = 0.0
    for i in range(0, len(VAL_IDX), 8):
        ix = VAL_IDX[i:i + 8]; X = torch.tensor(np.stack([VA[j:j + T + 1] for j in ix]), device=DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=c["amp"] and DEV == "cuda"):
            lg = m(X[:, :-1], c["sdpa"])
        tot += F.cross_entropy(lg.float().reshape(-1, V), X[:, 1:].reshape(-1)).item() * len(ix)
    m.train(); return tot / len(VAL_IDX)

RES = {}
for name in VARS:
    c = CFG[name]; torch.manual_seed(0); m = GPT(c["modern"]).to(DEV); rng = np.random.default_rng(0)
    mats = [p for n, p in m.named_parameters() if p.ndim == 2 and "wte" not in n]; rest = [p for n, p in m.named_parameters() if not (p.ndim == 2 and "wte" not in n)]
    opts = ([Muon(mats, lr=0.02), torch.optim.AdamW(rest, lr=c["lr"], betas=(0.9, 0.95), weight_decay=0.0)] if c["muon"]
            else [torch.optim.AdamW(m.parameters(), lr=c["lr"], betas=(0.9, 0.95), weight_decay=0.01)])
    base_lr = [[g["lr"] for g in o.param_groups] for o in opts]
    scaler = torch.amp.GradScaler(enabled=c["amp"] and DEV == "cuda")
    n_par = sum(p.numel() for p in m.parameters()); curve = []; step = 0; toks = 0; t_s = time.time(); next_ev = 0.0
    if DEV == "cuda": torch.cuda.synchronize()
    while True:
        el = (time.time() - t_s) / 60; fr = el / MIN
        if el >= next_ev or fr >= 1:
            vl = val_loss(m, c); curve.append((round(el, 2), step, toks, vl)); next_ev += MIN / 14
            if fr >= 1: break
        ix = rng.integers(0, len(TR) - T - 1, BS); X = torch.tensor(np.stack([TR[j:j + T + 1] for j in ix]), device=DEV)
        mult = (min(1.0, (step + 1) / 100) * (0.5 * (1 + math.cos(math.pi * min(1.0, fr))) * 0.9 + 0.1)) if c["sched"] else 1.0
        for o, bl in zip(opts, base_lr):
            for g, b in zip(o.param_groups, bl): g["lr"] = b * mult
        with torch.autocast(DEV, dtype=torch.float16, enabled=c["amp"] and DEV == "cuda"):
            lg = m(X[:, :-1], c["sdpa"])
        loss = F.cross_entropy(lg.float().reshape(-1, V), X[:, 1:].reshape(-1))
        for o in opts: o.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        for o in opts: scaler.unscale_(o)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        for o in opts: scaler.step(o)
        scaler.update(); step += 1; toks += BS * T
    tps = toks / (MIN * 60)
    RES[name] = dict(cfg=c, params=n_par, steps=step, tokens=toks, tok_per_s=tps, final=curve[-1][3], curve=curve)
    log(f"[{name}] {n_par/1e6:.1f}M параметров, шагов {step}, {tps/1e3:.0f}k слов/с, итоговый val loss {curve[-1][3]:.3f} | кривая " + " ".join(f"{e:.1f}м:{l:.2f}" for e, _, _, l in curve[::3]))
    del m, opts
    if DEV == "cuda": torch.cuda.empty_cache()

# ---------------- во сколько раз быстрее достигнут финальный loss базы
if "v0" in RES:
    tgt = RES["v0"]["final"]
    log(f"ИТОГ: цель = финальный loss базы v0 {tgt:.3f} за {MIN:g} мин")
    for n, r in RES.items():
        hit = next((e for e, _, _, l in r["curve"] if l <= tgt), None)
        r["speedup"] = (MIN / hit) if hit else None
        log(f"  {n}: loss {r['final']:.3f}; цель достигнута на {hit if hit is not None else '—'} мин → " + (f"×{MIN/hit:.1f}" if hit else "не достиг") +
            f"; {r['tok_per_s']/1e3:.0f}k слов/с (×{r['tok_per_s']/RES['v0']['tok_per_s']:.1f} от скорости железа)")
json.dump(dict(min=MIN, d=D, layers=NL, heads=NH, T=T, bs=BS, V=V, res=RES), open(os.path.join(OUT, "metrics.json"), "w"), indent=1, default=float)
log("[done]")
