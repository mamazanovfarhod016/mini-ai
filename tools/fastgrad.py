"""fastgrad — «градиент без минусов: дополнить или немного изменить, чтобы скорость была предельно высокой» (25.10.2026).

Минусы градиента, уже измеренные у нас, и чем каждый закрываем (всё в одном прогоне, одна T4):
  1. железо простаивает (наш движок ×1)            → fp16 + SDPA                     (eff_lm: ×3 слов/с)
  2. шаг идёт в плохую сторону                      → Muon                            (eff_lm: ×2 на слово)
  3. стартует с шума                                → смысл слов PPMI-SVD на старте   (grad_weak: ×1,2)
  4. НЕ учит с одного раза (1 шаг: −3,5 % ошибки)   → память случаев с ВОРОТАМИ (включается только когда узнала случай)
  5. забывает старое, когда учит новое               → память не трогает веса → забывания нет по построению
ЧАСТЬ А (скорость): A = fp16+SDPA+Muon, B = A + смысл слов. Одинаковые MIN минут, loss на отложенных рассказах.
ЧАСТЬ Б (с одного раза + забывание) на обученной B: 200 новых фактов «Tim has a ball named <редкое слово>.» показаны ОДИН раз.
  Проверка: тот же вопрос («Tim has a ball named» → слово) и ПЕРЕФРАЗ («one day , tim played with the ball named» → слово);
  точность top-1, и loss на общем тексте ДО/ПОСЛЕ (забывание). Способы: grad×1 (каждый факт один раз), grad×K (столько повторов,
  сколько нужно градиенту), память (1 раз), память + grad×1.
Числа успеха (до запуска): память ≥ 0,9 на том же вопросе с забыванием ≤ 0,005; grad×1 — ожидаем мало; честно отметить перефраз.
Локально: SMOKE=1 python3 tools/fastgrad.py
"""
import os, re, io, json, math, time, copy, collections, urllib.request
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

E = os.environ.get; SMOKE = E("SMOKE") == "1"
OUT = E("OUT", "/kaggle/working" if os.path.isdir("/kaggle/working") else "/tmp/fastgrad"); os.makedirs(OUT, exist_ok=True)
MIN = float(E("MIN", 0.1 if SMOKE else 10)); D = int(E("D", 64 if SMOKE else 384)); NL = int(E("NL", 2 if SMOKE else 6)); NH = int(E("NH", 4 if SMOKE else 6))
T = int(E("T", 64 if SMOKE else 256)); BS = int(E("BS", 8 if SMOKE else 48)); V = int(E("V", 2000 if SMOKE else 8192)); N_VAL = int(E("N_VAL", 8 if SMOKE else 80))
DEV = "cuda" if torch.cuda.is_available() else "cpu"; AMP = DEV == "cuda"; T0 = time.time()
def log(*a):
    s = " ".join(str(x) for x in a); print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)
    open(os.path.join(OUT, "log.txt"), "a").write(s + "\n")
def sync():
    if DEV == "cuda": torch.cuda.synchronize()

def hf_parquet(repo, split):
    lst = json.loads(urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/api/datasets/{repo}/parquet", headers={"User-Agent": "mini-ai"}), timeout=60).read())
    import pyarrow.parquet as pq
    return pq.read_table(io.BytesIO(urllib.request.urlopen(urllib.request.Request(lst["default"][split][0], headers={"User-Agent": "mini-ai"}), timeout=900).read())).to_pandas()["text"].tolist()
val_txt = hf_parquet("roneneldan/TinyStories", "validation")
if SMOKE: tr_txt, val_txt = val_txt[:3000], val_txt[3000:3400]
else: tr_txt, val_txt = hf_parquet("roneneldan/TinyStories", "train")[: int(E("N_STORIES", 300000))], val_txt[:4000]
TOK = re.compile(r"[a-z]+(?:'[a-z]+)?|\d|[^\sa-z\d]"); tk = lambda s: TOK.findall(s.lower())
wc = collections.Counter(w for s in tr_txt[:100000] for w in tk(s))
itos = ["<unk>", "<eos>"] + [w for w, _ in wc.most_common(V - 2)]; stoi = {w: i for i, w in enumerate(itos)}; V = len(itos)
def stream(txts):
    out = []
    for s in txts: out += [stoi.get(w, 0) for w in tk(s)] + [1]
    return np.array(out, np.int64)
TR = stream(tr_txt); VA = stream(val_txt); del tr_txt; TRt = torch.tensor(TR, device=DEV)
VX = torch.tensor(np.stack([VA[j:j + T + 1] for j in np.random.default_rng(123).integers(0, len(VA) - T - 1, N_VAL)]), device=DEV)
log(f"данные: train {len(TR)/1e6:.1f}M слов, словарь {V}, {DEV}, {MIN:g} мин на вариант")

def ppmi_emb():
    sync(); t0 = time.time(); C = torch.zeros(V * V, device=DEV); src = TRt[: 20_000_000]
    for k in (1, 2):
        a, b = src[:-k], src[k:]; C += torch.bincount(a * V + b, minlength=V * V).float() / k; C += torch.bincount(b * V + a, minlength=V * V).float() / k
    C = C.view(V, V); r = C.sum(1, keepdim=True); c = C.sum(0, keepdim=True); N = C.sum()
    P = torch.clamp(torch.log(C * N / (r * c + 1e-9) + 1e-9), min=0); P[C == 0] = 0; del C
    U, S, _ = torch.svd_lowrank(P, q=D, niter=4); Em = U * S.sqrt(); sync(); return Em / Em.std() * 0.02, time.time() - t0

def rope(x, cos, sin):
    x0, x1 = x[..., 0::2], x[..., 1::2]; return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], -1).flatten(-2)
class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.n1 = nn.LayerNorm(D); s.n2 = nn.LayerNorm(D); s.qkv = nn.Linear(D, 3 * D); s.o = nn.Linear(D, D); s.f1 = nn.Linear(D, 4 * D); s.f2 = nn.Linear(4 * D, D)
    def forward(s, x, cos, sin):
        B, L, d = x.shape; q, k, v = s.qkv(s.n1(x)).view(B, L, 3, NH, d // NH).permute(2, 0, 3, 1, 4)
        x = x + s.o(F.scaled_dot_product_attention(rope(q, cos, sin), rope(k, cos, sin), v, is_causal=True).transpose(1, 2).reshape(B, L, d))
        return x + s.f2(F.gelu(s.f1(s.n2(x)), approximate="tanh"))
class GPT(nn.Module):
    def __init__(s):
        super().__init__(); s.wte = nn.Embedding(V, D); nn.init.normal_(s.wte.weight, std=0.02); s.blocks = nn.ModuleList(Block() for _ in range(NL)); s.nf = nn.LayerNorm(D)
        dh = D // NH; th = 10000 ** (-torch.arange(0, dh, 2).float() / dh); ang = torch.arange(T).float()[:, None] * th[None]
        s.register_buffer("cos", ang.cos(), persistent=False); s.register_buffer("sin", ang.sin(), persistent=False)
        for mod in s.modules():
            if isinstance(mod, nn.Linear): nn.init.normal_(mod.weight, std=0.02); nn.init.zeros_(mod.bias)
        for b in s.blocks: nn.init.normal_(b.o.weight, std=0.02 / math.sqrt(2 * NL)); nn.init.normal_(b.f2.weight, std=0.02 / math.sqrt(2 * NL))
    def forward(s, idx):
        x = s.wte(idx); L = idx.shape[1]
        for b in s.blocks: x = b(x, s.cos[:L].to(x.dtype), s.sin[:L].to(x.dtype))
        h = s.nf(x); return h @ s.wte.weight.T, h
def run(m, X):
    with torch.autocast(DEV, dtype=torch.float16, enabled=AMP):
        lg, h = m(X)
    return lg.float(), h.float()

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
                buf = s.state[p].setdefault("m", torch.zeros_like(p.grad)); buf.mul_(g["mom"]).add_(p.grad)
                p.add_(ns5(p.grad + g["mom"] * buf).to(p.dtype), alpha=-g["lr"] * max(1, p.shape[0] / p.shape[1]) ** 0.5)

@torch.no_grad()
def val_loss(m):
    m.eval(); tot = 0.0
    for i in range(0, N_VAL, 8):
        X = VX[i:i + 8]; lg, _ = run(m, X[:, :-1]); tot += F.cross_entropy(lg.reshape(-1, V), X[:, 1:].reshape(-1)).item() * len(X)
    m.train(); return tot / N_VAL

def train(embinit):
    torch.manual_seed(0); m = GPT().to(DEV); pre = 0.0
    if embinit: Em, pre = ppmi_emb(); m.wte.weight.data.copy_(Em)
    mats = [p for n, p in m.named_parameters() if p.ndim == 2 and "wte" not in n]; rest = [p for n, p in m.named_parameters() if not (p.ndim == 2 and "wte" not in n)]
    opts = [Muon(mats, lr=0.02), torch.optim.AdamW(rest, lr=2e-3, betas=(0.9, 0.95), weight_decay=0.0)]; base = [[g["lr"] for g in o.param_groups] for o in opts]
    scaler = torch.amp.GradScaler(enabled=AMP); rng = np.random.default_rng(0); tt = pre / 60; step = 0; curve = []; nxt = 0.0
    while True:
        if tt >= nxt or tt >= MIN:
            curve.append((round(tt, 2), val_loss(m))); nxt += MIN / 10
            if tt >= MIN: break
        sync(); t_s = time.time(); fr = tt / MIN; mult = min(1.0, (step + 1) / 100) * (0.5 * (1 + math.cos(math.pi * min(1.0, fr))) * 0.9 + 0.1)
        for o, bl in zip(opts, base):
            for g, b in zip(o.param_groups, bl): g["lr"] = b * mult
        ix = rng.integers(0, len(TR) - T - 1, BS); X = torch.tensor(np.stack([TR[j:j + T + 1] for j in ix]), device=DEV)
        lg, _ = run(m, X[:, :-1]); loss = F.cross_entropy(lg.reshape(-1, V), X[:, 1:].reshape(-1))
        for o in opts: o.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        for o in opts: scaler.unscale_(o)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        for o in opts: scaler.step(o)
        scaler.update(); sync(); tt += (time.time() - t_s) / 60; step += 1
    return m, curve, step, pre

# ================= ЧАСТЬ А: скорость
RES = {}
mA, cA, sA, _ = train(False); log(f"[A fp16+SDPA+Muon] шагов {sA}, loss {cA[-1][1]:.3f} | " + " ".join(f"{a:.1f}м:{b:.2f}" for a, b in cA[::2]))
del mA
if DEV == "cuda": torch.cuda.empty_cache()
mB, cB, sB, preB = train(True); log(f"[B A+смысл слов] шагов {sB}, формула {preB:.1f} с, loss {cB[-1][1]:.3f} | " + " ".join(f"{a:.1f}м:{b:.2f}" for a, b in cB[::2]))
hit = next((a for a, b in cB if b <= cA[-1][1]), None)
RES["speed"] = dict(A=cA, B=cB, stepsA=sA, stepsB=sB, B_hits_A_at=hit)
log(f"ИТОГ А: B догоняет финал A ({cA[-1][1]:.3f}) на {hit} мин из {MIN:g} → " + (f"×{MIN/max(hit, MIN/100):.2f}" if hit is not None else "не догнал"))

# ================= ЧАСТЬ Б: с одного раза + забывание
names = [w for w in "tim lily ben sue tom anna max mia sam lucy jack emma bob kate leo ella joe amy dan zoe".split() if w in stoi][:20]
objs = [w for w in "ball dog cat toy doll car boat kite bear fish bird hat".split() if w in stoi][:10]
rare = [w for w in itos[1500:6000] if re.fullmatch(r"[a-z]{4,}", w) and w not in names and w not in objs]
rr = np.random.default_rng(42); vals = list(rr.choice(rare, len(names) * len(objs), replace=False))
FACTS = [(p, o, vals[i * len(objs) + j]) for i, p in enumerate(names) for j, o in enumerate(objs)]
teach = [f"{p} has a {o} named {x} ." for p, o, x in FACTS]
q_same = [(f"{p} has a {o} named", x) for p, o, x in FACTS]
q_para = [(f"one day , {p} played with the {o} named", x) for p, o, x in FACTS]
enc = lambda s: [stoi.get(w, 0) for w in tk(s)]
log(f"[факты] {len(FACTS)} новых: напр. «{teach[0]}» / перефраз «{q_para[0][0]} → {q_para[0][1]}»")

@torch.no_grad()
def last_state(m, prompts):
    m.eval(); hs, lps = [], []
    for s in prompts:
        ids = torch.tensor([enc(s)], device=DEV); lg, h = run(m, ids); hs.append(F.normalize(h[0, -1], dim=-1)); lps.append(torch.log_softmax(lg[0, -1], -1))
    m.train(); return torch.stack(hs), torch.stack(lps)
def score(m, qs, mem=None, gate=0.9, lam=0.9):
    H, LP = last_state(m, [q for q, _ in qs]); tgt = torch.tensor([stoi[x] for _, x in qs], device=DEV); P = LP.exp(); used = 0
    if mem is not None:
        K, Vv = mem; sim = H @ K.T; best, j = sim.max(-1); on = best >= gate; used = int(on.sum())
        Pm = torch.zeros_like(P); Pm[torch.arange(len(qs)), Vv[j]] = 1.0
        P = torch.where(on[:, None], (1 - lam) * P + lam * Pm, P)
    top1 = float((P.argmax(-1) == tgt).float().mean()); lp = float(torch.log(P[torch.arange(len(qs)), tgt] + 1e-9).mean())
    return dict(top1=top1, logp=lp, mem_used=used)
IDF = None
def _idf():
    global IDF
    if IDF is None:
        cnt = torch.bincount(TRt[: 20_000_000], minlength=V).float(); IDF = torch.log((cnt.sum() + V) / (cnt + 1)).cpu().numpy()
    return IDF
def score_words(m, qs, keys, vals, gate=0.6, lam=0.9):
    """память по СЛОВАМ (как analogy.py): ключ = набор слов факта; похожесть = вес общих слов / вес слов ключа (вес = редкость слова)"""
    idf = _idf(); _, LP = last_state(m, [q for q, _ in qs]); tgt = torch.tensor([stoi[x] for _, x in qs], device=DEV); P = LP.exp(); used = 0
    for i, (q, _) in enumerate(qs):
        qw = set(enc(q)); best, bj = -1.0, -1
        for j, kw in enumerate(keys):
            sim = sum(idf[w] for w in kw & qw) / (sum(idf[w] for w in kw) + 1e-9)
            if sim > best: best, bj = sim, j
        if best >= gate:
            used += 1; pm = torch.zeros(V, device=DEV); pm[vals[bj]] = 1.0; P[i] = (1 - lam) * P[i] + lam * pm
    return dict(top1=float((P.argmax(-1) == tgt).float().mean()), used=used)
def build_memory(m):
    H, _ = last_state(m, [t.rsplit(" named", 1)[0] + " named" for t in teach]); return H, torch.tensor([stoi[x] for _, _, x in FACTS], device=DEV)
def grad_teach(m, epochs, lr=1e-3):
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.0); seqs = [enc(t) for t in teach]; n = 0
    for ep in range(epochs):
        order = np.random.default_rng(ep).permutation(len(seqs))
        for i in range(0, len(order), 20):
            ch = [seqs[k] for k in order[i:i + 20]]; L = max(map(len, ch)); X = torch.full((len(ch), L), 1, device=DEV)
            for r_, s_ in enumerate(ch): X[r_, :len(s_)] = torch.tensor(s_)
            lg, _ = run(m, X[:, :-1]); loss = F.cross_entropy(lg.reshape(-1, V), X[:, 1:].reshape(-1)); opt.zero_grad(); loss.backward(); opt.step(); n += 1
    return n

base_val = val_loss(mB); w0 = copy.deepcopy(mB.state_dict()); rows = []
def row(name, m, mem=None, steps=0, sec=0.0):
    a = score(m, q_same, mem); b = score(m, q_para, mem); vl = val_loss(m)
    r = dict(name=name, same=a["top1"], para=b["top1"], same_logp=a["logp"], val=vl, forget=vl - base_val, mem_used=(a["mem_used"], b["mem_used"]), steps=steps, sec=sec); rows.append(r)
    log(f"  {name:<22} тот же вопрос {a['top1']:.0%} | перефраз {b['top1']:.0%} | забывание {vl-base_val:+.4f} (общий loss {vl:.3f})" + (f" | память включилась {a['mem_used']}/{len(FACTS)} и {b['mem_used']}/{len(FACTS)}" if mem is not None else "") + (f" | шагов {steps}, {sec:.1f} с" if steps or sec else ""))
log("ИТОГ Б (200 фактов показаны один раз):")
row("до учёбы", mB)
for ep in (1, 3, 10, 30):
    mB.load_state_dict(w0); sync(); t_ = time.time(); n = grad_teach(mB, ep); sync(); row(f"градиент ×{ep}", mB, steps=n, sec=time.time() - t_)
mB.load_state_dict(w0); sync(); t_ = time.time(); mem = build_memory(mB); sync(); row("память ×1", mB, mem, sec=time.time() - t_)
mB.load_state_dict(w0); grad_teach(mB, 1); mem2 = build_memory(mB); row("память ×1 + градиент ×1", mB, mem2)
mB.load_state_dict(w0); wk = [set(enc(t.rsplit(" named", 1)[0] + " named")) for t in teach]; wv = [stoi[x] for _, _, x in FACTS]
a = score_words(mB, q_same, wk, wv); b = score_words(mB, q_para, wk, wv)
rows.append(dict(name="память по смыслу слов ×1", same=a["top1"], para=b["top1"], forget=0.0, mem_used=(a["used"], b["used"])))
log(f"  {'память по смыслу слов ×1':<22} тот же вопрос {a['top1']:.0%} | перефраз {b['top1']:.0%} | забывание +0.0000 | включилась {a['used']}/{len(FACTS)} и {b['used']}/{len(FACTS)}")
RES["oneshot"] = rows
json.dump(RES, open(os.path.join(OUT, "metrics.json"), "w"), indent=1, default=float); log("[done]")
