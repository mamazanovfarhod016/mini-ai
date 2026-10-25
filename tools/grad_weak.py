"""grad_weak — ПОЧЕМУ градиенту нужно так много шагов и чем это заполнить (25.10.2026, его «поймём почему требует так много
шагов, найдём эту слабость и чем-то заполним или дополним»).

ДИАГНОСТИКА (на базовом прогоне; все числа — измерения):
  D1 «переучивает формулу»: сколько минут градиент тратит, чтобы дойти до loss, который ПОДСЧЁТ пар слов (биграмма, без градиента)
     даёт за секунды. Доля бюджета, ушедшая на то, что формула знает сразу.
  D2 «с одного раза не запоминает»: loss на куске текста → после ОДНОГО шага на нём → после 50 других шагов (забывание);
     то же для памяти случаев (кусок положен в память один раз).
  D3 «редкие слова»: loss по корзинам частоты (100 частых / 100–1000 / редкие) — модель против формулы.
ЗАПЛАТКИ = наши механизмы без градиента ВНУТРИ градиентной модели; одинаковые минуты ОБУЧЕНИЯ на T4 (время проверок не считается,
время формул — считается):
  base    — трансформер, fp16 + SDPA, AdamW lr 2e-3 + разогрев + косинус
  ngram   — к ответу модели прибавлен log P(слово | предыдущее) из подсчёта пар: градиент учит только то, чего формула не знает
  embinit — эмбеддинги стартуют со смысла слов из подсчёта соседей (PPMI-SVD, как наш meaning.py), а не со случайного шума
  combo   — ngram + embinit
  +память — у base и combo: память случаев (скрытое состояние → следующее слово, N_MEM пар из train, без обучения);
            ответ = смесь модели и ближайших случаев (λ, τ подбираются на первой половине проверки, число — на второй).
Число успеха (записано до запуска): «во сколько раз быстрее» = минуты base до её финального loss / минуты варианта до того же loss.
≥ 2× — слабость найдена и закрыта. Локально: SMOKE=1 python3 tools/grad_weak.py (числа ничего не значат).
"""
import os, re, io, json, math, time, collections, urllib.request
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

E = os.environ.get; SMOKE = E("SMOKE") == "1"
OUT = E("OUT", "/kaggle/working" if os.path.isdir("/kaggle/working") else "/tmp/grad_weak"); os.makedirs(OUT, exist_ok=True)
MIN = float(E("MIN", 0.1 if SMOKE else 8)); D = int(E("D", 64 if SMOKE else 384)); NL = int(E("NL", 2 if SMOKE else 6)); NH = int(E("NH", 4 if SMOKE else 6))
T = int(E("T", 64 if SMOKE else 256)); BS = int(E("BS", 8 if SMOKE else 48)); V = int(E("V", 2000 if SMOKE else 8192)); LR = float(E("LR", 2e-3))
N_MEM = int(E("N_MEM", 4000 if SMOKE else 400000)); N_VAL = int(E("N_VAL", 8 if SMOKE else 80))
RUNS = E("RUNS", "base,ngram,embinit,combo").split(","); KNN_RUNS = set(E("KNN_RUNS", "base,combo").split(","))
DEV = "cuda" if torch.cuda.is_available() else "cpu"; AMP = DEV == "cuda"; KD = torch.float16 if AMP else torch.float32; T0 = time.time()
def log(*a):
    s = " ".join(str(x) for x in a); print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)
    open(os.path.join(OUT, "log.txt"), "a").write(s + "\n")
def sync():
    if DEV == "cuda": torch.cuda.synchronize()

# ---------------- данные (TinyStories), словарь по словам: индекс = ранг частоты
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
TR = stream(tr_txt); VA = stream(val_txt); del tr_txt
TRt = torch.tensor(TR, device=DEV)
VAL_IDX = np.random.default_rng(123).integers(0, len(VA) - T - 1, N_VAL)
VX = torch.tensor(np.stack([VA[j:j + T + 1] for j in VAL_IDX]), device=DEV)     # (N_VAL, T+1)
HALF = N_VAL // 2
log(f"данные: train {len(TR)/1e6:.1f}M слов, val {len(VA)/1e3:.0f}k, словарь {V}, устройство {DEV}, бюджет обучения {MIN:g} мин на вариант")

# ---------------- ФОРМУЛЫ без градиента: подсчёт слов и пар (D1 + заплатка ngram)
sync(); t0 = time.time()
cnt = torch.bincount(TRt[:-1] * V + TRt[1:], minlength=V * V).float().view(V, V)
uni = torch.bincount(TRt, minlength=V).float()
LB = torch.log((cnt + 0.1) / (cnt.sum(1, keepdim=True) + 0.1 * V)).to(KD); LU = torch.log((uni + 1) / (uni.sum() + V)); del cnt
sync(); BIGRAM_SEC = time.time() - t0
tgt_all = VX[:, 1:]; uni_loss = float(-LU[tgt_all].mean()); bi_tok = -LB[VX[:, :-1], tgt_all].float(); bi_loss = float(bi_tok.mean())
log(f"[формула] подсчёт пар за {BIGRAM_SEC:.1f} с: loss униграммы {uni_loss:.3f}, биграммы {bi_loss:.3f}")

def ppmi_emb():
    sync(); t0 = time.time(); C = torch.zeros(V * V, device=DEV); src = TRt[: 20_000_000]
    for k in (1, 2):
        a, b = src[:-k], src[k:]
        C += torch.bincount(a * V + b, minlength=V * V).float() / k; C += torch.bincount(b * V + a, minlength=V * V).float() / k
    C = C.view(V, V); r = C.sum(1, keepdim=True); c = C.sum(0, keepdim=True); N = C.sum()
    P = torch.clamp(torch.log(C * N / (r * c + 1e-9) + 1e-9), min=0); P[C == 0] = 0; del C
    U, S, _ = torch.svd_lowrank(P, q=D, niter=4); Em = U * S.sqrt(); Em = Em / Em.std() * 0.02
    sync(); return Em, time.time() - t0
EMB, EMB_SEC = ppmi_emb() if any(r in ("embinit", "combo") for r in RUNS) else (None, 0.0)
if EMB is not None:
    probe = [w for w in ("cat", "dog", "happy", "sad", "ball", "toy") if w in stoi]
    if len(probe) >= 4:
        en = F.normalize(EMB, dim=-1); s_ = lambda a, b: float(en[stoi[a]] @ en[stoi[b]])
        log(f"[формула] смысл слов (PPMI-SVD) за {EMB_SEC:.1f} с; похожесть cat–dog {s_('cat','dog'):.2f}, happy–sad {s_('happy','sad'):.2f}, cat–happy {s_('cat','happy'):.2f}")

# ---------------- модель (наш GPT: RoPE, pre-LN, связанные веса)
def rope(x, cos, sin):
    x0, x1 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], -1).flatten(-2)
class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.n1 = nn.LayerNorm(D); s.n2 = nn.LayerNorm(D)
        s.qkv = nn.Linear(D, 3 * D); s.o = nn.Linear(D, D); s.f1 = nn.Linear(D, 4 * D); s.f2 = nn.Linear(4 * D, D)
    def forward(s, x, cos, sin):
        B, L, d = x.shape; dh = d // NH
        q, k, v = s.qkv(s.n1(x)).view(B, L, 3, NH, dh).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(rope(q, cos, sin), rope(k, cos, sin), v, is_causal=True)
        x = x + s.o(y.transpose(1, 2).reshape(B, L, d))
        return x + s.f2(F.gelu(s.f1(s.n2(x)), approximate="tanh"))
class GPT(nn.Module):
    def __init__(s):
        super().__init__(); s.wte = nn.Embedding(V, D); nn.init.normal_(s.wte.weight, std=0.02)
        s.blocks = nn.ModuleList(Block() for _ in range(NL)); s.nf = nn.LayerNorm(D)
        dh = D // NH; th = 10000 ** (-torch.arange(0, dh, 2).float() / dh); ang = torch.arange(T).float()[:, None] * th[None]
        s.register_buffer("cos", ang.cos(), persistent=False); s.register_buffer("sin", ang.sin(), persistent=False)
        for mod in s.modules():
            if isinstance(mod, nn.Linear): nn.init.normal_(mod.weight, std=0.02); nn.init.zeros_(mod.bias)
        for b in s.blocks: nn.init.normal_(b.o.weight, std=0.02 / math.sqrt(2 * NL)); nn.init.normal_(b.f2.weight, std=0.02 / math.sqrt(2 * NL))
    def forward(s, idx):
        x = s.wte(idx); L = idx.shape[1]
        for b in s.blocks: x = b(x, s.cos[:L].to(x.dtype), s.sin[:L].to(x.dtype))
        h = s.nf(x); return h @ s.wte.weight.T, h

def logits_of(m, X, ngram):
    with torch.autocast(DEV, dtype=torch.float16, enabled=AMP):
        lg, h = m(X)
    lg = lg.float()
    if ngram: lg = lg + LB[X].float()
    return lg, h
def batch(rng):
    ix = rng.integers(0, len(TR) - T - 1, BS); return torch.tensor(np.stack([TR[j:j + T + 1] for j in ix]), device=DEV)

# ---------------- память случаев (без обучения): ключ = нормированное скрытое состояние, значение = следующее слово
@torch.no_grad()
def build_mem(m, ngram, extra=None):
    m.eval(); rng = np.random.default_rng(5); Ks, Vs = [], []
    for _ in range(max(1, N_MEM // (BS * T))):
        X = batch(rng); _, h = logits_of(m, X[:, :-1], ngram); Ks.append(F.normalize(h.float(), dim=-1).to(KD).reshape(-1, D)); Vs.append(X[:, 1:].reshape(-1))
    if extra is not None:
        _, h = logits_of(m, extra[:, :-1], ngram); Ks.append(F.normalize(h.float(), dim=-1).to(KD).reshape(-1, D)); Vs.append(extra[:, 1:].reshape(-1))
    m.train(); return torch.cat(Ks), torch.cat(Vs)
@torch.no_grad()
def knn_top(q, K, k=32):
    tops, idxs = [], []
    for i in range(0, len(q), 1024):
        s = (q[i:i + 1024] @ K.T).float(); t_, i_ = s.topk(k, dim=-1); tops.append(t_); idxs.append(i_)
    return torch.cat(tops), torch.cat(idxs)
def knn_p_target(top, idx, Vmem, tgt, tau):
    """вероятность ВЕРНОГО слова по памяти: сумма весов соседей, у которых следующее слово = цель"""
    w = torch.softmax(top / tau, -1); return (w * (Vmem[idx] == tgt[:, None]).float()).sum(-1)

@torch.no_grad()
def evaluate(m, ngram, knn=False, per_token=False):
    m.eval(); lps, hs = [], []
    for i in range(0, N_VAL, 8):
        X = VX[i:i + 8]; lg, h = logits_of(m, X[:, :-1], ngram)
        lps.append(torch.log_softmax(lg, -1).gather(-1, X[:, 1:, None])[..., 0]); hs.append(h.float())
    m.train(); lp = torch.cat(lps)                                        # (N_VAL, T) log p модели для верного слова
    out = dict(loss=float(-lp.mean()), loss_b=float(-lp[HALF:].mean()))
    if per_token: out["tok"] = -lp
    if knn:
        K, Vm = build_mem(m, ngram); q = F.normalize(torch.cat(hs), dim=-1).to(KD).reshape(-1, D); top, idx = knn_top(q, K)
        tgt = VX[:, 1:].reshape(-1); pm = lp.reshape(-1).exp(); n_a = HALF * T; best = None
        for tau in (0.02, 0.05, 0.1):
            pk = knn_p_target(top, idx, Vm, tgt, tau)
            for lam in (0.1, 0.25, 0.4, 0.6):
                la = float(-torch.log((1 - lam) * pm[:n_a] + lam * pk[:n_a] + 1e-9).mean())
                if best is None or la < best[0]: best = (la, tau, lam)
        _, tau, lam = best; pk = knn_p_target(top, idx, Vm, tgt, tau)
        out.update(knn_b=float(-torch.log((1 - lam) * pm[n_a:] + lam * pk[n_a:] + 1e-9).mean()), tau=tau, lam=lam)
        del K, Vm
    return out

def make_opt(m): return torch.optim.AdamW(m.parameters(), lr=LR, betas=(0.9, 0.95), weight_decay=0.01)
def train_step(m, opt, scaler, X, ngram, lr):
    for g in opt.param_groups: g["lr"] = lr
    lg, _ = logits_of(m, X[:, :-1], ngram); loss = F.cross_entropy(lg.reshape(-1, V), X[:, 1:].reshape(-1))
    opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); scaler.step(opt); scaler.update(); return float(loss)

RES = {}
for name in RUNS:
    ngram = name in ("ngram", "combo"); embinit = name in ("embinit", "combo"); knn = name in KNN_RUNS
    torch.manual_seed(0); m = GPT().to(DEV); opt = make_opt(m); scaler = torch.amp.GradScaler(enabled=AMP); rng = np.random.default_rng(0)
    pre = (BIGRAM_SEC if ngram else 0) + (EMB_SEC if embinit else 0)                    # время формул идёт в счёт бюджета
    if embinit: m.wte.weight.data.copy_(EMB)
    tt = pre / 60; step = 0; curve = []; next_ev = 0.0
    while True:
        if tt >= next_ev or tt >= MIN:
            ev = evaluate(m, ngram, knn=knn and (tt >= MIN or len(curve) % 2 == 0)); ev.update(min=round(tt, 2), step=step); curve.append(ev); next_ev += MIN / 10
            if tt >= MIN: break
        sync(); t_s = time.time(); fr = tt / MIN
        train_step(m, opt, scaler, batch(rng), ngram, LR * min(1.0, (step + 1) / 100) * (0.5 * (1 + math.cos(math.pi * min(1.0, fr))) * 0.9 + 0.1))
        sync(); tt += (time.time() - t_s) / 60; step += 1
    fin = curve[-1]; RES[name] = dict(curve=curve, final=fin["loss"], final_b=fin["loss_b"], knn_b=fin.get("knn_b"), steps=step, pre_sec=pre)
    log(f"[{name}] шагов {step}, loss {fin['loss']:.3f}" + (f" | +память (λ={fin['lam']}, τ={fin['tau']}): {fin['knn_b']:.3f} против {fin['loss_b']:.3f} без неё" if "knn_b" in fin else "")
        + " | кривая " + " ".join(f"{c['min']:.1f}м:{c['loss']:.2f}" for c in curve[::2]))
    if name == "base":
        # D1: когда градиент догнал формулы
        t_uni = next((c["min"] for c in curve if c["loss"] <= uni_loss), None); t_bi = next((c["min"] for c in curve if c["loss"] <= bi_loss), None)
        RES["D1"] = dict(uni=uni_loss, bi=bi_loss, t_uni=t_uni, t_bi=t_bi, bigram_sec=BIGRAM_SEC)
        log(f"D1 «переучивает формулу»: униграмму ({uni_loss:.2f}) догнал на {t_uni} мин, биграмму ({bi_loss:.2f}) на {t_bi} мин из {MIN:g} "
            f"({(t_bi or MIN)/MIN:.0%} бюджета); формула считает биграмму за {BIGRAM_SEC:.1f} с")
        # D3: корзины частоты (индекс слова = ранг частоты)
        tok = evaluate(m, False, per_token=True)["tok"]; tg = VX[:, 1:]; d3 = {}
        for lab, lo, hi in (("частые (топ-100)", 2, 102), ("средние (100–1000)", 102, 1002), ("редкие (1000+)", 1002, V)):
            msk = (tg >= lo) & (tg < hi)
            if msk.any(): d3[lab] = dict(share=float(msk.float().mean()), model=float(tok[msk].mean()), bigram=float(bi_tok[msk].mean()))
        RES["D3"] = d3; log("D3 «редкие слова»: " + "; ".join(f"{k} ({v['share']:.0%} слов): модель {v['model']:.2f} / формула {v['bigram']:.2f}" for k, v in d3.items()))
        # D2: запоминание с одного раза + забывание; память случаев на том же куске
        Xp = batch(np.random.default_rng(99))
        def bl(mm):
            with torch.no_grad():
                mm.eval(); lg, _ = logits_of(mm, Xp[:, :-1], False); mm.train(); return float(F.cross_entropy(lg.reshape(-1, V), Xp[:, 1:].reshape(-1)))
        l0 = bl(m); l_mem = None
        if knn:
            K, Vm = build_mem(m, False, extra=Xp)
            with torch.no_grad():
                m.eval(); lg, h = logits_of(m, Xp[:, :-1], False); m.train()
            pm = torch.log_softmax(lg, -1).gather(-1, Xp[:, 1:, None])[..., 0].reshape(-1).exp()
            top, idx = knn_top(F.normalize(h.float(), dim=-1).to(KD).reshape(-1, D), K); pk = knn_p_target(top, idx, Vm, Xp[:, 1:].reshape(-1), fin["tau"])
            l_mem = float(-torch.log((1 - fin["lam"]) * pm + fin["lam"] * pk + 1e-9).mean()); del K, Vm
        train_step(m, opt, scaler, Xp, False, LR * 0.5); l1 = bl(m)
        rng2 = np.random.default_rng(7)
        for _ in range(50): train_step(m, opt, scaler, batch(rng2), False, LR * 0.5)
        l2 = bl(m); RES["D2"] = dict(before=l0, after1=l1, after50=l2, memory=l_mem)
        log(f"D2 «с одного раза»: кусок текста loss {l0:.3f} → после 1 шага градиента {l1:.3f} (−{(l0-l1)/l0:.1%}) → после 50 других шагов {l2:.3f}"
            + (f"; память случаев, положили 1 раз: {l_mem:.3f} (−{(l0-l_mem)/l0:.0%})" if l_mem is not None else ""))
    del m, opt
    if DEV == "cuda": torch.cuda.empty_cache()

# ---------------- во сколько раз быстрее до финального loss base
if "base" in RES:
    tgt = RES["base"]["final"]; tgt_b = RES["base"]["final_b"]; log(f"ИТОГ: цель = финальный loss base {tgt:.3f} за {MIN:g} мин обучения")
    for n in RUNS:
        r = RES[n]; hit = next((c["min"] for c in r["curve"] if c["loss"] <= tgt), None)
        hk = next((c["min"] for c in r["curve"] if c.get("knn_b") is not None and c["knn_b"] <= tgt_b), None)
        sp = lambda h: MIN / max(h, MIN / 100)          # цель достигнута уже на старте (время формул) → «≥×100», а не деление на 0
        r["speedup"] = sp(hit) if hit is not None else None; r["speedup_mem"] = sp(hk) if hk is not None else None
        log(f"  {n}: loss {r['final']:.3f} (Δ {r['final']-tgt:+.3f}); цель на {hit if hit is not None else '—'} мин → " + (f"×{sp(hit):.1f}" if hit is not None else "не достиг")
            + (f"; с памятью {r['knn_b']:.3f}, цель на {hk} мин → ×{sp(hk):.1f}" if hk is not None else (f"; с памятью {r['knn_b']:.3f}" if r.get("knn_b") else "")))
json.dump(dict(min=MIN, d=D, layers=NL, T=T, bs=BS, V=V, n_mem=N_MEM, res=RES), open(os.path.join(OUT, "metrics.json"), "w"), indent=1, default=float)
log("[done]")
