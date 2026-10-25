"""gpt2_train — путь к уровню GPT-2 (124 млн) на НАШЕМ рецепте (25.10.2026, его «как дойти до GPT-2, все возможности, знания и умения»).

Модель с нуля (не чужая сеть): 12 слоёв, d=768, 12 голов, контекст 1024 = размер GPT-2 small (124 млн).
Рецепт из наших замеров: fp16 + SDPA (×3 железо) + Muon для матриц (×2 на слово), AdamW для эмбеддингов.
Словарь — СВОЙ BPE (кусочки слов), обученный на тех же текстах (не словарь GPT-2).
Данные — FineWeb-Edu (открытый набор отобранных образовательных текстов из интернета, sample-10BT).
Линейки GPT-2 (числа известны заранее): HellaSwag (выбрать правильное продолжение из 4; GPT-2 124M = 29,4 %, случайно 25 %),
val loss на FineWeb-Edu (кривая). Сохранение каждые CK_MIN минут в /kaggle/working/ckpt.pt; RESUME=1 + INIT=/kaggle/input → продолжить.
Пилот (MIN=60): цель — честная скорость (слов/с на T4), кривая, оценка «сколько часов до 29,4 %». Локально: SMOKE=1.
"""
import os, io, re, json, math, time, glob, urllib.request
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

E = os.environ.get; SMOKE = E("SMOKE") == "1"
OUT = E("OUT", "/kaggle/working" if os.path.isdir("/kaggle/working") else "/tmp/gpt2_train"); os.makedirs(OUT, exist_ok=True)
MIN = float(E("MIN", 0.3 if SMOKE else 60)); D = int(E("D", 128 if SMOKE else 768)); NL = int(E("NL", 2 if SMOKE else 12)); NH = int(E("NH", 4 if SMOKE else 12))
T = int(E("T", 128 if SMOKE else 1024)); BS = int(E("BS", 4 if SMOKE else 8)); ACC = int(E("ACC", 1 if SMOKE else 8)); VOC = int(E("VOC", 2000 if SMOKE else 32768))
N_TOK = int(E("N_TOK", 2_000_000 if SMOKE else 400_000_000)); CK_MIN = float(E("CK_MIN", 20)); HS_N = int(E("HS_N", 40 if SMOKE else 2000))
DEV = "cuda" if torch.cuda.is_available() else "cpu"; AMP = DEV == "cuda"; T0 = time.time()
def log(*a):
    s = " ".join(str(x) for x in a); print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)
    open(os.path.join(OUT, "log.txt"), "a").write(s + "\n")
def get(url, timeout=900): return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "mini-ai"}), timeout=timeout).read()

# ---------------- данные: FineWeb-Edu sample-10BT, файлы parquet по очереди, пока не наберём N_TOK кусочков
def fineweb_texts():
    tree = json.loads(get("https://huggingface.co/api/datasets/HuggingFaceFW/fineweb-edu/tree/main/sample/10BT", 60))
    files = sorted(x["path"] for x in tree if x["path"].endswith(".parquet"))
    import pyarrow.parquet as pq
    for f in files:
        log("скачиваю", f); loc = os.path.join("/tmp", os.path.basename(f))
        with urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/resolve/main/{f}", headers={"User-Agent": "mini-ai"}), timeout=1800) as r_, open(loc, "wb") as w_:
            while True:
                ch = r_.read(1 << 24)
                if not ch: break
                w_.write(ch)
        tb = pq.read_table(loc, columns=["text"]); os.remove(loc)
        for i in range(0, tb.num_rows, 20000):
            yield tb.slice(i, 20000).column("text").to_pylist()
INIT = E("INIT", "")
def find(name):
    c = glob.glob(os.path.join(INIT, "**", name), recursive=True) if INIT else []; return c[0] if c else None
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
tok_path = find("tokenizer.json") or os.path.join(OUT, "tokenizer.json")
src = fineweb_texts() if not SMOKE else iter([[f"the cat number {i} sat on the mat and looked at the dog {i%7} . " * 20 for i in range(2000)]])
first = next(src)
if os.path.exists(tok_path): tk = Tokenizer.from_file(tok_path); log("словарь из", tok_path)
else:
    tk = Tokenizer(models.BPE()); tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(first, trainers.BpeTrainer(vocab_size=VOC, special_tokens=["<|eot|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tk.save(os.path.join(OUT, "tokenizer.json")); log(f"свой BPE-словарь {tk.get_vocab_size()} кусочков обучен на {len(first)} текстах")
V = tk.get_vocab_size(); EOT = tk.token_to_id("<|eot|>")
tok_file = find("tokens.u16") or os.path.join(OUT, "tokens.u16")
if not os.path.exists(tok_file):
    t0 = time.time(); n = 0; parts = []
    def add(batch):
        global n
        for e in tk.encode_batch(batch): ids = np.array(e.ids + [EOT], np.uint16); parts.append(ids); n += len(ids)
    add(first)
    for b in src:
        if n >= N_TOK: break
        add(b)
    np.concatenate(parts).tofile(tok_file); del parts
    log(f"нарезка: {n/1e6:.0f} млн кусочков за {time.time()-t0:.0f} с")
DATA = np.memmap(tok_file, dtype=np.uint16, mode="r"); NV = min(len(DATA) // 100, 2_000_000)
TRN, VAL = DATA[:-NV], DATA[-NV:]
log(f"данные: train {len(TRN)/1e6:.0f} млн кусочков, val {len(VAL)/1e6:.1f} млн, словарь {V}, {DEV}")

# ---------------- модель (наш GPT: RoPE, pre-norm, связанные веса)
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
        return s.nf(x) @ s.wte.weight.T
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

m = GPT().to(DEV); n_par = sum(p.numel() for p in m.parameters())
mats = [p for n, p in m.named_parameters() if p.ndim == 2 and "wte" not in n]; rest = [p for n, p in m.named_parameters() if not (p.ndim == 2 and "wte" not in n)]
opts = [Muon(mats, lr=0.02), torch.optim.AdamW(rest, lr=3e-3, betas=(0.9, 0.95), weight_decay=0.0)]; base = [[g["lr"] for g in o.param_groups] for o in opts]
scaler = torch.amp.GradScaler(enabled=AMP); state = dict(step=0, tokens=0, minutes=0.0, curve=[])
ck_in = find("ckpt.pt") if E("RESUME") == "1" else None
if ck_in:
    ck = torch.load(ck_in, map_location=DEV); m.load_state_dict(ck["model"]); [o.load_state_dict(s_) for o, s_ in zip(opts, ck["opts"])]; state = ck["state"]
    log(f"[продолжение] с шага {state['step']}, {state['tokens']/1e6:.0f} млн кусочков, {state['minutes']:.0f} мин учёбы")
TOTAL_MIN = float(E("TOTAL_MIN", MIN))    # длина всего расписания (для косинуса) — через несколько сессий
log(f"[модель] {n_par/1e6:.1f} млн параметров, контекст {T}, батч {BS}×{ACC}×{T} = {BS*ACC*T/1e3:.0f}k кусочков на шаг; бюджет сессии {MIN:g} мин, всё расписание {TOTAL_MIN:g} мин")

@torch.no_grad()
def val_loss(n=20 if not SMOKE else 4):
    m.eval(); r = np.random.default_rng(1); tot = 0.0
    for _ in range(n):
        ix = r.integers(0, len(VAL) - T - 1, BS); X = torch.tensor(np.stack([VAL[j:j + T + 1].astype(np.int64) for j in ix]), device=DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=AMP): lg = m(X[:, :-1])
        tot += F.cross_entropy(lg.float().reshape(-1, V), X[:, 1:].reshape(-1)).item()
    m.train(); return tot / n
HS = None
def hellaswag():
    """HellaSwag val: контекст + 4 окончания → выбираем окончание с наименьшей средней ошибкой (как в llm.c); GPT-2 124M = 29,4 %"""
    global HS
    if HS is None:
        try:
            import pyarrow.parquet as pq
            d_ = pq.read_table(io.BytesIO(get("https://huggingface.co/api/datasets/Rowan/hellaswag/parquet/default/validation/0.parquet", 300))).to_pandas()
            HS = [dict(ctx=r["ctx"], endings=list(r["endings"]), label=int(r["label"])) for _, r in d_.head(HS_N).iterrows()]
        except Exception as e: log("HellaSwag не скачался:", e); HS = []
    if not HS: return None
    m.eval(); ok = 0
    with torch.no_grad():
        for ex in HS:
            ctx = tk.encode(ex["ctx"]).ids; rows = []; masks = []
            for end in ex["endings"]:
                e_ = tk.encode(" " + end).ids; rows.append((ctx + e_)[-T:]); masks.append(len(e_))
            L = max(map(len, rows)); X = torch.full((4, L), EOT, device=DEV)
            for i, r_ in enumerate(rows): X[i, :len(r_)] = torch.tensor(r_)
            with torch.autocast(DEV, dtype=torch.float16, enabled=AMP): lg = m(X).float()
            lp = torch.log_softmax(lg[:, :-1], -1).gather(-1, X[:, 1:, None])[..., 0]; sc = []
            for i, r_ in enumerate(rows): n_ = min(masks[i], len(r_) - 1); sc.append(float(-lp[i, len(r_) - 1 - n_:len(r_) - 1].mean()))
            ok += int(np.argmin(sc) == int(ex["label"]))
    m.train(); return ok / len(HS)
def save():
    torch.save(dict(model=m.state_dict(), opts=[o.state_dict() for o in opts], state=state, cfg=dict(D=D, NL=NL, NH=NH, T=T, V=V)), os.path.join(OUT, "ckpt.pt"))
    json.dump(state, open(os.path.join(OUT, "metrics.json"), "w"), indent=1)

rng = np.random.default_rng(state["step"]); t_s = time.time(); last_ck = time.time(); last_ev = -1e9; tok0 = state["tokens"]; min0 = state["minutes"]
if DEV == "cuda": torch.cuda.synchronize()
while True:
    el = (time.time() - t_s) / 60; tot_min = min0 + el; fr = min(1.0, tot_min / TOTAL_MIN)
    if el - last_ev >= (MIN / 6) or el >= MIN:
        vl = val_loss(); hs = hellaswag(); last_ev = el
        state["curve"].append(dict(min=round(tot_min, 1), step=state["step"], tokens=state["tokens"], val=vl, hellaswag=hs))
        tps = (state["tokens"] - tok0) / max(1.0, time.time() - t_s)
        log(f"[кривая] {tot_min:.0f} мин, {state['tokens']/1e6:.0f} млн кусочков: val {vl:.3f}" + (f", HellaSwag {hs:.1%} (GPT-2 124M = 29,4 %, случайно 25 %)" if hs is not None else "") + f" | {tps/1e3:.1f}k кусочков/с")
        if el >= MIN: break
    lr_m = min(1.0, (state["step"] + 1) / 200) * (0.5 * (1 + math.cos(math.pi * fr)) * 0.9 + 0.1)
    for o, bl in zip(opts, base):
        for g, b in zip(o.param_groups, bl): g["lr"] = b * lr_m
    for o in opts: o.zero_grad(set_to_none=True)
    for _ in range(ACC):
        ix = rng.integers(0, len(TRN) - T - 1, BS); X = torch.tensor(np.stack([TRN[j:j + T + 1].astype(np.int64) for j in ix]), device=DEV)
        with torch.autocast(DEV, dtype=torch.float16, enabled=AMP): lg = m(X[:, :-1])
        loss = F.cross_entropy(lg.float().reshape(-1, V), X[:, 1:].reshape(-1)) / ACC; scaler.scale(loss).backward()
    for o in opts: scaler.unscale_(o)
    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    for o in opts: scaler.step(o)
    scaler.update(); state["step"] += 1; state["tokens"] += BS * ACC * T; state["minutes"] = min0 + (time.time() - t_s) / 60
    if state["step"] % 50 == 0: log(f"[шаг] {state['step']} loss {loss.item()*ACC:.3f}")
    if (time.time() - last_ck) / 60 >= CK_MIN: save(); last_ck = time.time()
save()
tps = (state["tokens"] - tok0) / max(1.0, time.time() - t_s)
log(f"ИТОГ: {n_par/1e6:.1f} млн параметров, {tps/1e3:.1f}k кусочков/с на {DEV}; за сессию {(state['tokens']-tok0)/1e6:.0f} млн кусочков; 1 млрд кусочков ≈ {1e9/tps/3600:.1f} ч видеокарты")
log("[done]")
