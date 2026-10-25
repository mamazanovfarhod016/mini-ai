"""fastlearn — БЫСТРОЕ ОБУЧЕНИЕ БЕЗ ЗАБЫВАНИЯ внутри нейросети (25.10.2026, его главная цель: «быстрое ПОЛНОЕ освоение нового, не забывает»).
Гипотеза (из Lin et al., Meta, «Sparse Memory Finetuning», 2025): если новое записывать только в немногие ячейки слоя-памяти, которые
сильно задеты НОВЫМ и мало задеты СТАРЫМ, забывание резко меньше, чем при полном дообучении (там NQ-F1 −89 % у full, −71 % у LoRA, −11 % у sparse).
Проверяем на НАШЕЙ маленькой нейросети (обучена с нуля здесь же), без чужих весов.

Сцена: в базовой выборке — текст задач GSM8K + «известные» сущности с фактами в ДВУХ формах («ke3 lives in kp7 .» / «the home of ke3 is kp7 .»).
Потом выдумываем НОВЫХ (токены, которых база не видела): сессия A (20 фактов), сессия B (20 фактов). Каждый факт показан ОДИН раз (одно предложение,
s шагов по нему). Меры:
  новое_та_же_форма — спросили в той же форме, как учили (запомнила);
  новое_другая_форма — в другой форме («the home of X is» после обучения на «X lives in») — переносится ли смысл, а не буквы;
  A_после_B — помнит ли сессию A после сессии B (забывание НОВОГО);
  известные — факты базы (забывание СТАРОГО); база_loss — на отложенном тексте.
Методы: full (все веса) · replay (full + куски базы) · memall (все значения памяти) · sparse (только top-ячейки по отношению новое/старое).
Успех шага (записан до запуска): sparse даёт новое_та_же_форма близко к full, а известные/база_loss изменяются заметно меньше.
Числа честные: маленькая модель, CPU; перенос формы может быть ≈ шанс — это тоже результат.
Запуск: tools/light.sh tools/fastlearn.py  (MIN=минуты базового обучения)
"""
import os, re, io, sys, json, time, math, copy, random, collections, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
E = os.environ.get; MIN = float(E("MIN", 4)); SEED = int(E("SEED", 0)); OUT = E("OUT", "/tmp/fastlearn"); os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(int(E("THREADS", 2))); torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED); T0 = time.time()
def log(*a):
    s = " ".join(str(x) for x in a); print(f"[{time.time()-T0:5.0f}s] {s}", flush=True); open(os.path.join(OUT, "log.txt"), "a").write(s + "\n")
# ---------- данные
def parquet(url):
    import pyarrow.parquet as pq
    return pq.read_table(io.BytesIO(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "mini-ai"}), timeout=120).read())).to_pandas()
tr = parquet("https://huggingface.co/datasets/openai/gsm8k/resolve/refs%2Fconvert%2Fparquet/main/train/0000.parquet")
def words(t): return re.findall(r"\d+|[a-z]+|[^\sa-z\d]", re.sub(r"<<[^<>]*>>", " ", t.lower().replace(",", "")))
docs = [words(q + " " + a.split("####")[0]) + ["<eos>"] for q, a in zip(tr["question"], tr["answer"])]
cnt = collections.Counter(w for d in docs for w in d)
KE, KP, NE, NP = int(E("KE", 60)), int(E("KP", 20)), 60, 60       # известные сущности/места; новые сущности/места
FORMS = ["{a} lives in {b} .", "the home of {a} is {b} .", "{a} stays in {b} .", "{a} is based in {b} ."]   # 0..2 — парафразы для обучения новому; 3 — НЕ показывается при обучении новому (перенос смысла)
special = ["<pad>", "<unk>", "<bos>"] + [f"ke{i}" for i in range(KE)] + [f"kp{i}" for i in range(KP)] + [f"nw{i}" for i in range(NE)] + [f"np{i}" for i in range(NP)] + ["lives", "in", "the", "home", "of", "is", "stays", "based", "a", "person", "place", "."]
itos = list(dict.fromkeys(special + [w for w, _ in cnt.most_common(2500)])); stoi = {w: i for i, w in enumerate(itos)}; V = len(itos); UNK = 1; BOS = 2; EOS = stoi["<eos>"]
enc = lambda ws: [stoi.get(w, UNK) for w in ws]
rng = random.Random(SEED)
known = [(f"ke{i}", f"kp{rng.randrange(KP)}") for i in range(KE)]
def fact_doc(a, b, form): return FORMS[form].format(a=a, b=b).split()
stream = []
for d in docs:
    stream += enc(d)
    if rng.random() < float(E("KNOWN_P", 0.9)):                 # факт-вставки: известные сущности, 4 формы; чаще, чтобы база ДЕЙСТВИТЕЛЬНО выучила известные факты (в 1-м прогоне 2,9 % = база не знала их)
        for _ in range(rng.randint(2, 4)):
            a, b = rng.choice(known); stream += [BOS] + enc(fact_doc(a, b, rng.randrange(4))) + [EOS]
    if rng.random() < 0.5:        # слова-«пустышки»: имена/места нового тоже встречаются в базе (эмбеддинги обучены), но БЕЗ связи имя→место
        stream += [BOS] + enc([rng.choice([f"nw{rng.randrange(NE)}", f"ke{rng.randrange(KE)}"]), "is", "a", "person", "."]) + [EOS] + [BOS] + enc([rng.choice([f"np{rng.randrange(NP)}", f"kp{rng.randrange(KP)}"]), "is", "a", "place", "."]) + [EOS]
stream = np.array(stream, np.int64); NV = len(stream) // 30; TRN, VAL = stream[:-NV], stream[-NV:]
log(f"словарь {V}, токенов базы {len(TRN)/1e6:.2f} млн, отложено {len(VAL)}")
L = 64
def window(arr, n, r):
    ix = r.integers(0, len(arr) - L - 1, n); return torch.tensor(np.stack([arr[i:i + L + 1] for i in ix]))
BOS_POS = np.flatnonzero(TRN == BOS)
def aligned(arr, n, r):
    """окна, начинающиеся ровно на BOS (начало факта) — формат проверки и обучения новому"""
    ix = BOS_POS[BOS_POS < len(arr) - L - 1]; ix = ix[r.integers(0, len(ix), n)]; return torch.tensor(np.stack([arr[i:i + L + 1] for i in ix]))
# ---------- модель: маленький GPT + слой-память (ключи/значения, top-k доступ)
D, NL_, NH_, KM, DK, TOPK = 128, 4, 4, int(E("KM", 8192)), 64, 8
class Mem(nn.Module):
    def __init__(s):
        super().__init__(); s.q = nn.Linear(D, DK); s.keys = nn.Parameter(torch.randn(KM, DK) * 0.1); s.values = nn.Parameter(torch.randn(KM, D) * 0.02); s.last = None
        s.w1 = nn.Linear(D, D); s.w2 = nn.Linear(D, D)   # входной вентиль, как в Berges et al. 2024: out = (y ⊙ silu(x W1)) W2
    def forward(s, h):
        B, T, _ = h.shape; sc = s.q(h).reshape(B * T, DK) @ s.keys.T; w, ix = sc.topk(TOPK, -1); s.last = ix
        y = (F.softmax(w, -1).unsqueeze(-1) * s.values[ix]).sum(1).reshape(B, T, D)
        return h + s.w2(y * F.silu(s.w1(h)))
class Block(nn.Module):
    def __init__(s):
        super().__init__(); s.n1 = nn.LayerNorm(D); s.n2 = nn.LayerNorm(D); s.qkv = nn.Linear(D, 3 * D); s.o = nn.Linear(D, D); s.f1 = nn.Linear(D, 4 * D); s.f2 = nn.Linear(4 * D, D)
    def forward(s, x):
        B, T, d = x.shape; q, k, v = s.qkv(s.n1(x)).view(B, T, 3, NH_, d // NH_).permute(2, 0, 3, 1, 4)
        x = x + s.o(F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, d)); return x + s.f2(F.gelu(s.f1(s.n2(x))))
class GPT(nn.Module):
    def __init__(s):
        super().__init__(); s.wte = nn.Embedding(V, D); s.wpe = nn.Embedding(L + 8, D); s.blocks = nn.ModuleList(Block() for _ in range(NL_)); s.mem = Mem(); s.nf = nn.LayerNorm(D)
        nn.init.normal_(s.wte.weight, std=0.02); nn.init.normal_(s.wpe.weight, std=0.02)
    def forward(s, idx):
        x = s.wte(idx) + s.wpe(torch.arange(idx.shape[1]))
        for i, b in enumerate(s.blocks):
            x = b(x)
            if i == 1: x = s.mem(x)
        return s.nf(x) @ s.wte.weight.T
m = GPT(); log(f"параметров {sum(p.numel() for p in m.parameters())/1e6:.2f} млн (память {KM}×{D}, top-{TOPK})")
# ---------- базовое обучение (MIN минут)
opt = torch.optim.AdamW(m.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.01); r_ = np.random.default_rng(1); t_s = time.time(); st = 0
while True:
    fr = (time.time() - t_s) / (MIN * 60)
    if fr >= 1: break
    for g in opt.param_groups: g["lr"] = 2e-3 * min(1, (st + 1) / 100) * (0.5 * (1 + math.cos(math.pi * fr)) * 0.9 + 0.1)
    X = torch.cat([window(TRN, 24, r_), aligned(TRN, 8, r_)]); loss = F.cross_entropy(m(X[:, :-1]).reshape(-1, V), X[:, 1:].reshape(-1)); opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step(); st += 1
    if st % 100 == 0: log(f"[база] шаг {st} loss {loss.item():.3f}")
@torch.no_grad()
def base_loss(mm, n=40):
    r = np.random.default_rng(5); X = window(VAL, n, r); return F.cross_entropy(mm(X[:, :-1]).reshape(-1, V), X[:, 1:].reshape(-1)).item()
@torch.no_grad()
def recall(mm, facts, form, cands):
    """спросили в форме form («a lives in» / «the home of a is») → argmax среди кандидатов-мест == верное место"""
    pre = [[BOS] + enc(FORMS[form].split(" {b}")[0].format(a=a).split()) for a, b in facts]
    X = torch.tensor(pre); lg = mm(X)[:, -1]; ci = torch.tensor([stoi[c] for c in cands]); pick = ci[lg[:, ci].argmax(-1)]
    return float(np.mean([int(pick[i]) == stoi[b] for i, (a, b) in enumerate(facts)]))
KP_C = [f"kp{i}" for i in range(KP)]; NP_C = [f"np{i}" for i in range(NP)]
def known_recall(mm): return np.mean([recall(mm, known, f, KP_C) for f in (0, 1, 2, 3)])
BASE_OK = None
log(f"[база готова] шагов {st}; loss отложенный {base_loss(m):.3f}; известные факты (обе формы) {known_recall(m):.1%} (шанс {1/KP:.1%})")
BASE = copy.deepcopy(m.state_dict()); base_l0 = base_loss(m); known0 = known_recall(m)
BASE_OK = float(known0) >= 0.7
log("ВОРОТА: база выучила известные факты" + (" — прогон ДЕЙСТВИТЕЛЕН" if BASE_OK else f" только на {float(known0):.0%} (<70 %) — результаты НЕ ДЕЙСТВИТЕЛЬНЫ, лечить базу"))
def new_facts(off, n=20):
    r = random.Random(100 + off); return [(f"nw{off+i}", f"np{r.randrange(NP)}") for i in range(n)]
FA, FB = new_facts(0), new_facts(30)
def fact_batch(facts, forms=(0,)):
    rows = [[BOS] + enc(fact_doc(a, b, f)) for a, b in facts for f in forms]; Lm = max(map(len, rows))
    return torch.tensor([r + [0] * (Lm - len(r)) for r in rows])      # дополняем <pad>=0; ниже в loss маскируем
def learn(mm, facts, method, steps, lr, slots=None):
    """steps шагов Adam по предложениям фактов (форма 0), один и тот же набор — «показали один раз» = steps малых шагов"""
    X = fact_batch(facts, (0, 1, 2)); r = np.random.default_rng(9)
    if method in ("memall", "sparse"):
        params = [mm.mem.values]
        for p in mm.parameters(): p.requires_grad_(p is mm.mem.values)
    else:
        params = list(mm.parameters())
        for p in params: p.requires_grad_(True)
    op = torch.optim.SGD(params, lr=lr) if method == "sparse" else torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        Xi = X if method != "replay" else torch.cat([X, window(TRN, 16, r)[:, :X.shape[1]]], 0)
        loss = F.cross_entropy(mm(Xi[:, :-1]).reshape(-1, V), Xi[:, 1:].reshape(-1), ignore_index=0); op.zero_grad(); loss.backward()
        if method == "sparse": mm.mem.values.grad *= slots[:, None]
        op.step()
    for p in mm.parameters(): p.requires_grad_(True)
@torch.no_grad()
def pick_slots(mm, facts, n_slots):
    """ячейки памяти, сильно задетые НОВЫМ и мало СТАРЫМ (как в Sparse Memory Finetuning)"""
    mm(fact_batch(facts, (0, 1, 2))[:, :-1]); c_new = torch.bincount(mm.mem.last.reshape(-1), minlength=KM).float()
    mm(window(TRN, 64, np.random.default_rng(3))[:, :-1]); c_old = torch.bincount(mm.mem.last.reshape(-1), minlength=KM).float()
    score = (c_new / c_new.sum()) / (c_old / c_old.sum() + 1e-4); mask = torch.zeros(KM); mask[score.topk(n_slots).indices] = 1; return mask
rows = []
GRID = {"full": [3e-4, 1e-3], "replay": [3e-4, 1e-3], "memall": [1e-2, 3e-2], "sparse": [1.0, 3.0, 10.0]}
for method in ["full", "replay", "memall", "sparse"]:
    for lr in GRID[method]:
        for steps in [1, 5, 20]:
            mm = GPT(); mm.load_state_dict(BASE); sl = pick_slots(mm, FA, 100) if method == "sparse" else None
            learn(mm, FA, method, steps, lr, sl); a_same = recall(mm, FA, 0, NP_C); a_other = recall(mm, FA, 3, NP_C)
            sl2 = pick_slots(mm, FB, 100) if method == "sparse" else None
            learn(mm, FB, method, steps, lr, sl2)
            row = dict(method=method, lr=lr, steps=steps, new_same=a_same, new_other=a_other, A_after_B=recall(mm, FA, 0, NP_C), B_same=recall(mm, FB, 0, NP_C),
                       known=float(known_recall(mm)), base_dloss=base_loss(mm) - base_l0, known_d=float(known_recall(mm)) - float(known0))
            rows.append(row); log(f"{method:7s} lr={lr:<6g} s={steps:<3d} новое(та же форма) {row['new_same']:.0%} (новая форма) {row['new_other']:.0%} | A после B {row['A_after_B']:.0%} | известные {row['known']:.0%} (Δ {row['known_d']:+.0%}) | база Δloss {row['base_dloss']:+.3f}")
json.dump(dict(base_steps=st, base_loss=base_l0, known0=float(known0), chance_new=1 / NP, chance_known=1 / KP, rows=rows), open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
best = {}
for r in rows:
    if r["new_same"] >= 0.8 and (r["method"] not in best or r["base_dloss"] + abs(r["known_d"]) < best[r["method"]]["base_dloss"] + abs(best[r["method"]]["known_d"])): best[r["method"]] = r
log("ИТОГ (лучшая строка каждого метода среди тех, где новое_та_же_форма ≥ 80 %; «цена» = Δloss базы + |Δ известных|):")
for k, r in best.items(): log(f"  {k}: lr={r['lr']} s={r['steps']} новое {r['new_same']:.0%}/{r['new_other']:.0%} A после B {r['A_after_B']:.0%} известные Δ{r['known_d']:+.0%} база Δ{r['base_dloss']:+.3f}")
if not best: log("  ни один метод не дошёл до 80 % на новом в этой сетке — смотри строки выше")
