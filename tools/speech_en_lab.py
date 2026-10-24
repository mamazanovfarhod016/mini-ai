"""speech_en_lab — ДВА МЕХАНИЗМА РЕЧИ БЕЗ ГРАДИЕНТА против плоского резервуара (24.10.2026, его «попробуй оба варианта через kaggle»).

Факт, от которого отталкиваемся (mini-ai-speech-en-quick, 9.9M токенов): плоский резервуар top-1 0.180 < триграмма 0.281,
кривая по объёму ПЛОСКАЯ (0.178 → 0.180 при ×10 данных), ответы = случайные. Это потолок механизма, не данных.

MODE=deep  — ГЛУБОКИЙ резервуар, каждый слой решается формулой (без градиента):
   слой k читает [код слова, ПРЕДСКАЗАНИЕ слоя k-1 о смысле следующего слова, коды 3 предыдущих слов];
   у слоя две «головы», обе = ridge-решение: (а) следующее слово, (б) код следующего слова (d чисел) — она и есть
   выученное ПРЕДСТАВЛЕНИЕ, которое идёт на вход следующему слою. Итог = решение по признакам всех слоёв сразу.
   Числа: top-1 по слоям (растёт ли с глубиной?), итог vs триграмма, ответы vs случайный, кривая 25/50/100 %.
MODE=cases — РЕЧЬ ИЗ СЛУЧАЕВ + СМЫСЛ БЕЗ ГРАДИЕНТА (как analogy.py/exp65, только по-английски и на корпусе):
   смысл слов = PPMI-матрица соседств → SVD (замкнутая формула, не обучение); ответ = ближайший по смыслу контекста
   случай из памяти + перенос имён/чисел по позиции; цепочка GSM8K = ближайшая решённая задача → её арифметика
   пересчитывается на новых числах (перенос случая по ФОРМЕ, как в analogy.py); истории = сцепка предложений по смыслу.
   ЧЕСТНО: ответ/истории здесь — слой поиска (память), не порождение; порождение — только пересчёт цепочки.
MODE=grad  — КОНТРОЛЬ С ГРАДИЕНТОМ (его «попробую также с градиентами», 24.10): наш трансформер (mini_ai/transformer.py: своя сеть,
   свой AdamW, RoPE; PyTorch считает только backprop — mini_ai/torch_engine.py) на ТЕХ ЖЕ данных и ТЕХ ЖЕ линейках. Бюджет
   времени GRAD_MIN минут на T4. Это точка отсчёта «что даёт GPT-2-мини на этих данных», чтобы честно мерить наши механизмы.
Успех записан ДО запуска: deep — top-1 ≥ триграммы И рост по слоям; cases — ответы ≥ ×1.5 случайного, GSM8K > 0 верных.
Локально: SMOKE=1 MODE=deep|cases python3 tools/speech_en_lab.py
"""
import os, re, io, sys, json, time, math, collections, urllib.request
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SMOKE = os.environ.get("SMOKE") == "1"; MODE = os.environ.get("MODE", "deep")
OUT = os.environ.get("OUT", "/kaggle/working" if os.path.isdir("/kaggle/working") else f"/tmp/speech_lab_{MODE}")
os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(*a):
    s = " ".join(str(x) for x in a); print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)
    with open(os.path.join(OUT, "log.txt"), "a") as f: f.write(s + "\n")
try:
    import cupy as cp; cp.zeros(1).sum(); xp = cp; GPU = True; import cupyx
except Exception:
    cp = None; xp = np; GPU = False
asnp = (lambda a: cp.asnumpy(a)) if GPU else (lambda a: np.asarray(a))
log("MODE", MODE, "GPU" if GPU else "CPU", "SMOKE" if SMOKE else "FULL")
E = os.environ.get
N_RES = int(E("N", 300 if SMOKE else 3000)); D_EMB = int(E("D", 48 if SMOKE else 160)); LAYERS = int(E("LAYERS", 3))
VOCAB = int(E("VOCAB", 2000 if SMOKE else 12000)); LAM = float(E("LAM", 2.0)); MAXLEN = 48 if SMOKE else 128; BS = 64 if SMOKE else int(E("BS_GRAD", 256))
STORY_SHARDS = int(E("STORY_SHARDS", 0 if SMOKE else 1)); MAX_STORIES = int(E("MAX_STORIES", 200 if SMOKE else 40000))
MAX_DIAL = int(E("MAX_DIAL", 3000 if SMOKE else 60000)); TRI_LIMIT = int(E("TRI_LIMIT", 3_000_000))

# ---------------------------------------------------------------- данные (как в speech_en_reservoir)
HF = "https://huggingface.co/datasets/{repo}/resolve/refs%2Fconvert%2Fparquet/{cfg}/{split}/{i:04d}.parquet"
def fetch(url, tries=3):
    for k in range(tries):
        try: return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "mini-ai"}), timeout=120).read()
        except Exception as e: log("  retry", k, url[-50:], type(e).__name__); time.sleep(5)
def parquet(url):
    b = fetch(url)
    if b is None: return None
    import pyarrow.parquet as pq; return pq.read_table(io.BytesIO(b)).to_pandas()
DIAL, TEXT, STORIES, DID = [], [], [], []
_did = [0]
def add_dialog(turns, src):
    turns = [t.strip() for t in turns if t and t.strip()]; _did[0] += 1
    for i in range(1, len(turns)): DIAL.append((" ".join(turns[max(0, i - 2):i]), turns[i], src)); DID.append(_did[0])
t = parquet(HF.format(repo="li2017dailydialog/daily_dialog", cfg="default", split="train", i=0))
if t is not None:
    for d in t["dialog"]: add_dialog(list(d), "dailydialog")
t = parquet(HF.format(repo="facebook/empathetic_dialogues", cfg="default", split="train", i=0))
if t is not None:
    for cid, g in t.sort_values(["conv_id", "utterance_idx"]).groupby("conv_id"): add_dialog([u.replace("_comma_", ",") for u in g["utterance"].tolist()], "empathetic")
t = parquet(HF.format(repo="bavard/personachat_truecased", cfg="full", split="train", i=0))
if t is not None:
    seen = set()
    for h in t["history"]:
        h = list(h); k = tuple(h[:3])
        if k not in seen: seen.add(k); add_dialog(h, "personachat")
rng = np.random.default_rng(0); _perm = rng.permutation(len(DIAL)); DIAL = [DIAL[i] for i in _perm][:MAX_DIAL]; DID = [DID[i] for i in _perm][:MAX_DIAL]; log("диалог-пар", len(DIAL), "диалогов", len(set(DID)))
GSM, GSM_TEST = [], []
t = parquet(HF.format(repo="openai/gsm8k", cfg="main", split="train", i=0))
if t is not None:
    for q, a in zip(t["question"], t["answer"]):
        GSM.append((q, a)); TEXT.append((q + " " + re.sub(r"<<[^>]*>>", "", a).replace("####", "the answer is"), "gsm8k"))
    tt = parquet(HF.format(repo="openai/gsm8k", cfg="main", split="test", i=0))
    if tt is not None: GSM_TEST = [(q, a.split("####")[-1].strip()) for q, a in zip(tt["question"][:200], tt["answer"][:200])]
for i in range(STORY_SHARDS):
    t = parquet(HF.format(repo="roneneldan/TinyStories", cfg="default", split="train", i=i))
    if t is None: break
    for s in t["text" if "text" in t.columns else t.columns[0]]:
        STORIES.append(s); TEXT.append((s, "tinystories"))
        if len(STORIES) >= MAX_STORIES: break
    if len(STORIES) >= MAX_STORIES: break
if SMOKE and not STORIES: STORIES = ["Tom had three apples. He ate one, so he had two apples left. Then he gave one to Ann. Ann was happy."] * 30; TEXT += [(s, "smoke") for s in STORIES]
log("тексты", len(TEXT), "историй", len(STORIES), "gsm8k", len(GSM))
def toks(s): return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+(?:\.\d+)?|[.,!?;:]", s.lower())
# разбивка ПО ДИАЛОГАМ (утечка 24.10: при разбивке по парам соседняя реплика того же разговора лежала в памяти → 44/300 «ответов» были куском контекста)
SPLIT_BY_DIALOG = E("SPLIT", "dialog") == "dialog"
if SPLIT_BY_DIALOG:
    TEST_D = [d for d, i in zip(DIAL, DID) if i % 40 == 0][:400]; TRAIN_D = [d for d, i in zip(DIAL, DID) if i % 40 != 0]
else:
    n_test = max(200, len(DIAL) // 50); TEST_D, TRAIN_D = DIAL[:n_test], DIAL[n_test:]
log("test пар", len(TEST_D), "train пар", len(TRAIN_D), "разбивка", "по диалогам" if SPLIT_BY_DIALOG else "по парам")
cnt = collections.Counter()
for c, r, _ in TRAIN_D: cnt.update(toks(c)); cnt.update(toks(r))
for s, _ in TEXT: cnt.update(toks(s))
itos = ["<pad>", "<unk>", "<eos>", "<bos>", "<sep>"] + [w for w, _ in cnt.most_common(VOCAB)]; stoi = {w: i for i, w in enumerate(itos)}; V = len(itos)
PAD, UNK, EOS, BOS, SEP = 0, 1, 2, 3, 4; enc = lambda ws: [stoi.get(w, UNK) for w in ws]
STOP = set("the a an to of and i you it is in that was he she my me we they for on with be have do are this at but not so as his her him there what how about your just can will would like really".split())
def content(s): return [w for w in toks(s) if w not in STOP and len(w) > 2]
def judge_pairs(outs, seed=0):
    """целые ответы: доля содержательных слов ответа, совпавших с эталоном, vs со случайным эталоном; echo = ответ лежит внутри контекста"""
    r = np.random.default_rng(seed); g = rr = n = 0; echo = 0
    for c, gold, a in outs:
        A = set(content(a)); G = set(content(gold)); R = set(content(TEST_D[int(r.integers(len(TEST_D)))][1]))
        if A: g += len(A & G) / len(A); rr += len(A & R) / len(A); n += 1
        echo += (a.strip() != "" and a.strip() in c)
    judge_pairs.echo = echo
    return g / max(1, n), rr / max(1, n)
def save(metrics, outs, chain, stories):
    with open(os.path.join(OUT, "samples.txt"), "w") as f:
        f.write("=== ОТВЕТЫ (без отбора) ===\n"); [f.write(f"CTX: {c}\nGOLD: {g}\nOURS: {a}\n\n") for c, g, a in outs]
        f.write("=== ЦЕПОЧКА GSM8K ===\n"); [f.write(f"Q: {q}\nGOLD: {g}\nOURS: {a}\n\n") for q, g, a in chain]
        f.write("=== ИСТОРИИ ===\n"); [f.write(f"{p} → {s}\n\n") for p, s in stories]
    metrics["seconds"] = time.time() - T0; json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=1, default=float)
    log(f"[done] {time.time()-T0:.0f}s → {OUT}")
STORY_PROMPTS = ["Once upon a time, there was a little girl named Lily.", "Tom had three apples.", "The dog was sad because it was raining.", "Lily wanted to go to the park, but her mom said no.", "One day, a small robot woke up in a garden."]

# ================================================================ MODE=deep
if MODE == "deep":
    def seq_dialog(c, r):
        x = [BOS] + enc(toks(c))[-(MAXLEN // 2):] + [SEP] + enc(toks(r))[: MAXLEN // 2] + [EOS]; k = x.index(SEP)
        return x, [0] * (k + 1) + [1] * (len(x) - k - 2)
    def seq_text(s):
        ids = enc(toks(s)); out = []
        for i in range(0, len(ids), MAXLEN - 2):
            x = [BOS] + ids[i:i + MAXLEN - 2] + [EOS]; out.append((x, [1] * (len(x) - 1)))
        return out
    TRAIN = [seq_dialog(c, r) for c, r, _ in TRAIN_D] + [p for s, _ in TEXT for p in seq_text(s)]; TEST = [seq_dialog(c, r) for c, r, _ in TEST_D]
    rng.shuffle(TRAIN); NTOK = sum(len(x) for x, _ in TRAIN); log("train seq", len(TRAIN), "токенов", NTOK, "test", len(TEST))
    def batches(seqs, bs):
        seqs = sorted(seqs, key=len)
        for i in range(0, len(seqs), bs):
            ch = seqs[i:i + bs]; T = max(len(x) for x, _ in ch); X = np.full((len(ch), T), EOS, np.int64); M = np.zeros((len(ch), T - 1), bool)
            for j, (x, m) in enumerate(ch): X[j, :len(x)] = x; M[j, :len(m)] = m
            yield X, M
    def ngram_eval(limit):
        uni, bi, tri = collections.Counter(), collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter); n_ = 0
        for x, m in TRAIN:
            for i in range(len(x) - 1):
                uni[x[i + 1]] += 1; bi[x[i]][x[i + 1]] += 1
                if i >= 1: tri[(x[i - 1], x[i])][x[i + 1]] += 1
            n_ += len(x)
            if n_ >= limit: break
        ok = n = 0
        for x, m in TEST:
            for i in range(len(x) - 1):
                if not m[i]: continue
                c = tri.get((x[i - 1], x[i])) if i >= 1 else None; p = (c or bi.get(x[i]) or uni).most_common(1)[0][0]; ok += (p == x[i + 1]); n += 1
        return ok / n
    tri_acc = ngram_eval(TRI_LIMIT); log(f"[контроль] триграмма top-1: {tri_acc:.3f}")
    HIST = 3
    class Layer:
        """резервуар слоя k: вход u = [E[x_t], pred_{k-1}(t)] ; признаки φ = [h, u, E[x_{t-1..t-3}], 1]; две головы — слово и код следующего слова"""
        def __init__(self, k, d_in, seed):
            r = np.random.default_rng(seed); N = N_RES
            self.Win = xp.asarray((r.normal(0, 1, (d_in, N)) * 0.4).astype(np.float32))
            W = r.normal(0, 1, (N, N)).astype(np.float32); W *= (r.random((N, N)) < 10 / N); v = r.normal(size=N).astype(np.float32)
            for _ in range(60): v = W @ v; v /= (np.linalg.norm(v) + 1e-9)
            W *= 0.9 / (np.linalg.norm(W @ v) + 1e-9); self.Wres = xp.asarray(W)
            self.k, self.N, self.F = k, N, N + d_in + HIST * D_EMB + 1
            self.A = xp.zeros((self.F, self.F), xp.float64); self.B = xp.zeros((V, self.F), xp.float64); self.Wout = None; self.Wemb = None
        def feats(self, Xg, prev_pred):
            """Xg (B,T) → φ (B,T,F); prev_pred (B,T,d) — предсказание смысла следующего слова предыдущим слоем (или нули)"""
            Bt, T = Xg.shape; h = xp.zeros((Bt, self.N), xp.float32); Phi = xp.zeros((Bt, T, self.F), xp.float32)
            for t in range(T):
                u = xp.concatenate([EMB[Xg[:, t]], prev_pred[:, t]], axis=1); h = xp.tanh(u @ self.Win + h @ self.Wres)
                hist = [EMB[Xg[:, t - j]] if t - j >= 0 else xp.zeros((Bt, D_EMB), xp.float32) for j in range(1, HIST + 1)]
                Phi[:, t] = xp.concatenate([h, u] + hist + [xp.ones((Bt, 1), xp.float32)], axis=1)
            return Phi
        def accumulate(self, Phi, Xg, M):
            m = xp.asarray(M); P = Phi[:, :-1][m]; y = Xg[:, 1:][m]
            self.A += (P.T @ P).astype(xp.float64)
            B32 = xp.zeros((V, self.F), xp.float32)
            if GPU: cupyx.scatter_add(B32, y, P)
            else: np.add.at(B32, y, P)
            self.B += B32.astype(xp.float64)
        def solve(self):
            A = self.A + LAM * xp.eye(self.F, dtype=xp.float64); self.Wout = xp.linalg.solve(A, self.B.T).astype(xp.float32)          # (F,V)
            self.Wemb = xp.linalg.solve(A, (EMB.astype(xp.float64).T @ self.B).T).astype(xp.float32)                               # (F,d): φ → код следующего слова
    EMB = xp.asarray((np.random.default_rng(0).choice([-1.0, 1.0], size=(V, D_EMB)) / np.sqrt(D_EMB)).astype(np.float32))
    layers = []; CURVE = []; per_layer = []
    def run_layers(Xg, upto):
        """признаки всех слоёв 0..upto для батча (слои < upto уже решены)"""
        Bt, T = Xg.shape; pred = xp.zeros((Bt, T, D_EMB), xp.float32); phis = []
        for L in layers[:upto + 1]:
            Phi = L.feats(Xg, pred); phis.append(Phi)
            if L.Wemb is not None: pred = Phi @ L.Wemb
        return phis
    all_b = list(batches(TRAIN, BS)); rng.shuffle(all_b)
    for k in range(LAYERS):
        L = Layer(k, D_EMB + (D_EMB if k > 0 else 0) if False else 2 * D_EMB, seed=k); layers.append(L)
        for X, M in all_b:
            Xg = xp.asarray(X); Phi = run_layers(Xg, k)[-1]; L.accumulate(Phi, Xg, M)
        L.solve()
        ok = n = 0
        for X, M in batches(TEST, BS):
            Xg = xp.asarray(X); Phi = run_layers(Xg, k)[-1]; m = xp.asarray(M); lg = Phi[:, :-1][m] @ L.Wout; ok += int((lg.argmax(1) == Xg[:, 1:][m]).sum()); n += int(m.sum())
        per_layer.append(ok / n); log(f"[слой {k+1}] F={L.F} top-1 {ok/n:.3f}   (триграмма {tri_acc:.3f})")
    # итоговая голова: признаки ВСЕХ слоёв вместе; кривая по объёму 25/50/100 %
    FT = sum(L.F for L in layers); A = xp.zeros((FT, FT), xp.float64); B = xp.zeros((V, FT), xp.float64); ntok = 0; nb = len(all_b); marks = {int(nb * 0.25): 0.25, int(nb * 0.5): 0.5, nb: 1.0}
    def final_feats(Xg): return xp.concatenate(run_layers(Xg, LAYERS - 1), axis=2)
    def eval_final(Wf):
        ok = ok5 = n = 0
        for X, M in batches(TEST, BS):
            Xg = xp.asarray(X); m = xp.asarray(M); lg = final_feats(Xg)[:, :-1][m] @ Wf; y = Xg[:, 1:][m]
            ok += int((lg.argmax(1) == y).sum()); ok5 += int((xp.argsort(-lg, axis=1)[:, :5] == y[:, None]).any(1).sum()); n += int(m.sum())
        return ok / n, ok5 / n
    def generate(Wf, prefix, max_new=40, temp=0.8, seed=1):
        r = np.random.default_rng(seed); ids = list(prefix)
        for _ in range(max_new):
            Xg = xp.asarray([ids[-MAXLEN:]]); out = asnp(final_feats(Xg)[0, -1] @ Wf); out[[PAD, BOS, SEP]] = 0; out[UNK] = min(out[UNK], 1e-4)
            p = np.clip(out, 1e-4, None) ** (1 / temp); p[[PAD, BOS, SEP]] = 0; p /= p.sum(); nxt = int(r.choice(V, p=p)); ids.append(nxt)
            if nxt == EOS: break
        return " ".join(itos[i] for i in ids[len(prefix):] if i != EOS)
    Wf = None
    for bi, (X, M) in enumerate(all_b, 1):
        Xg = xp.asarray(X); Phi = final_feats(Xg); m = xp.asarray(M); P = Phi[:, :-1][m]; y = Xg[:, 1:][m]
        A += (P.T @ P).astype(xp.float64); B32 = xp.zeros((V, FT), xp.float32)
        if GPU: cupyx.scatter_add(B32, y, P)
        else: np.add.at(B32, y, P)
        B += B32.astype(xp.float64); ntok += int(m.sum())
        if bi in marks:
            Wf = xp.linalg.solve(A + LAM * xp.eye(FT, dtype=xp.float64), B.T).astype(xp.float32); a1, a5 = eval_final(Wf)
            outs = [(c, g, generate(Wf, [BOS] + enc(toks(c))[-(MAXLEN // 2):] + [SEP], seed=i)) for i, (c, g, _) in enumerate(TEST_D[:100])]; gj, rj = judge_pairs(outs)
            CURVE.append(dict(frac=marks[bi], tokens=ntok, top1=a1, top5=a5, reply_gold=gj, reply_rand=rj)); log(f"[кривая] {int(marks[bi]*100):3d}% {ntok:,} ток: top-1 {a1:.3f} top-5 {a5:.3f} | ответ↔эталон {gj:.3f} vs случайный {rj:.3f}")
    a1, a5 = eval_final(Wf); outs = [(c, g, generate(Wf, [BOS] + enc(toks(c))[-(MAXLEN // 2):] + [SEP], seed=i)) for i, (c, g, _) in enumerate(TEST_D[:300])]; gj, rj = judge_pairs(outs)
    chain = []; right = steps = 0
    for q, gold in GSM_TEST[:50]:
        a = generate(Wf, [BOS] + enc(toks(q))[-100:], max_new=80, temp=0.6, seed=3); nums = re.findall(r"\d+", a); steps += bool(re.search(r"= \d+", a)); right += (gold in nums[-2:]) if nums else 0; chain.append((q, gold, a))
    stories = [(p, generate(Wf, [BOS] + enc(toks(p)), max_new=60, seed=s)) for s, p in enumerate(STORY_PROMPTS)]
    for p, s in stories: log(f"[история] {p} → {s[:150]}")
    log(f"ИТОГ deep: слои {['%.3f' % x for x in per_layer]} → все слои {a1:.3f} (триграмма {tri_acc:.3f}); top-5 {a5:.3f}; ответы {gj:.3f} vs случ {rj:.3f} (×{gj/max(rj,1e-9):.1f}); GSM8K верных {right}/50, структура {steps}")
    save(dict(mode="deep", N=N_RES, d=D_EMB, layers=LAYERS, V=V, tokens=NTOK, trigram=tri_acc, per_layer=per_layer, top1=a1, top5=a5, reply_gold=gj, reply_rand=rj, curve=CURVE, gsm_right=right, gsm_steps=steps), outs, chain, stories)

# ================================================================ MODE=grad
elif MODE == "grad":
    import torch
    from mini_ai.transformer import init_params, n_params
    from mini_ai.torch_engine import TorchEngine
    GRAD_MIN = float(E("GRAD_MIN", 0.5 if SMOKE else 100)); D_T = int(E("D_T", 64 if SMOKE else 256)); NL = int(E("NL", 2 if SMOKE else 6)); NH = int(E("NH", 4 if SMOKE else 8))
    def seq_dialog(c, r):
        x = [BOS] + enc(toks(c))[-(MAXLEN // 2):] + [SEP] + enc(toks(r))[: MAXLEN // 2] + [EOS]; k = x.index(SEP)
        return x, [0] * (k + 1) + [1] * (len(x) - k - 2)
    def seq_text(s):
        ids = enc(toks(s)); out = []
        for i in range(0, len(ids), MAXLEN - 2):
            x = [BOS] + ids[i:i + MAXLEN - 2] + [EOS]; out.append((x, [1] * (len(x) - 1)))
        return out
    TRAIN = [seq_dialog(c, r) for c, r, _ in TRAIN_D] + [p for s, _ in TEXT for p in seq_text(s)]; TEST = [seq_dialog(c, r) for c, r, _ in TEST_D]
    rng.shuffle(TRAIN); NTOK = sum(len(x) for x, _ in TRAIN); log("train seq", len(TRAIN), "токенов", NTOK, "test", len(TEST))
    def batches(seqs, bs):
        seqs = sorted(seqs, key=len)
        for i in range(0, len(seqs), bs):
            ch = seqs[i:i + bs]; T = max(len(x) for x, _ in ch); X = np.full((len(ch), T), EOS, np.int64); M = np.zeros((len(ch), T - 1), bool)
            for j, (x, m) in enumerate(ch): X[j, :len(x)] = x; M[j, :len(m)] = m
            yield X, M
    def ngram_eval(limit):
        uni, bi, tri = collections.Counter(), collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter); n_ = 0
        for x, m in TRAIN:
            for i in range(len(x) - 1):
                uni[x[i + 1]] += 1; bi[x[i]][x[i + 1]] += 1
                if i >= 1: tri[(x[i - 1], x[i])][x[i + 1]] += 1
            n_ += len(x)
            if n_ >= limit: break
        ok = n = 0
        for x, m in TEST:
            for i in range(len(x) - 1):
                if not m[i]: continue
                c = tri.get((x[i - 1], x[i])) if i >= 1 else None; p = (c or bi.get(x[i]) or uni).most_common(1)[0][0]; ok += (p == x[i + 1]); n += 1
        return ok / n
    tri_acc = ngram_eval(TRI_LIMIT); log(f"[контроль] триграмма top-1: {tri_acc:.3f}")
    p0, cfg = init_params(V, d=D_T, n_layer=NL, n_head=NH, pos="rope", seed=0); eng = TorchEngine(p0, cfg, lr=6e-4, wd=0.01, seed=0)
    log(f"[grad] трансформер d={D_T} слоёв {NL} голов {NH} параметров {n_params(p0):,} на {eng.dev}; бюджет {GRAD_MIN:g} мин")
    all_b = list(batches(TRAIN, BS)); rng.shuffle(all_b)
    @torch.no_grad()
    def eval_top1(k=None):
        ok = ok5 = n = 0
        for X, M in list(batches(TEST, BS))[:k]:
            lg = __import__("mini_ai.torch_engine", fromlist=["forward"]).forward(eng.p, cfg, eng._t(X, torch.long))[:, :-1]; m = eng._t(M, torch.bool); y = eng._t(X, torch.long)[:, 1:][m]; lg = lg[m]
            ok += int((lg.argmax(1) == y).sum()); ok5 += int((lg.topk(5, dim=1).indices == y[:, None]).any(1).sum()); n += int(m.sum())
        return ok / n, ok5 / n
    @torch.no_grad()
    def generate(prefix, max_new=40, temp=0.8, seed=1):
        r = np.random.default_rng(seed); ids = list(prefix); fwd = __import__("mini_ai.torch_engine", fromlist=["forward"]).forward
        for _ in range(max_new):
            lg = fwd(eng.p, cfg, eng._t([ids[-MAXLEN:]], torch.long))[0, -1].float().cpu().numpy(); lg[[PAD, BOS, SEP]] = -1e9; lg[UNK] -= 5
            z = (lg - lg.max()) / temp; pr = np.exp(z); pr /= pr.sum(); nxt = int(r.choice(V, p=pr)); ids.append(nxt)
            if nxt == EOS: break
        return " ".join(itos[i] for i in ids[len(prefix):] if i != EOS)
    t_start = time.time(); step = 0; seen = 0; CURVE = []; next_mark = 0.1; warm = 300; base_lr = 6e-4
    while time.time() - t_start < GRAD_MIN * 60:
        for X, M in all_b:
            lr = base_lr * min(1.0, (step + 1) / warm)
            loss, nrm = eng.step(X[:, :-1], X[:, 1:], M, lr, drop=0.05); step += 1; seen += int(M.sum())
            if step % 200 == 0: log(f"[grad] шаг {step} токенов {seen:,} loss {loss:.3f} {(time.time()-t_start)/60:.1f} мин")
            if (time.time() - t_start) / (GRAD_MIN * 60) >= next_mark or time.time() - t_start >= GRAD_MIN * 60:
                a1, a5 = eval_top1(k=20); outs = [(c, g, generate([BOS] + enc(toks(c))[-(MAXLEN // 2):] + [SEP], seed=i)) for i, (c, g, _) in enumerate(TEST_D[:60])]; gj, rj = judge_pairs(outs)
                CURVE.append(dict(frac=round(next_mark, 2), step=step, tokens=seen, top1=a1, top5=a5, reply_gold=gj, reply_rand=rj, loss=loss)); log(f"[кривая] {int(next_mark*100):3d}% времени, шаг {step}, {seen:,} ток: top-1 {a1:.3f} top-5 {a5:.3f} loss {loss:.3f} | ответ↔эталон {gj:.3f} vs случ {rj:.3f}")
                next_mark = round(next_mark + (0.15 if next_mark < 0.25 else 0.25), 2)
                if next_mark > 1.0: next_mark = 9
            if time.time() - t_start >= GRAD_MIN * 60: break
    a1, a5 = eval_top1(); outs = [(c, g, generate([BOS] + enc(toks(c))[-(MAXLEN // 2):] + [SEP], seed=i)) for i, (c, g, _) in enumerate(TEST_D[:300])]; gj, rj = judge_pairs(outs)
    chain = []; right = steps = 0
    for q, gold in GSM_TEST[:50]:
        a = generate([BOS] + enc(toks(q))[-100:], max_new=80, temp=0.6, seed=3); nums = re.findall(r"\d+", a); steps += bool(re.search(r"= \d+", a)); right += (gold in nums[-2:]) if nums else 0; chain.append((q, gold, a))
    stories = [(p, generate([BOS] + enc(toks(p)), max_new=60, seed=s)) for s, p in enumerate(STORY_PROMPTS)]
    for p, s_ in stories: log(f"[история] {p} → {s_[:150]}")
    log(f"ИТОГ grad: top-1 {a1:.3f} (триграмма {tri_acc:.3f}) top-5 {a5:.3f}; ответы {gj:.3f} vs случ {rj:.3f} (×{gj/max(rj,1e-9):.1f}); GSM8K верных {right}/50, структура {steps}; шагов {step}, токенов {seen:,}, {(time.time()-t_start)/60:.0f} мин")
    np.savez_compressed(os.path.join(OUT, "grad_model.npz"), **eng.params_numpy()); json.dump(cfg, open(os.path.join(OUT, "grad_cfg.json"), "w")); json.dump(itos, open(os.path.join(OUT, "vocab.json"), "w"))
    save(dict(mode="grad", d=D_T, layers=NL, params=n_params(p0), V=V, tokens=NTOK, steps=step, seen=seen, trigram=tri_acc, top1=a1, top5=a5, reply_gold=gj, reply_rand=rj, curve=CURVE, gsm_right=right, gsm_steps=steps), outs, chain, stories)

# ================================================================ MODE=hybrid
elif MODE == "hybrid":
    """ГИБРИД (24.10, его «скрестим лучшее»): смысл слов формулой (PPMI→SVD) + память случаев
       + резервуар как ПРИЗНАКИ контекста (слои из deep) + ridge-решение «контекст → СМЫСЛ ОТВЕТА» (выученное отображение,
       не поиск по похожести контекста) + СБОРКА ответа: каркас случая, слоты (имена/числа/содержательные слова) заменяются
       по смыслу из нового контекста. Линейки: ответы vs случайный (разбивка по диалогам), echo, доля НОВЫХ ответов (не копия
       ни одного ответа памяти), сравнение «поиск по контексту» vs «по предсказанному смыслу ответа»."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import svds
    WIN = 4; DM = 48 if SMOKE else 200
    co = collections.Counter(); wc = collections.Counter()
    corpus = [toks(c) + toks(r) for c, r, _ in TRAIN_D] + [toks(s) for s, _ in TEXT]
    for ws in corpus:
        ids = [stoi.get(w, UNK) for w in ws]
        for i, a in enumerate(ids):
            if a == UNK: continue
            wc[a] += 1
            for j in range(max(0, i - WIN), min(len(ids), i + WIN + 1)):
                if j != i and ids[j] != UNK: co[(a, ids[j])] += 1
    tot = sum(co.values()); r_, c_, v_ = zip(*[(a, b, max(0.0, math.log(n * tot / (wc[a] * wc[b])))) for (a, b), n in co.items()])
    U, S, _ = svds(sp.csr_matrix((v_, (r_, c_)), shape=(V, V), dtype=np.float32), k=DM); WV = (U * np.sqrt(S)).astype(np.float32)
    WV /= (np.linalg.norm(WV, axis=1, keepdims=True) + 1e-6); WV[UNK] = 0; WVg = xp.asarray(WV)
    idf = {w: math.log(len(corpus) / (1 + n)) for w, n in collections.Counter(w for ws in corpus for w in set(ws)).items()}
    def vec(text):
        ws = [w for w in toks(text) if w not in STOP and w in stoi]
        if not ws: return np.zeros(DM, np.float32)
        v = sum(idf.get(w, 1.0) * WV[stoi[w]] for w in ws); return v / (np.linalg.norm(v) + 1e-6)
    log(f"[смысл] PPMI→SVD d={DM}; happy →", [itos[j] for j in np.argsort(-(WV @ WV[stoi['happy']]))[1:5]] if 'happy' in stoi else '-')
    # ---- резервуар как признаки контекста: читает последнюю реплику контекста словами-смыслами, конечное состояние = признак
    NR = 300 if SMOKE else 1500; rr_ = np.random.default_rng(1)
    Win = xp.asarray((rr_.normal(0, 1, (DM, NR)) * 0.5).astype(np.float32)); Wr = rr_.normal(0, 1, (NR, NR)).astype(np.float32); Wr *= (rr_.random((NR, NR)) < 10 / NR)
    vv = rr_.normal(size=NR).astype(np.float32)
    for _ in range(60): vv = Wr @ vv; vv /= (np.linalg.norm(vv) + 1e-9)
    Wr = xp.asarray(Wr * (0.9 / (np.linalg.norm(Wr @ vv) + 1e-9)))
    def ctx_feats(texts):
        """[средний смысл всего контекста | средний смысл последней реплики | состояние резервуара по последней реплике | 1]"""
        out = np.zeros((len(texts), DM * 2 + NR + 1), np.float32)
        for i, t in enumerate(texts):
            last = re.split(r"(?<=[.!?])\s+", t.strip())[-1] if t.strip() else t
            ids = [stoi[w] for w in toks(last) if w in stoi and w not in STOP][-32:]
            h = xp.zeros((1, NR), xp.float32)
            for w in ids: h = xp.tanh(WVg[w][None, :] @ Win + h @ Wr)
            out[i, :DM] = vec(t); out[i, DM:2 * DM] = vec(last); out[i, 2 * DM:2 * DM + NR] = asnp(h)[0]; out[i, -1] = 1.0
        return out
    t1 = time.time(); PHI = ctx_feats([c for c, _, _ in TRAIN_D]); Y = np.stack([vec(r) for _, r, _ in TRAIN_D]); log(f"признаки контекста {PHI.shape} за {time.time()-t1:.0f}s")
    Pg = xp.asarray(PHI); A = Pg.T.astype(xp.float64) @ Pg.astype(xp.float64); B = Pg.T.astype(xp.float64) @ xp.asarray(Y).astype(xp.float64)
    Wmap = asnp(xp.linalg.solve(A + 5.0 * xp.eye(A.shape[0], dtype=xp.float64), B)).astype(np.float32)     # контекст-признаки → смысл ответа (одна формула)
    RV = np.stack([vec(r) for _, r, _ in TRAIN_D]); RVg = xp.asarray(RV); CV = PHI[:, :DM]; CVg = xp.asarray(CV)
    PHI_T = ctx_feats([c for c, _, _ in TEST_D]); PRED = PHI_T @ Wmap; PRED /= (np.linalg.norm(PRED, axis=1, keepdims=True) + 1e-6)
    gold_v = np.stack([vec(g) for _, g, _ in TEST_D]); rnd = gold_v[np.random.default_rng(0).permutation(len(gold_v))]
    log(f"[отображение] cos(предсказанный смысл ответа, настоящий) {np.mean(np.sum(PRED*gold_v,1)):.3f} vs случайный {np.mean(np.sum(PRED*rnd,1)):.3f} | по контексту: {np.mean(np.sum(PHI_T[:, :DM]*gold_v,1)):.3f}")
    NAME_RX = re.compile(r"\b([A-Z][a-z]{2,})\b"); NUM_RX = re.compile(r"\b\d+(?:\.\d+)?\b")
    TRAIN_REPLIES = set(r.strip() for _, r, _ in TRAIN_D)
    def compose(reply, case_ctx, new_ctx, pred_v):
        """1) имена/числа по позиции; 2) содержательное слово каркаса, которое было и в контексте случая, но не в новом →
           ближайшее по смыслу слово НОВОГО контекста (слот, как [U:word] в analogy.py)."""
        SENT_START = {"once", "the", "one", "then", "they", "she", "he", "it", "but", "and", "when", "after", "there", "this", "that", "his", "her", "you", "what", "how", "why", "where", "yes", "oh", "well", "okay", "i'm", "it's", "that's"}
        name = lambda m: m.lower() not in SENT_START and m.lower() not in STOP
        for rx in (NAME_RX, NUM_RX):
            old = [m for m in rx.findall(case_ctx) if m in reply and name(m)]; new = [m for m in rx.findall(new_ctx) if name(m)]
            for o, n in zip(old, new): reply = re.sub(r"\b" + re.escape(o) + r"\b", n, reply)
        cw = [w for w in toks(case_ctx) if w not in STOP and w in stoi]; nw = [w for w in toks(new_ctx) if w not in STOP and w in stoi and len(w) > 2]
        if not nw: return reply
        NWV = WV[[stoi[w] for w in nw]]
        out = []
        for tok in re.findall(r"[A-Za-z']+|[^A-Za-z']+", reply):
            w = tok.lower()
            if w in stoi and w not in STOP and len(w) > 2 and w in cw and w not in nw:
                sims = NWV @ WV[stoi[w]]; j = int(np.argmax(sims))
                if sims[j] > 0.35: tok = nw[j] if tok.islower() else nw[j].capitalize()
            out.append(tok)
        return "".join(out)
    used = collections.Counter()
    def answer(i, by="pred", k=5, do_compose=True):
        c = TEST_D[i][0]; q = PRED[i] if by == "pred" else PHI_T[i, :DM]
        s = asnp((RVg if by == "pred" else CVg) @ xp.asarray(q)); top = np.argsort(-s)[:k]
        best = min(top, key=lambda j: (used[(by, j)], -s[j])); used[(by, best)] += 1
        rep = TRAIN_D[best][1]; return (compose(rep, TRAIN_D[best][0], c, PRED[i]) if do_compose else rep), float(s[best])
    res = {}
    for name, by, comp in (("поиск по контексту", "ctx", False), ("по смыслу ответа", "pred", False), ("по смыслу ответа + сборка", "pred", True)):
        outs = [(c, g, answer(i, by, do_compose=comp)[0]) for i, (c, g, _) in enumerate(TEST_D[:300])]; gj, rj = judge_pairs(outs)
        novel = sum(1 for _, _, a in outs if a.strip() not in TRAIN_REPLIES); res[name] = dict(gold=gj, rand=rj, x=gj / max(rj, 1e-9), echo=judge_pairs.echo, novel=novel, outs=outs)
        log(f"[{name}] ответ↔эталон {gj:.3f} vs случайный {rj:.3f} (×{gj/max(rj,1e-9):.1f}); echo {judge_pairs.echo}/300; новых ответов (нет в памяти) {novel}/300")
    outs = res["по смыслу ответа + сборка"]["outs"]
    SENTS, NEXT = [], []
    for st in STORIES[:20000]:
        ss = [x.strip() for x in re.split(r"(?<=[.!?])\s+", st) if x.strip()]
        for i in range(len(ss) - 1): SENTS.append(ss[i]); NEXT.append(ss[i + 1])
    stories = []
    if SENTS:
        SVg = xp.asarray(np.stack([vec(x) for x in SENTS]))
        for p in STORY_PROMPTS:
            out = []; cur = p; seen = set()
            for _ in range(5):
                s = asnp(SVg @ xp.asarray(vec(cur))); i = next(j for j in np.argsort(-s)[:30] if j not in seen); seen.add(i)
                nxt = compose(NEXT[i], SENTS[i], p, None); out.append(nxt); cur = nxt
            stories.append((p, " ".join(out)))
        for p, st in stories: log(f"[история] {p} → {st[:220]}")
    chain = []
    log(f"ИТОГ hybrid: поиск ×{res['поиск по контексту']['x']:.1f} → смысл ответа ×{res['по смыслу ответа']['x']:.1f} → +сборка ×{res['по смыслу ответа + сборка']['x']:.1f}; новых ответов {res['по смыслу ответа + сборка']['novel']}/300; echo {res['по смыслу ответа + сборка']['echo']}")
    save(dict(mode="hybrid", V=V, dm=DM, cases=len(TRAIN_D), split="dialog" if SPLIT_BY_DIALOG else "pairs", variants={k: {kk: vv for kk, vv in v.items() if kk != "outs"} for k, v in res.items()}), outs, chain, stories)

# ================================================================ MODE=cases
elif MODE == "cases":
    # ---- смысл слов без градиента: PPMI соседств (окно 4) → SVD (формула)
    WIN = 4; DM = 48 if SMOKE else 200; TOPW = min(V, 12000)
    rows, cols, vals = [], [], []
    co = collections.Counter(); wc = collections.Counter(); nsent = 0
    corpus = [toks(c) + toks(r) for c, r, _ in TRAIN_D] + [toks(s) for s, _ in TEXT]
    for ws in corpus:
        ids = [stoi.get(w, UNK) for w in ws]
        for i, a in enumerate(ids):
            if a == UNK: continue
            wc[a] += 1
            for j in range(max(0, i - WIN), min(len(ids), i + WIN + 1)):
                if j != i and ids[j] != UNK: co[(a, ids[j])] += 1
    tot = sum(co.values()); log("соседств", len(co), "токенов", sum(wc.values()))
    import scipy.sparse as sp
    r_, c_, v_ = zip(*[(a, b, max(0.0, math.log(n * tot / (wc[a] * wc[b])))) for (a, b), n in co.items()])
    Mppmi = sp.csr_matrix((v_, (r_, c_)), shape=(V, V), dtype=np.float32)
    from scipy.sparse.linalg import svds
    U, S, _ = svds(Mppmi, k=DM); WV = (U * np.sqrt(S)).astype(np.float32); WV /= (np.linalg.norm(WV, axis=1, keepdims=True) + 1e-6); WV[UNK] = 0
    def sim_words(w, k=5):
        i = stoi.get(w)
        if i is None: return []
        s = WV @ WV[i]; return [itos[j] for j in np.argsort(-s)[1:k + 1]]
    log("[смысл] happy →", sim_words("happy"), "| money →", sim_words("money"), "| dog →", sim_words("dog"))
    idf = {w: math.log(len(corpus) / (1 + n)) for w, n in collections.Counter(w for ws in corpus for w in set(ws)).items()}
    def vec(text):
        ws = [w for w in toks(text) if w not in STOP and w in stoi]
        if not ws: return np.zeros(DM, np.float32)
        v = sum(idf.get(w, 1.0) * WV[stoi[w]] for w in ws); return v / (np.linalg.norm(v) + 1e-6)
    # ---- память случаев: контекст → ответ
    CV = np.stack([vec(c) for c, _, _ in TRAIN_D]); CVg = xp.asarray(CV); log("память случаев", len(TRAIN_D))
    NAME_RX = re.compile(r"\b([A-Z][a-z]{2,})\b"); NUM_RX = re.compile(r"\b\d+(?:\.\d+)?\b")
    def adapt(reply, case_ctx, new_ctx):
        """перенос по позиции: имена/числа ответа, которые были в контексте случая, → имена/числа нового контекста"""
        for rx in (NAME_RX, NUM_RX):
            old = [m for m in rx.findall(case_ctx) if m in reply]; new = rx.findall(new_ctx)
            for o, n in zip(old, new): reply = reply.replace(o, n)
        return reply
    used = collections.Counter()
    def reply(ctx, k=5):
        s = asnp(CVg @ xp.asarray(vec(ctx))); top = np.argsort(-s)[:k]
        best = min(top, key=lambda i: (used[i], -s[i])); used[best] += 1
        return adapt(TRAIN_D[best][1], TRAIN_D[best][0], ctx), float(s[best])
    outs = []; sims = []
    for c, g, _ in TEST_D[:300]:
        a, s = reply(c); outs.append((c, g, a)); sims.append(s)
    gj, rj = judge_pairs(outs); log(f"[ответы] ответ↔эталон {gj:.3f} vs случайный {rj:.3f} (×{gj/max(rj,1e-9):.1f}); средняя близость случая {np.mean(sims):.2f}; разных ответов {len(set(a for _,_,a in outs))}/300")
    # ---- цепочка: ближайшая решённая задача → та же арифметика на новых числах (перенос случая по форме)
    def nums_of(s): return [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s.replace(",", ""))]
    def ops_chain(qnums, sol):
        """из решения «<<a*b=c>>» строим план: каждое число результата = op(индексы известных чисел); возвращает список шагов или None"""
        known = list(qnums); plan = []
        for a, op, b, c in re.findall(r"<<([\d.]+)([-+*/])([\d.]+)=([\d.]+)>>", sol.replace(",", "")):
            a, b, c = float(a), float(b), float(c)
            def idx(v):
                for i, kv in enumerate(known):
                    if abs(kv - v) < 1e-6: return i
                return None
            ia, ib = idx(a), idx(b)
            if ia is None or ib is None: return None
            plan.append((ia, op, ib)); known.append(c)
        return plan if plan else None
    GV = np.stack([vec(q) for q, _ in GSM]) if GSM else None; GVg = xp.asarray(GV) if GSM else None
    def roles(q):
        """форма задачи: слово-роль при каждом числе (ближайшее содержательное слово после/перед числом): «3 boxes», «$5 each»"""
        ws = toks(q.replace(",", "")); out = []
        for i, w in enumerate(ws):
            if re.fullmatch(r"\d+(?:\.\d+)?", w):
                nb = [x for x in ws[i + 1:i + 3] + ws[max(0, i - 2):i] if x not in STOP and not re.fullmatch(r"\d+(?:\.\d+)?", x) and x in stoi]
                out.append(nb[0] if nb else "")
        return out
    def form_sim(ra, rb):
        if len(ra) != len(rb) or not ra: return 0.0
        return float(np.mean([(WV[stoi[a]] @ WV[stoi[b]]) if a and b else 0.0 for a, b in zip(ra, rb)]))
    FORM_THR = float(E("FORM_THR", 0.35))
    def solve_chain(q, k=12):
        qn = nums_of(q); qr = roles(q)
        if not qn or GV is None: return None, None
        s = asnp(GVg @ xp.asarray(vec(q)))
        for i in np.argsort(-s)[:k]:
            cq, csol = GSM[i]; cn = nums_of(cq)
            if len(cn) != len(qn): continue
            if form_sim(qr, roles(cq)) < FORM_THR: continue      # та же ФОРМА (роли чисел близки по смыслу), не только та же тема
            plan = ops_chain(cn, csol)
            if not plan: continue
            known = list(qn); steps = []
            for ia, op, ib in plan:
                a, b = known[ia], known[ib]
                if op == "/" and b == 0: break
                c = {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[op]; known.append(c); steps.append(f"{a:g} {op} {b:g} = {c:g}")
            else:
                return " ; ".join(steps) + f" ; the answer is {known[-1]:g}", cq
        return None, None
    chain = []; right = tried = 0
    for q, gold in GSM_TEST:
        a, src = solve_chain(q); tried += a is not None
        if a is not None:
            try: right += abs(float(a.split()[-1]) - float(gold.replace(",", ""))) < 1e-6
            except Exception: pass
        chain.append((q, gold, (a or "— нет случая той же формы") + (f"   [по случаю: {src[:80]}…]" if src else "")))
    log(f"[цепочка] GSM8K {len(GSM_TEST)} задач: взялась за {tried}, верных {right} ({right/len(GSM_TEST):.1%}); молчит {len(GSM_TEST)-tried}")
    # ---- истории: сцепка предложений по смыслу (слой поиска, честно)
    SENTS = []; NEXT = []
    for s in STORIES[:20000]:
        ss = [x.strip() for x in re.split(r"(?<=[.!?])\s+", s) if x.strip()]
        for i in range(len(ss) - 1): SENTS.append(ss[i]); NEXT.append(ss[i + 1])
    SV = np.stack([vec(x) for x in SENTS]) if SENTS else None; SVg = xp.asarray(SV) if SENTS else None
    def story(prompt, n=5):
        out = []; cur = prompt; seen = set()
        for _ in range(n):
            s = asnp(SVg @ xp.asarray(vec(cur)));
            for i in np.argsort(-s)[:20]:
                if i not in seen: seen.add(i); break
            nxt = adapt(NEXT[i], SENTS[i], prompt); out.append(nxt); cur = nxt
        return " ".join(out)
    stories = [(p, story(p)) for p in STORY_PROMPTS] if SENTS else []
    for p, s in stories: log(f"[история] {p} → {s[:200]}")
    log(f"ИТОГ cases: ответы ×{gj/max(rj,1e-9):.1f} случайного; цепочка {right}/{len(GSM_TEST)} верных переносом случая; истории — сцепка чужих предложений (поиск)")
    save(dict(mode="cases", V=V, dm=DM, cases=len(TRAIN_D), reply_gold=gj, reply_rand=rj, mean_sim=float(np.mean(sims)), distinct=len(set(a for _, _, a in outs)), gsm_tried=tried, gsm_right=right, gsm_n=len(GSM_TEST)), outs, chain, stories)
