"""speech_en_reservoir — РЕЧЬ ПО-АНГЛИЙСКИ БЕЗ ГРАДИЕНТА, на Kaggle T4 (24.10.2026, его «прогони обучение для диалогов на
английском, отборного английского, макс объём, чтоб цепочку понимал и мог сам выдумывать»).

Механизм — наш резервуар (exp62 → solver.py): случайная замороженная рекуррентная сеть читает текст, единственный выход
решается ОДНОЙ формулой W = (A+λI)⁻¹B из накопленных сумм. Ни одного шага градиента, ни одной чужой нейросети.
Данные (качественный английский, качаются прямо на Kaggle): DailyDialog, EmpatheticDialogues, PersonaChat (диалоги),
TinyStories (связные истории — «цепочка» событий: because/so/then), GSM8K (задачи с пошаговым решением — цепочка рассуждения).
Гипотеза + числа успеха, записанные ДО запуска:
  A. next-word top-1 на отложенных репликах ≥ триграммы (контроль) — иначе резервуар не выучил язык;
  B. кривая по объёму: точность на 10 % / 25 % / 50 % / 100 % данных растёт (наклон > 0 на последнем отрезке — иначе упёрлись);
  C. целые ответы: доля содержательных слов ответа, совпавших с эталоном, выше, чем со случайным эталоном (≥ ×1.5);
  D. цепочка: GSM8K-тест, 50 задач, генерация решения — честно ждём ≈0 правильных чисел; смотрим, появляются ли шаги вида
     «... = N» (структура цепочки), это и есть «понимает цепочку» на первом уровне;
  E. one-shot: 20 новых предложений, одно обновление → их продолжения ≥ 0.5, тест не падает > 0.01 (то, чего нет у GPT).
Выход (/kaggle/working): metrics.json, samples.txt (все сгенерированные ответы/истории, без отбора), wout.npy (float16), vocab.json, cfg.json.
Локальная проверка: SMOKE=1 python3 tools/speech_en_reservoir.py (CPU, крошечные данные, ~1 мин).
"""
import os, re, io, sys, json, time, math, collections, urllib.request
import numpy as np

SMOKE = os.environ.get("SMOKE") == "1"
OUT = os.environ.get("OUT", "/kaggle/working" if os.path.isdir("/kaggle/working") else "/tmp/speech_en")
os.makedirs(OUT, exist_ok=True)
T0 = time.time()
def log(*a):
    s = " ".join(str(x) for x in a); print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)
    with open(os.path.join(OUT, "log.txt"), "a") as f: f.write(s + "\n")

# ---------------------------------------------------------------- вычислитель: cupy (T4) или numpy
try:
    import cupy as cp; cp.zeros(1).sum(); xp = cp; GPU = True
    import cupyx
except Exception as e:
    cp = None; xp = np; GPU = False
log("GPU" if GPU else "CPU", "cupy" if GPU else "numpy", "SMOKE" if SMOKE else "FULL")

N_RES = int(os.environ.get("N", 300 if SMOKE else 8000)); D_EMB = int(os.environ.get("D", 64 if SMOKE else 192))
VOCAB = int(os.environ.get("VOCAB", 2000 if SMOKE else 16000)); LAM = float(os.environ.get("LAM", 2.0))
MAXLEN = 48 if SMOKE else 128; BS = 64 if SMOKE else 256
STORY_SHARDS = int(os.environ.get("STORY_SHARDS", 0 if SMOKE else 2)); MAX_STORIES = int(os.environ.get("MAX_STORIES", 200 if SMOKE else 400000))

# ---------------------------------------------------------------- данные
HF = "https://huggingface.co/datasets/{repo}/resolve/refs%2Fconvert%2Fparquet/{cfg}/{split}/{i:04d}.parquet"
def fetch(url, tries=3):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mini-ai"}); return urllib.request.urlopen(req, timeout=120).read()
        except Exception as e:
            log("  retry", k, url[-60:], type(e).__name__); time.sleep(5)
    return None
def parquet(url):
    b = fetch(url)
    if b is None: return None
    import pyarrow.parquet as pq
    return pq.read_table(io.BytesIO(b)).to_pandas()

DIAL = []      # (context_text, reply_text, source)
TEXT = []      # (text, source)  — истории и решения задач
def add_dialog(turns, src):
    turns = [t.strip() for t in turns if t and t.strip()]
    for i in range(1, len(turns)):
        ctx = " ".join(turns[max(0, i - 2):i]); DIAL.append((ctx, turns[i], src))

t = parquet(HF.format(repo="li2017dailydialog/daily_dialog", cfg="default", split="train", i=0))
if t is not None:
    for d in t["dialog"]: add_dialog(list(d), "dailydialog")
    log("dailydialog", len(t), "диалогов →", len(DIAL), "пар")
t = parquet(HF.format(repo="facebook/empathetic_dialogues", cfg="default", split="train", i=0))
if t is not None:
    n0 = len(DIAL)
    for cid, g in t.sort_values(["conv_id", "utterance_idx"]).groupby("conv_id"):
        add_dialog([u.replace("_comma_", ",") for u in g["utterance"].tolist()], "empathetic")
    log("empathetic", len(DIAL) - n0, "пар")
t = parquet(HF.format(repo="bavard/personachat_truecased", cfg="full", split="train", i=0))
if t is not None:
    n0 = len(DIAL); seen = set()
    for h in t["history"]:
        h = list(h); key = tuple(h[:3])
        if key in seen: continue
        seen.add(key); add_dialog(h, "personachat")
    log("personachat", len(DIAL) - n0, "пар")
if SMOKE: DIAL = DIAL[:3000]
MAX_DIAL = int(os.environ.get("MAX_DIAL", 0))
if MAX_DIAL: DIAL = DIAL[:MAX_DIAL]
t = parquet(HF.format(repo="openai/gsm8k", cfg="main", split="train", i=0))
GSM_TEST = []
if t is not None:
    for q, a in zip(t["question"], t["answer"]):
        a = re.sub(r"<<[^>]*>>", "", a).replace("####", "the answer is")
        TEXT.append((q + " " + a, "gsm8k"))
    tt = parquet(HF.format(repo="openai/gsm8k", cfg="main", split="test", i=0))
    if tt is not None: GSM_TEST = [(q, a.split("####")[-1].strip()) for q, a in zip(tt["question"][:50], tt["answer"][:50])]
    log("gsm8k", len(t), "задач с решением")
for i in range(STORY_SHARDS):
    t = parquet(HF.format(repo="roneneldan/TinyStories", cfg="default", split="train", i=i))
    if t is None: break
    col = "text" if "text" in t.columns else t.columns[0]
    for s in t[col]:
        TEXT.append((s, "tinystories"))
        if sum(1 for _, src in TEXT if src == "tinystories") >= MAX_STORIES: break
    log("tinystories shard", i, "→ всего историй", sum(1 for _, src in TEXT if src == "tinystories"))
    if sum(1 for _, src in TEXT if src == "tinystories") >= MAX_STORIES: break
if SMOKE and not TEXT: TEXT = [("Tom had three apples. He ate one, so he had two apples left. Then he gave one to Ann.", "smoke")] * 50
log("ИТОГО: диалог-пар", len(DIAL), "текстов", len(TEXT))

def toks(s): return re.findall(r"[a-z]+(?:'[a-z]+)?|\d+|[.,!?;:]", s.lower())
rng = np.random.default_rng(0); idx = rng.permutation(len(DIAL)); n_test = max(200, len(DIAL) // 50)
TEST_D = [DIAL[i] for i in idx[:n_test]]; TRAIN_D = [DIAL[i] for i in idx[n_test:]]
cnt = collections.Counter()
for c, r, _ in TRAIN_D: cnt.update(toks(c)); cnt.update(toks(r))
for s, _ in TEXT: cnt.update(toks(s))
itos = ["<pad>", "<unk>", "<eos>", "<bos>", "<sep>"] + [w for w, _ in cnt.most_common(VOCAB)]
stoi = {w: i for i, w in enumerate(itos)}; V = len(itos); PAD, UNK, EOS, BOS, SEP = 0, 1, 2, 3, 4
enc = lambda ws: [stoi.get(w, UNK) for w in ws]
cover = sum(cnt[w] for w in itos) / max(1, sum(cnt.values())); log(f"словарь {V} слов, покрытие {cover:.3f}, всего токенов в корпусе {sum(cnt.values()):,}")

def seq_dialog(c, r):
    x = [BOS] + enc(toks(c))[-(MAXLEN // 2):] + [SEP] + enc(toks(r))[: MAXLEN // 2] + [EOS]
    m = [0] * (x.index(SEP) + 1) + [1] * (len(x) - x.index(SEP) - 1)  # учим только ответ
    return x, m[:-1]
def seq_text(s):
    ids = enc(toks(s)); out = []
    for i in range(0, len(ids), MAXLEN - 2):
        x = [BOS] + ids[i:i + MAXLEN - 2] + [EOS]; out.append((x, [1] * (len(x) - 1)))
    return out
TRAIN = [seq_dialog(c, r) for c, r, _ in TRAIN_D] + [p for s, _ in TEXT for p in seq_text(s)]
TEST = [seq_dialog(c, r) for c, r, _ in TEST_D]
rng.shuffle(TRAIN); log("последовательностей: train", len(TRAIN), "test", len(TEST), "токенов train", sum(len(x) for x, _ in TRAIN))

def batches(seqs, bs):
    seqs = sorted(seqs, key=len)
    for i in range(0, len(seqs), bs):
        chunk = seqs[i:i + bs]; T = max(len(x) for x, _ in chunk)
        X = np.full((len(chunk), T), EOS, np.int64); M = np.zeros((len(chunk), T - 1), bool)
        for j, (x, m) in enumerate(chunk): X[j, :len(x)] = x; M[j, :len(m)] = m
        yield X, M

# ---------------------------------------------------------------- контроль: триграмма с откатом
def ngram_eval():
    uni, bi, tri = collections.Counter(), collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    for x, m in TRAIN:
        for i in range(len(x) - 1):
            uni[x[i + 1]] += 1; bi[x[i]][x[i + 1]] += 1
            if i >= 1: tri[(x[i - 1], x[i])][x[i + 1]] += 1
    ok = n = 0
    for x, m in TEST:
        for i in range(len(x) - 1):
            if not m[i]: continue
            c = tri.get((x[i - 1], x[i])) if i >= 1 else None
            p = (c or bi.get(x[i]) or uni).most_common(1)[0][0]; ok += (p == x[i + 1]); n += 1
    return ok / n
TRI_LIMIT = int(os.environ.get("TRI_LIMIT", 0))   # контроль на срезе корпуса (ОЗУ Kaggle ~13 ГБ: триграмма на 88M токенов не влезает)
if TRI_LIMIT:
    _all = TRAIN; TRAIN = []; _n = 0
    for x, m in _all:
        TRAIN.append((x, m)); _n += len(x)
        if _n >= TRI_LIMIT: break
    tri_acc = ngram_eval(); TRAIN = _all; log(f"[контроль] триграмма (на {_n:,} токенах) top-1 на ответах: {tri_acc:.3f}")
else:
    tri_acc = ngram_eval(); log(f"[контроль] триграмма top-1 на ответах: {tri_acc:.3f}")

# ---------------------------------------------------------------- резервуар
class Reservoir:
    def __init__(self, V, d, N, rho=0.9, in_scale=0.4, k=10, lam=1.0, seed=0):
        r = np.random.default_rng(seed)
        E = (r.choice([-1.0, 1.0], size=(V, d)) / np.sqrt(d)).astype(np.float32)
        Win = (r.normal(0, 1, (d, N)) * in_scale).astype(np.float32)
        W = r.normal(0, 1, (N, N)).astype(np.float32); W *= (r.random((N, N)) < k / N)
        v = r.normal(size=N).astype(np.float32)
        for _ in range(60): v = W @ v; v /= (np.linalg.norm(v) + 1e-9)
        W *= rho / (np.linalg.norm(W @ v) + 1e-9)
        self.E, self.Win, self.Wres = xp.asarray(E), xp.asarray(Win), xp.asarray(W)
        self.V, self.d, self.N, self.lam = V, d, N, lam; self.F = N + 2 * d + 1
        self.A = xp.zeros((self.F, self.F), xp.float64); self.B = xp.zeros((V, self.F), xp.float64); self.Wout = None; self.ntok = 0
    def step(self, h, x_ids, prev_ids):
        x = self.E[x_ids]; h = xp.tanh(x @ self.Win + h @ self.Wres)
        phi = xp.concatenate([h, x, self.E[prev_ids], xp.ones((len(x_ids), 1), xp.float32)], axis=1); return h, phi
    def accumulate(self, X, M, weight=1.0):
        Xg = xp.asarray(X); Bt, T = X.shape; h = xp.zeros((Bt, self.N), xp.float32)
        A32 = xp.zeros((self.F, self.F), xp.float32); B32 = xp.zeros((self.V, self.F), xp.float32)
        for t in range(T - 1):
            h, phi = self.step(h, Xg[:, t], Xg[:, t - 1] if t > 0 else Xg[:, t])
            m = xp.asarray(M[:, t])
            if not bool(m.any()): continue
            P = phi[m]; y = Xg[m, t + 1]
            A32 += P.T @ P
            if GPU: cupyx.scatter_add(B32, y, P)
            else: np.add.at(B32, y, P)
            self.ntok += int(m.sum())
        self.A += weight * A32.astype(xp.float64); self.B += weight * B32.astype(xp.float64)
    def solve(self):
        A = self.A + self.lam * xp.eye(self.F, dtype=xp.float64)
        self.Wout = xp.linalg.solve(A, self.B.T).astype(xp.float32)
    def eval(self, seqs, bs=256):
        ok1 = ok5 = n = 0
        for X, M in batches(seqs, bs):
            Xg = xp.asarray(X); h = xp.zeros((X.shape[0], self.N), xp.float32)
            for t in range(X.shape[1] - 1):
                h, phi = self.step(h, Xg[:, t], Xg[:, t - 1] if t > 0 else Xg[:, t]); m = xp.asarray(M[:, t])
                if not bool(m.any()): continue
                lg = phi[m] @ self.Wout; y = Xg[m, t + 1]
                ok1 += int((lg.argmax(1) == y).sum()); top5 = xp.argsort(-lg, axis=1)[:, :5]; ok5 += int((top5 == y[:, None]).any(1).sum()); n += int(m.sum())
        return ok1 / max(1, n), ok5 / max(1, n)
    def generate(self, prefix, max_new=40, temp=0.8, seed=1, greedy=False):
        r = np.random.default_rng(seed); ids = list(prefix); h = xp.zeros((1, self.N), xp.float32)
        for i in range(len(ids) + max_new):
            h, phi = self.step(h, xp.asarray([ids[i]]), xp.asarray([ids[i - 1] if i > 0 else ids[i]]))
            if i < len(ids) - 1: continue
            out = phi @ self.Wout; out = out[0] if not GPU else cp.asnumpy(out[0])
            out[PAD] = out[BOS] = out[SEP] = -1e9; out[UNK] = min(out[UNK], 1e-4)
            if greedy: nxt = int(out.argmax())
            else:   # выход ridge ≈ вероятности (не логиты): срез, степень 1/temp, нормировка — как в exp62
                p = np.clip(out, 1e-4, None) ** (1 / max(temp, 1e-3)); p[[PAD, BOS, SEP]] = 0; p /= p.sum(); nxt = int(r.choice(self.V, p=p))
            ids.append(nxt)
            if nxt == EOS: break
        return " ".join(itos[i] for i in ids[len(prefix):] if i != EOS)

R = Reservoir(V, D_EMB, N_RES, lam=LAM); log(f"резервуар N={N_RES} d={D_EMB} F={R.F} V={V}; память A {R.F*R.F*8/1e9:.2f} ГБ, B {V*R.F*8/1e9:.2f} ГБ")
STOP = set("the a an to of and i you it is in that was he she my me we they for on with be have do are this at but not so as his her him".split())
def content(s): return [w for w in toks(s) if w not in STOP and len(w) > 2]
def reply_judge(k=300, seed=0):
    """целые ответы: совпадение содержательных слов ответа с эталоном vs со случайным эталоном"""
    r = np.random.default_rng(seed); pick = r.choice(len(TEST_D), size=min(k, len(TEST_D)), replace=False); gold_hit = rand_hit = 0; n = 0; outs = []
    for i in pick:
        c, g, _ = TEST_D[i]; pre = [BOS] + enc(toks(c))[-(MAXLEN // 2):] + [SEP]
        a = R.generate(pre, seed=int(i)); A = set(content(a)); G = set(content(g)); Rg = set(content(TEST_D[int(r.integers(len(TEST_D)))][1]))
        if A: gold_hit += len(A & G) / len(A); rand_hit += len(A & Rg) / len(A); n += 1
        outs.append((c, g, a))
    return gold_hit / max(1, n), rand_hit / max(1, n), outs
def chain_test():
    res = []; steps_seen = 0; right = 0
    for q, gold in GSM_TEST:
        pre = [BOS] + enc(toks(q))[-100:]; a = R.generate(pre, max_new=80, seed=3, temp=0.6)
        nums = re.findall(r"\d+", a); steps_seen += len(re.findall(r"= \d+", a)) > 0; right += gold in nums[-2:] if nums else 0; res.append((q, gold, a))
    return right, steps_seen, res

# ---------------------------------------------------------------- обучение по частям → кривая от объёма
CURVE = []; milestones = [0.1, 0.25, 0.5, 1.0]; all_b = list(batches(TRAIN, BS)); nb = len(all_b); done = 0; mi = 0
rng.shuffle(all_b)
for bi, (X, M) in enumerate(all_b, 1):
    R.accumulate(X, M)
    if bi >= milestones[mi] * nb - 1e-9 or bi == nb:
        t1 = time.time(); R.solve(); a1, a5 = R.eval(TEST); gj, rj, _ = reply_judge(k=100)
        CURVE.append(dict(frac=milestones[mi], tokens=R.ntok, top1=a1, top5=a5, reply_gold=gj, reply_rand=rj, t=time.time() - T0))
        log(f"[кривая] {int(milestones[mi]*100):3d}% данных, {R.ntok:,} токенов: top-1 {a1:.3f} top-5 {a5:.3f} | ответ↔эталон {gj:.3f} vs случайный {rj:.3f} | решение {time.time()-t1:.0f}s")
        mi += 1
        if mi >= len(milestones): break
if R.Wout is None: R.solve()
a1, a5 = R.eval(TEST); gj, rj, outs = reply_judge(k=300)
slope = (CURVE[-1]["top1"] - CURVE[-2]["top1"]) / max(1e-9, CURVE[-1]["tokens"] - CURVE[-2]["tokens"]) * 1e6 if len(CURVE) >= 2 else None
log(f"ИТОГ top-1 {a1:.3f} (триграмма {tri_acc:.3f}) top-5 {a5:.3f}; целые ответы {gj:.3f} vs случайный {rj:.3f} (×{gj/max(rj,1e-9):.1f}); наклон последнего отрезка {slope} top-1 на 1M токенов")

# ---------------------------------------------------------------- цепочка (GSM8K) + выдумывание (истории)
right, steps_seen, chain = chain_test() if GSM_TEST else (0, 0, [])
log(f"[цепочка] GSM8K 50 задач: верных чисел {right}, со структурой «= N» {steps_seen}")
STORY_PROMPTS = ["once upon a time , there was a little", "tom had three apples . he", "the dog was sad because", "lily wanted to go to the park , but", "one day , a small robot"]
stories = [(p, R.generate([BOS] + enc(toks(p)), max_new=60, seed=s, temp=0.8)) for s, p in enumerate(STORY_PROMPTS)]
for p, s in stories: log(f"[история] {p} → {s}")

# ---------------------------------------------------------------- one-shot: 20 новых предложений
NEW = ["the purple fox painted the old boat with honey .", "grandma hid the golden spoon under the red sofa .", "a tiny robot sang to the sleepy moon every night .",
       "the brave snail carried a blue flag to the top of the hill .", "my cousin keeps a pet cloud in a glass jar .", "the hungry piano ate three green apples for lunch .",
       "a silver dragon learned to bake bread with the baker .", "the clever duck opened the door with a feather .", "tom planted a candy tree in the kitchen .",
       "the wind borrowed my hat and gave it to the river .", "a shy mountain whispered secrets to the stars .", "the lazy clock refused to wake the sun .",
       "lily built a castle out of warm pancakes .", "the frog wore yellow boots to the party .", "a lonely sock searched the house for its brother .",
       "the teacher kept a dinosaur in the garden .", "max traded his bike for a talking fish .", "the rain painted the street with rainbows .",
       "a paper bird flew over the sleeping town .", "the baby elephant learned to whistle in a week ."]
new_seqs = [([BOS] + enc(toks(s)) + [EOS], [1] * (len(enc(toks(s))) + 1)) for s in NEW]
def new_acc():
    ok = n = 0
    for x, m in new_seqs:
        Xg = xp.asarray([x]); h = xp.zeros((1, R.N), xp.float32)
        for t in range(len(x) - 1):
            h, phi = R.step(h, Xg[:, t], Xg[:, t - 1] if t > 0 else Xg[:, t])
            if t >= 3: ok += int((phi @ R.Wout).argmax(1)[0] == x[t + 1]); n += 1
    return ok / max(1, n)
nb0 = new_acc(); t2 = time.time()
for X, M in batches(new_seqs, 32): R.accumulate(X, M, weight=30.0)
R.solve(); t_one = time.time() - t2; nb1 = new_acc(); a1_after, _ = R.eval(TEST)
log(f"[one-shot] 20 новых предложений: {nb0:.2f} → {nb1:.2f} за одно обновление ({t_one:.1f}s); тест {a1:.3f} → {a1_after:.3f}")

# ---------------------------------------------------------------- сохранить
with open(os.path.join(OUT, "samples.txt"), "w") as f:
    f.write("=== ОТВЕТЫ НА ОТЛОЖЕННЫЕ РЕПЛИКИ (все 300, без отбора) ===\n")
    for c, g, a in outs: f.write(f"CTX: {c}\nGOLD: {g}\nOURS: {a}\n\n")
    f.write("=== ЦЕПОЧКА GSM8K ===\n")
    for q, g, a in chain: f.write(f"Q: {q}\nGOLD: {g}\nOURS: {a}\n\n")
    f.write("=== ИСТОРИИ ===\n")
    for p, s in stories: f.write(f"{p} → {s}\n")
json.dump(dict(gpu=GPU, N=N_RES, d=D_EMB, V=V, lam=LAM, dial_pairs=len(DIAL), texts=len(TEXT), train_tokens=sum(len(x) for x, _ in TRAIN), vocab_cover=cover,
               trigram_top1=tri_acc, top1=a1, top5=a5, reply_gold=gj, reply_rand=rj, curve=CURVE, slope_top1_per_1M=slope, gsm_right=right, gsm_steps=steps_seen,
               oneshot=dict(before=nb0, after=nb1, test_before=a1, test_after=a1_after, seconds=t_one), seconds=time.time() - T0),
          open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
if not SMOKE:
    np.save(os.path.join(OUT, "wout.npy"), (cp.asnumpy(R.Wout) if GPU else R.Wout).astype(np.float16))
    json.dump(itos, open(os.path.join(OUT, "vocab.json"), "w")); json.dump(dict(N=N_RES, d=D_EMB, seed=0, rho=0.9, in_scale=0.4, k=10, lam=LAM, maxlen=MAXLEN), open(os.path.join(OUT, "cfg.json"), "w"))
log(f"[done] {time.time()-T0:.0f}s → {OUT}")
