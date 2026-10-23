"""exp62 — ЗАМЕНА ГРАДИЕНТАМ для речи: резервуар (эхо-сеть) + решение системы уравнений (ridge) вместо backprop.

Гипотеза: случайная фиксированная рекуррентная сеть (никогда не обучается) + ОДНО замкнутое решение
W = (ΦᵀΦ + λI)⁻¹ ΦᵀY даёт предсказание следующего слова не хуже биграммы и не хуже маленького трансформера,
обученного градиентом за то же время CPU; новые предложения выучивает за один проход (без шагов обучения).

Критерий успеха (записан ДО запуска):
  A. top-1 next-word на тесте (data/corpus/en_test.txt): резервуар >= биграмма (контроль) и >= трансформер при равном времени.
  B. one-shot: 20 новых предложений, одно обновление (накопить + решить) → предсказание их продолжений >= 0.5,
     тест при этом не падает больше чем на 0.01.
Контроли: униграмма, биграмма, триграмма с откатом, трансформер (transformer.py, word-level, тот же словарь, то же время).
Запуск: tools/heavy.sh tools/exp62_nograd_lm.py [N=1500] [train_lines=38659] [vocab=8000]
"""
import sys, os, re, time, json, collections
sys.path.insert(0, ".")
import numpy as np
import scipy.sparse as sp

ARGS = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
N_RES = int(ARGS.get("N", 1500)); TRAIN_LINES = int(ARGS.get("train_lines", 10**9)); VOCAB = int(ARGS.get("vocab", 8000))
D_EMB = int(ARGS.get("d", 128)); LAM = float(ARGS.get("lam", 1.0)); TEST_LINES = int(ARGS.get("test_lines", 4000))
TF_SECONDS = ARGS.get("tf_seconds")  # по умолчанию = время обучения резервуара
OUT = os.path.expanduser("~/Developer/mini-ai-runs/learning-core/exp62_nograd_lm.json")
RNG = np.random.default_rng(0)

def toks(line): return re.findall(r"[a-z]+|[.,!?;:]", line.lower())

def load(path, limit):
    out = []
    with open(path) as f:
        for line in f:
            t = toks(line)
            if 3 <= len(t) <= 48: out.append(t)
            if len(out) >= limit: break
    return out

t0 = time.time()
train = load("data/corpus/en.txt", TRAIN_LINES); test = load("data/corpus/en_test.txt", TEST_LINES)
cnt = collections.Counter(w for l in train for w in l)
itos = ["<pad>", "<unk>", "<eos>", "<bos>"] + [w for w, _ in cnt.most_common(VOCAB)]
stoi = {w: i for i, w in enumerate(itos)}; V = len(itos); PAD, UNK, EOS, BOS = 0, 1, 2, 3
enc = lambda l: [BOS] + [stoi.get(w, UNK) for w in l] + [EOS]
train_ids = [enc(l) for l in train]; test_ids = [enc(l) for l in test]
n_train_tok = sum(len(x) - 1 for x in train_ids); n_test_tok = sum(len(x) - 1 for x in test_ids)
print(f"[data] train lines {len(train_ids)} tokens {n_train_tok}  test lines {len(test_ids)} tokens {n_test_tok}  V={V}  ({time.time()-t0:.0f}s)", flush=True)

def batches(seqs, bs):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]; T = max(len(seqs[i]) for i in idx)
        X = np.full((len(idx), T), PAD, np.int64); M = np.zeros((len(idx), T), bool)
        for r, i in enumerate(idx): X[r, :len(seqs[i])] = seqs[i]; M[r, :len(seqs[i]) - 1] = True   # позиции, у которых есть следующий токен
        yield X, M

# ---------------------------------------------------------------- контроли: n-граммы
def ngram_eval():
    uni = np.bincount(np.concatenate([np.array(x[1:]) for x in train_ids]), minlength=V); uni_best = int(uni.argmax())
    bi = collections.defaultdict(collections.Counter); tri = collections.defaultdict(collections.Counter)
    for x in train_ids:
        for i in range(len(x) - 1):
            bi[x[i]][x[i + 1]] += 1
            if i >= 1: tri[(x[i - 1], x[i])][x[i + 1]] += 1
    bi_best = {k: c.most_common(1)[0][0] for k, c in bi.items()}; tri_best = {k: c.most_common(1)[0][0] for k, c in tri.items()}
    ok = {"uni": 0, "bi": 0, "tri": 0}; n = 0
    for x in test_ids:
        for i in range(len(x) - 1):
            y = x[i + 1]; n += 1
            ok["uni"] += (uni_best == y)
            pb = bi_best.get(x[i], uni_best); ok["bi"] += (pb == y)
            pt = tri_best.get((x[i - 1], x[i]), pb) if i >= 1 else pb; ok["tri"] += (pt == y)
    return {k: v / n for k, v in ok.items()}

res = {"V": V, "train_tokens": n_train_tok, "test_tokens": n_test_tok, "N": N_RES}
res["ngram"] = ngram_eval()
print(f"[control] unigram {res['ngram']['uni']:.3f}  bigram {res['ngram']['bi']:.3f}  trigram+backoff {res['ngram']['tri']:.3f}", flush=True)

# ---------------------------------------------------------------- резервуар без градиента
class Reservoir:
    """Фиксированные случайные коды слов E, фиксированные случайные веса Win/Wres (спектральный радиус rho).
    Единственное, что «учится», — выход Wout, и он не подбирается шагами, а РЕШАЕТСЯ: Wout = (A+λI)⁻¹B,
    где A = Σ φφᵀ, B = Σ φ·onehot(y) — суммы, которые можно пополнять новыми предложениями в любой момент."""
    def __init__(self, V, d=128, N=1500, rho=0.9, in_scale=0.4, k=10, lam=1.0, seed=0):
        rng = np.random.default_rng(seed)
        self.E = (rng.choice([-1.0, 1.0], size=(V, d)) / np.sqrt(d)).astype(np.float32)
        self.Win = (rng.normal(0, 1, (d, N)) * in_scale).astype(np.float32)
        W = rng.normal(0, 1, (N, N)).astype(np.float32); W *= (rng.random((N, N)) < k / N)
        v = rng.normal(size=N).astype(np.float32)
        for _ in range(60): v = W @ v; v /= (np.linalg.norm(v) + 1e-9)
        W *= rho / (np.linalg.norm(W @ v) + 1e-9); self.Wres = W
        self.V, self.d, self.N, self.F, self.lam = V, d, N, N + d + 1, lam
        self.A = np.zeros((self.F, self.F), np.float64); self.B = np.zeros((V, self.F), np.float64); self.Wout = None
    def features(self, X):
        Bt, T = X.shape; h = np.zeros((Bt, self.N), np.float32); F = np.zeros((Bt, T, self.F), np.float32)
        for t in range(T):
            x = self.E[X[:, t]]; h = np.tanh(x @ self.Win + h @ self.Wres)
            F[:, t, :self.N] = h; F[:, t, self.N:self.N + self.d] = x; F[:, t, -1] = 1.0
        return F
    def accumulate(self, X, M, weight=1.0):
        Phi = self.features(X)[M]; y = X[:, 1:][M[:, :-1]] if False else np.roll(X, -1, axis=1)[M]
        self.A += weight * (Phi.T.astype(np.float64) @ Phi.astype(np.float64))
        oh = sp.csr_matrix((np.full(len(y), weight), (y, np.arange(len(y)))), shape=(self.V, len(y)))
        self.B += oh @ Phi.astype(np.float64)
        return len(y)
    def solve(self):
        A = self.A + self.lam * np.eye(self.F); self.Wout = np.linalg.solve(A, self.B.T).astype(np.float32)   # (F, V)
    def predict_eval(self, seqs, bs=256):
        ok1 = ok5 = n = 0
        for X, M in batches(seqs, bs):
            F = self.features(X); Y = np.roll(X, -1, axis=1)
            for t in range(X.shape[1] - 1):
                m = M[:, t]
                if not m.any(): continue
                logits = F[m, t] @ self.Wout; y = Y[m, t]
                ok1 += int((logits.argmax(1) == y).sum()); top5 = np.argpartition(-logits, 5, axis=1)[:, :5]
                ok5 += int((top5 == y[:, None]).any(1).sum()); n += len(y)
        return ok1 / n, ok5 / n
    def generate(self, prefix, max_new=12, temp=0.7, rng=None):
        rng = rng or np.random.default_rng(1); ids = list(prefix); h = np.zeros((1, self.N), np.float32)
        for i in range(len(ids) + max_new):
            x = self.E[[ids[i]]]; h = np.tanh(x @ self.Win + h @ self.Wres)
            if i < len(ids) - 1: continue
            phi = np.concatenate([h[0], x[0], [1.0]]); out = phi @ self.Wout; out = np.clip(out, 1e-4, None) ** (1 / temp); p = out / out.sum()
            nxt = int(rng.choice(self.V, p=p)); ids.append(nxt)
            if nxt == EOS: break
        return " ".join(itos[i] for i in ids[len(prefix):] if i != EOS)

t1 = time.time(); R = Reservoir(V, d=D_EMB, N=N_RES, lam=LAM); n = 0
for X, M in batches(train_ids, 512): n += R.accumulate(X, M)
t_feat = time.time() - t1; R.solve(); t_fit = time.time() - t1
print(f"[reservoir] N={N_RES} F={R.F}  accumulate {n} tokens {t_feat:.0f}s  solve {t_fit - t_feat:.1f}s  total fit {t_fit:.0f}s  (0 шагов градиента)", flush=True)
acc1, acc5 = R.predict_eval(test_ids); res["reservoir"] = {"top1": acc1, "top5": acc5, "fit_seconds": t_fit, "F": R.F}
print(f"[reservoir] TEST top-1 {acc1:.3f}  top-5 {acc5:.3f}", flush=True)
samples = [R.generate(enc(toks(p))[:-1], rng=np.random.default_rng(s)) for s, p in enumerate(["once upon a time", "the little dog", "she looked at the"])]
for p, s in zip(["once upon a time", "the little dog", "she looked at the"], samples): print(f"[sample] {p} → {s}")
res["samples"] = samples

# ---------------------------------------------------------------- B. one-shot: 20 новых предложений
NEW = ["the purple fox painted the old boat with honey .", "grandma hid the golden spoon under the red sofa .",
       "a tiny robot sang to the sleepy moon every night .", "the brave snail carried a blue flag to the top of the hill .",
       "my cousin keeps a pet cloud in a glass jar .", "the hungry piano ate three green apples for lunch .",
       "a silver dragon learned to bake bread with the baker .", "the clever duck opened the door with a feather .",
       "tom planted a candy tree in the kitchen .", "the wind borrowed my hat and gave it to the river .",
       "a shy mountain whispered secrets to the stars .", "the lazy clock refused to wake the sun .",
       "lily built a castle out of warm pancakes .", "the frog wore yellow boots to the party .",
       "a lonely sock searched the house for its brother .", "the teacher kept a dinosaur in the garden .",
       "max traded his bike for a talking fish .", "the rain painted the street with rainbows .",
       "a paper bird flew over the sleeping town .", "the baby elephant learned to whistle in a week ."]
new_ids = [enc(toks(s)) for s in NEW]
def new_acc(model_pred):
    """точность предсказания токенов новых предложений (позиции 3..конец, teacher forcing)"""
    ok = n = 0
    for x in new_ids:
        pred = model_pred(x)
        for i in range(3, len(x) - 1): ok += (pred[i] == x[i + 1]); n += 1
    return ok / n
def res_pred(x):
    X = np.array([x]); F = R.features(X)[0]; return (F @ R.Wout).argmax(1)
before_new = new_acc(res_pred); before_test = acc1
t2 = time.time()
for X, M in batches(new_ids, 32): R.accumulate(X, M, weight=float(ARGS.get("w", 30)))
R.solve(); t_one = time.time() - t2
after_new = new_acc(res_pred); after_test, _ = R.predict_eval(test_ids)
res["oneshot"] = {"new_before": before_new, "new_after": after_new, "test_before": before_test, "test_after": after_test, "seconds": t_one}
print(f"[one-shot] новые предложения: до {before_new:.2f} → после {after_new:.2f} (одно обновление, {t_one:.1f}s); тест {before_test:.3f} → {after_test:.3f}", flush=True)

# ---------------------------------------------------------------- контроль: трансформер (градиент), то же время CPU
from mini_ai.transformer import init_params, n_params, loss_and_grads, AdamW, clip_grads, forward
budget = float(TF_SECONDS) if TF_SECONDS else t_fit
p, cfg = init_params(V, d=128, n_layer=2, n_head=4, pos="rope", seed=0); opt = AdamW(p, lr=3e-3, wd=0.01)
print(f"[transformer] params {n_params(p):,}  budget {budget:.0f}s", flush=True)
flat = np.concatenate([np.array(x[1:]) for x in train_ids]); T_BLK, BS = 64, 16
def tf_eval(n_tok=20000):
    ft = np.concatenate([np.array(x[1:]) for x in test_ids])[:n_tok + 1]; ok = n = 0
    for b in range(0, len(ft) - T_BLK - 1, T_BLK * BS):
        rows = [ft[b + r * T_BLK: b + r * T_BLK + T_BLK + 1] for r in range(BS) if b + r * T_BLK + T_BLK + 1 <= len(ft)]
        if not rows: break
        Xb = np.stack(rows); logits, _ = forward(p, cfg, Xb[:, :-1]); logits = np.asarray(logits)
        ok += int((logits.argmax(-1) == Xb[:, 1:]).sum()); n += Xb[:, 1:].size
    return ok / n
tf_curve = []; t3 = time.time(); step = 0
while time.time() - t3 < budget:
    starts = RNG.integers(0, len(flat) - T_BLK - 1, BS); Xb = np.stack([flat[s:s + T_BLK + 1] for s in starts])
    loss, grads = loss_and_grads(p, cfg, Xb[:, :-1], Xb[:, 1:], np.ones((BS, T_BLK))); clip_grads(grads, 1.0); opt.step(p, grads, 3e-3); step += 1
    if step % 100 == 0:
        a = tf_eval(8000); tf_curve.append((round(time.time() - t3), step, round(float(loss), 3), a)); print(f"[transformer] {time.time()-t3:.0f}s step {step} loss {loss:.3f} test top-1 {a:.3f}", flush=True)
tf_acc = tf_eval(); res["transformer"] = {"top1": tf_acc, "steps": step, "seconds": time.time() - t3, "curve": tf_curve, "params": n_params(p)}
print(f"[transformer] TEST top-1 {tf_acc:.3f} after {step} steps / {time.time()-t3:.0f}s", flush=True)
# one-shot для трансформера: столько же секунд дообучения на 20 новых предложениях
def tf_pred(x):
    logits, _ = forward(p, cfg, np.array([x])); return np.asarray(logits)[0].argmax(-1)
tb = new_acc(tf_pred); t4 = time.time(); k = 0
Xn = np.full((len(new_ids), max(len(x) for x in new_ids)), EOS); Mn = np.zeros(Xn.shape[:1] + (Xn.shape[1] - 1,))
for i, x in enumerate(new_ids): Xn[i, :len(x)] = x; Mn[i, :len(x) - 1] = 1
while time.time() - t4 < max(t_one, 1.0):
    loss, grads = loss_and_grads(p, cfg, Xn[:, :-1], Xn[:, 1:], Mn); clip_grads(grads, 1.0); opt.step(p, grads, 3e-3); k += 1
ta = new_acc(tf_pred); tt = tf_eval()
res["transformer"]["oneshot"] = {"new_before": tb, "new_after": ta, "test_after": tt, "steps": k}
print(f"[transformer one-shot] новые: до {tb:.2f} → после {ta:.2f} ({k} шагов за {time.time()-t4:.1f}s); тест {tf_acc:.3f} → {tt:.3f}", flush=True)
json.dump(res, open(OUT, "w"), indent=1, default=float)
print(f"[done] {time.time()-t0:.0f}s → {OUT}")
