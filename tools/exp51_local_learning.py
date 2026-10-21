"""exp51 — обучение интентов чата ЛОКАЛЬНЫМ правилом (без backprop).

Гипотеза: HDC-прототипы (класс = bundle кодов примеров; предсказание = ближайший прототип;
дообучение = прибавить ошибочный пример к своему классу и вычесть из чужого — перцептронное
правило, локальное, без градиента по сети) дают ≥0.80 на том же тесте, где MLP (градиент) даёт 0.88.
И главное, чего MLP не умеет: новый класс из 5 примеров онлайн, без переобучения — ≥0.6 на его тесте,
старые классы не падают больше чем на 0.02.

Признаки: те же, что у MLP (mini_ai.features.vectorize: мешок слов + символьные 3-граммы, 2048 хеш-ячеек),
проецированные случайной ±1 матрицей в D=10000 биполярных кодов (сито).
Контроль: 1-NN по сырым признакам (косинус), случайный класс (1/28), частый класс.
"""
import json, random, sys, time, os
import numpy as np
sys.path.insert(0, os.path.expanduser("~/Developer/mini-ai"))
from mini_ai.features import normalize, vectorize
from train import split, DIM

ROOT = os.path.expanduser("~/Developer/mini-ai"); OUT = os.path.expanduser("~/Developer/mini-ai-runs/learning-core")
rng = np.random.default_rng(0); D = 10000

def feats(phrases):
    X = np.stack([vectorize(p, DIM) for p in phrases]).astype(np.float32)
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9); return X

class HDC:
    """прототипы классов в биполярном пространстве; обучение — локальное правило"""
    def __init__(self, K, dim=D):
        self.P = rng.choice([-1.0, 1.0], size=(DIM, dim)).astype(np.float32)   # сито: случайная проекция, фиксирована
        self.W = np.zeros((K, dim), dtype=np.float32); self.n = np.zeros(K)
    def enc(self, X): return np.sign(X @ self.P + 1e-9)                         # ±1 код фразы
    def fit_bundle(self, X, y):
        H = self.enc(X)
        for h, k in zip(H, y): self.W[k] += h; self.n[k] += 1
    def predict(self, X):
        H = self.enc(X); Wn = self.W / (np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-9)
        return np.argmax(H @ Wn.T, axis=1)
    def refine(self, X, y, epochs=10, lr=1.0):
        """локальная коррекция: ошибся → пример к своему классу, минус из предсказанного (перцептрон на прототипах)"""
        H = self.enc(X); ep = 0
        for ep in range(epochs):
            wrong = 0
            for h, k in zip(H, y):
                Wn = self.W / (np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-9)
                p = int(np.argmax(Wn @ h))
                if p != k: self.W[k] += lr * h; self.W[p] -= lr * h; wrong += 1
            if wrong == 0: break
        return ep + 1
    def add_class(self, X, k):
        """новый класс онлайн: просто bundle его примеров (один проход, без переобучения остального)"""
        if k >= len(self.W):
            self.W = np.vstack([self.W, np.zeros((k + 1 - len(self.W), self.W.shape[1]), np.float32)])
        H = self.enc(X); self.W[k] += H.sum(0)

def acc(pred, y): return float(np.mean(np.asarray(pred) == np.asarray(y)))

if __name__ == "__main__":
    t0 = time.time()
    data = json.load(open(os.path.join(ROOT, "data/intents.json"), encoding="utf-8")); intents = data["intents"]
    tags = [it["tag"] for it in intents]; K = len(tags)
    tr, te = split(intents, random.Random(42))
    Xtr, ytr = feats([p for p, _ in tr]), np.array([k for _, k in tr]); Xte, yte = feats([p for p, _ in te]), np.array([k for _, k in te])
    print(f"train {len(tr)} test {len(te)} classes {K}")
    res = {}
    # контроли
    res["random"] = 1 / K; res["majority"] = acc(np.full(len(yte), np.bincount(ytr).argmax()), yte)
    nn = np.argmax(Xte @ Xtr.T, axis=1); res["1nn_raw"] = acc(ytr[nn], yte)
    # HDC bundle (один проход)
    t = time.time(); m = HDC(K); m.fit_bundle(Xtr, ytr); res["hdc_bundle"] = acc(m.predict(Xte), yte); res["hdc_bundle_sec"] = round(time.time() - t, 2)
    # + локальная коррекция
    t = time.time(); eps = m.refine(Xtr, ytr, epochs=15); res["hdc_refined"] = acc(m.predict(Xte), yte); res["hdc_refined_sec"] = round(time.time() - t, 2); res["hdc_epochs"] = eps
    res["hdc_params"] = int(m.W.size)
    print(f"контроль: random {res['random']:.3f} | majority {res['majority']:.3f} | 1-NN сырые {res['1nn_raw']:.3f}")
    print(f"HDC bundle (1 проход, {res['hdc_bundle_sec']}s): {res['hdc_bundle']:.3f}")
    print(f"HDC + локальная коррекция ({eps} эп., {res['hdc_refined_sec']}s, {res['hdc_params']} чисел): {res['hdc_refined']:.3f}   [MLP-контроль из README: 0.88]")

    # ОНЛАЙН: убираем один класс из обучения, учим, потом добавляем его 5 примерами
    scores_new, scores_old_before, scores_old_after = [], [], []
    for hold in range(K):
        keep = ytr != hold; m2 = HDC(K); m2.fit_bundle(Xtr[keep], ytr[keep]); m2.refine(Xtr[keep], ytr[keep], epochs=10)
        old = yte != hold; scores_old_before.append(acc(m2.predict(Xte[old]), yte[old]))
        idx = np.where(~keep)[0][:5]; m2.add_class(Xtr[idx], hold)
        scores_old_after.append(acc(m2.predict(Xte[old]), yte[old])); scores_new.append(acc(m2.predict(Xte[~old]), yte[~old]))
    res["online_new_class_acc"] = float(np.mean(scores_new)); res["online_old_before"] = float(np.mean(scores_old_before)); res["online_old_after"] = float(np.mean(scores_old_after))
    print(f"ОНЛАЙН новый класс из 5 примеров (среднее по {K} классам): новый {res['online_new_class_acc']:.3f} | старые до {res['online_old_before']:.3f} → после {res['online_old_after']:.3f}")
    res["seconds"] = round(time.time() - t0, 1); json.dump(res, open(os.path.join(OUT, "exp51_local_learning.json"), "w"), indent=1); print("sec", res["seconds"])
