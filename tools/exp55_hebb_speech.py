"""exp55 — свободная речь без градиента: Хебб-память «контекст → следующее слово».

Задача: собеседник, который САМ строит ответ по словам (не выбирает из шаблонов).
Механизм: сито даёт код каждому слову (±1, D). Контекст = Σ perm^i(code(w_{t-i})) — позиционное
связывание последних k слов + код реплики пользователя (мешок). Память W[слово] += code(контекст)
— чистое правило Хебба, один проход по данным, без backprop. Генерация: следующее слово =
argmax_w <W[w], контекст> (с штрафом на повтор), пока не <eos>.

Данные: data/dialogs/ru.jsonl (600 диалогов, 3150 реплик), ответы бота. Train 80% / test 20% диалогов.
Числа (ДО запуска): перплексия-заменитель = доля правильно угаданных следующих слов на тесте (teacher forcing);
контроль — биграммная таблица (самое частое следующее слово) и униграмма. Успех: Хебб ≥ биграмм на тесте
(значит, код контекста ОБОБЩАЕТ, а не просто запоминает), и ≥10 сгенерированных ответов, которые
читаются как связная русская речь и отвечают на реплику (показать как есть, без отбора).
"""
import json, os, re, time, collections, numpy as np
rng = np.random.default_rng(0)
ROOT = os.path.expanduser("~/Developer/mini-ai"); OUT = os.path.expanduser("~/Developer/mini-ai-runs/learning-core")
D = 4096; K = 4
def toks(s): return re.findall(r"[а-яёa-z0-9]+|[,.!?—:;]", s.lower())

class Speaker:
    def __init__(self):
        self.W = {}; self.code = {}; self.perm = [rng.permutation(D) for _ in range(K)]; self.uni = collections.Counter(); self.bi = collections.defaultdict(collections.Counter)
    def c(self, w):
        if w not in self.code: self.code[w] = rng.choice([-1.0, 1.0], size=D).astype(np.float32)
        return self.code[w]
    def ctx(self, prev, user):
        v = np.zeros(D, np.float32)
        for i, w in enumerate(reversed(prev[-K:])): v += self.c(w)[self.perm[i]] * (1.0 if i == 0 else 0.7)   # ближнее слово весомее
        u = sum((self.c("u:" + w) for w in user), np.zeros(D, np.float32))
        if len(user): v += 0.6 * u / np.sqrt(len(user))
        return v
    def learn(self, user, reply):
        ws = reply + ["<eos>"]; prev = ["<bos>"]
        for w in ws:
            x = self.ctx(prev, user)
            self.W[w] = self.W.get(w, np.zeros(D, np.float32)) + x                      # Хебб
            self.uni[w] += 1; self.bi[prev[-1]][w] += 1; prev.append(w)
    def finalize(self):
        self.words = list(self.W); M = np.stack([self.W[w] for w in self.words])
        self.M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    def scores(self, prev, user):
        x = self.ctx(prev, user); return self.M @ (x / (np.linalg.norm(x) + 1e-9))
    def generate(self, user, max_len=18):
        prev = ["<bos>"]; out = []; used = collections.Counter()
        for _ in range(max_len):
            s = self.scores(prev, user)
            for w, n in used.items(): s[self.words.index(w)] -= 0.15 * n
            w = self.words[int(np.argmax(s))]
            if w == "<eos>": break
            out.append(w); used[w] += 1; prev.append(w)
        return out

def detok(ws): return re.sub(r"\s+([,.!?:;])", r"\1", " ".join(ws))

if __name__ == "__main__":
    t0 = time.time()
    dl = [json.loads(l) for l in open(os.path.join(ROOT, "data/dialogs/ru.jsonl"), encoding="utf-8")]; rng.shuffle(dl)
    n_te = len(dl) // 5; te, tr = dl[:n_te], dl[n_te:]
    def pairs(ds):
        for d in ds:
            t = d["turns"]
            for i in range(1, len(t)):
                if t[i]["role"] == "bot" and t[i - 1]["role"] == "user": yield toks(t[i - 1]["text"]), toks(t[i]["text"])
    S = Speaker()
    for u, r in pairs(tr): S.learn(u, r)
    S.finalize(); print(f"train пар {sum(1 for _ in pairs(tr))}, слов в памяти {len(S.words)}, {time.time()-t0:.1f}s")
    # оценка: следующее слово (teacher forcing)
    hit = hit_bi = hit_uni = n = 0; top_uni = S.uni.most_common(1)[0][0]
    for u, r in pairs(te):
        prev = ["<bos>"]
        for w in r + ["<eos>"]:
            s = S.scores(prev, u); hit += S.words[int(np.argmax(s))] == w
            hit_bi += (S.bi[prev[-1]].most_common(1)[0][0] if S.bi[prev[-1]] else top_uni) == w
            hit_uni += top_uni == w; n += 1; prev.append(w)
    res = dict(hebb=round(hit / n, 3), bigram=round(hit_bi / n, 3), unigram=round(hit_uni / n, 3), n=n)
    print(f"следующее слово на тесте: Хебб {res['hebb']:.3f} | биграмма {res['bigram']:.3f} | униграмма {res['unigram']:.3f}  (n={n})")
    # генерация — первые 12 тестовых реплик, без отбора
    print("\nгенерация (тест, без отбора):"); ex = []
    for i, (u, r) in enumerate(pairs(te)):
        if i >= 12: break
        g = detok(S.generate(u)); ex.append((detok(u), g, detok(r))); print(f"  Ю: {detok(u)}\n  ИИ: {g}\n  (эталон: {detok(r)})")
    res["examples"] = ex; res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(OUT, "exp55_hebb_speech.json"), "w"), ensure_ascii=False, indent=1); print("sec", res["seconds"])
