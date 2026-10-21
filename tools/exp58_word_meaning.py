"""exp58 — смысл слова строится САМ из контекстов (без градиента), для каждого нового слова сразу.

Гипотеза Фархода: код слова = не случайный шум сита, а сумма Σ (роль-позиции ⊛ код соседа) по всем
контекстам, где слово встретилось («ДНК слова» = слияние его окружений). Тогда у слов с похожим
употреблением коды сближаются САМИ: city≈town, river≈lake, king≈queen, red≈blue — никто не говорил,
что они похожи. Новое слово получает смысл с первого предложения и уточняет с каждым следующим.

Механизм (локальный, один проход, никакого backprop): базовые коды слов — случайные ±1 (D=4096);
контекст слова w в предложении = Σ_{i=-k..k, i≠0} perm_i(base(c_i)) (позиция = фиксированная перестановка);
sem(w) = Σ по всем вхождениям контекстов → знак. Это HDC-аналог word2vec (random indexing), но инкрементальный.
Данные: Simple Wikipedia text.jsonl (первые 300k предложений). Тест ДО кода: 40 пар «похожие слова» и
40 пар «случайные слова»; метрика — доля похожих пар, у которых косинус выше медианы случайных пар
(ожидание для шума 0.5; успех ≥0.85). Второй тест — соседи: для 12 слов печатаем 5 ближайших без отбора.
Третий — скорость: новое слово (вымышленное «glorb» = заменяем им «city» в 20 предложениях) →
его ближайший сосед должен стать town/city после 20 предложений (успех: в топ-5).
"""
import json, os, re, time, collections, numpy as np
rng = np.random.default_rng(0)
SRC = os.path.expanduser("~/Developer/mini-ai-runs/memory-core/simplewiki_full/text.jsonl")
OUT = os.path.expanduser("~/Developer/mini-ai-runs/learning-core")
D = 4096; K = 3; LIMIT = 150000
STOP = set("a an the of in on at to for from by with and or as is was are were be been it its this that these those he she they his her their has have had do does did not also which who whom what when where there here than then".split())
def toks(s): return re.findall(r"[a-z]+", s.lower())

class Meaning:
    def __init__(self):
        self.base = {}; self.sem = {}; self.n = collections.Counter(); self.perm = {i: rng.permutation(D) for i in range(-K, K + 1) if i}
    def b(self, w):
        if w not in self.base: self.base[w] = rng.choice([-1.0, 1.0], size=D).astype(np.float32)
        return self.base[w]
    def read(self, sent):
        ws = [w for w in toks(sent) if w not in STOP]
        if len(ws) < 2: return
        B = [self.b(w) for w in ws]
        for j, w in enumerate(ws):
            ctx = np.zeros(D, np.float32)
            for i in range(-K, K + 1):
                if i and 0 <= j + i < len(ws): ctx += B[j + i][self.perm[i]]
            self.sem[w] = self.sem.get(w, np.zeros(D, np.float32)) + ctx; self.n[w] += 1
    def vec(self, w):
        v = self.sem.get(w); return None if v is None else v / (np.linalg.norm(v) + 1e-9)
    def cos(self, a, b):
        va, vb = self.vec(a), self.vec(b); return None if va is None or vb is None else float(va @ vb)
    def neighbors(self, w, k=5, min_n=30):
        vw = self.vec(w)
        if vw is None: return []
        words = [x for x in self.sem if self.n[x] >= min_n and x != w]
        M = np.stack([self.vec(x) for x in words]); s = M @ vw; idx = np.argsort(-s)[:k]
        return [(words[i], round(float(s[i]), 3)) for i in idx]

SIMILAR = [("city", "town"), ("river", "lake"), ("king", "queen"), ("red", "blue"), ("car", "truck"), ("dog", "cat"), ("january", "february"),
           ("monday", "friday"), ("france", "germany"), ("actor", "actress"), ("mother", "father"), ("mountain", "hill"), ("book", "novel"),
           ("singer", "musician"), ("north", "south"), ("big", "large"), ("small", "little"), ("film", "movie"), ("war", "battle"), ("school", "university"),
           ("football", "basketball"), ("gold", "silver"), ("water", "oil"), ("island", "peninsula"), ("church", "cathedral"), ("village", "town"),
           ("president", "minister"), ("born", "died"), ("english", "french"), ("two", "three"), ("million", "thousand"), ("son", "daughter"),
           ("wife", "husband"), ("sea", "ocean"), ("album", "song"), ("game", "sport"), ("doctor", "physician"), ("chemical", "compound"), ("planet", "star"), ("america", "europe")]

if __name__ == "__main__":
    t0 = time.time(); M = Meaning(); n = 0
    with open(SRC) as f:
        for line in f:
            d = json.loads(line); s = d[0] if isinstance(d, list) else (d.get("text", "") if isinstance(d, dict) else str(d))
            M.read(s); n += 1
            if n >= LIMIT: break
    print(f"прочитано {n} предложений, слов со смыслом {len(M.sem)}, {time.time()-t0:.0f}s")
    # тест 1: похожие vs случайные
    vocab = [w for w in M.sem if M.n[w] >= 30]; rand_pairs = [(vocab[i], vocab[j]) for i, j in rng.integers(0, len(vocab), size=(400, 2)) if i != j]
    rc = [M.cos(a, b) for a, b in rand_pairs]; med = float(np.median(rc)); p90 = float(np.percentile(rc, 90))
    sims = [(a, b, M.cos(a, b)) for a, b in SIMILAR if M.cos(a, b) is not None]
    above = sum(c > med for _, _, c in sims); above90 = sum(c > p90 for _, _, c in sims)
    res = dict(sentences=n, words=len(M.sem), similar_above_median=round(above / len(sims), 3), similar_above_p90=round(above90 / len(sims), 3), random_median=round(med, 3), random_p90=round(p90, 3))
    print(f"похожие пары выше медианы случайных: {above}/{len(sims)} = {res['similar_above_median']:.2f} (шум: 0.50); выше 90-го перцентиля: {above90}/{len(sims)} = {res['similar_above_p90']:.2f} (шум: 0.10)")
    print("  примеры: " + ", ".join(f"{a}~{b} {c:.2f}" for a, b, c in sims[:12]) + f" | случайные медиана {med:.2f}")
    # тест 2: соседи
    print("соседи (без отбора):")
    for w in ["city", "river", "king", "red", "january", "france", "actor", "mother", "football", "gold", "born", "album"]:
        print(f"  {w}: " + ", ".join(f"{x} {c}" for x, c in M.neighbors(w)))
    # тест 3: новое слово
    cnt = 0; ok_at = None
    with open(SRC) as f:
        for line in f:
            d = json.loads(line); s = d[0] if isinstance(d, list) else (d.get("text", "") if isinstance(d, dict) else str(d))
            if re.search(r"\bcity\b", s.lower()) and cnt < 30:
                M.read(re.sub(r"\bcity\b", "glorb", s, flags=re.I)); cnt += 1
                if cnt in (1, 3, 5, 10, 20, 30):
                    nb = M.neighbors("glorb", k=5, min_n=30); print(f"  новое слово «glorb» после {cnt} предл.: " + ", ".join(f"{x} {c}" for x, c in nb))
                    if ok_at is None and any(x in ("city", "town", "capital", "village") for x, _ in nb): ok_at = cnt
            if cnt >= 30: break
    res["new_word_hit_after"] = ok_at; print(f"«glorb» ≈ city/town в топ-5 после {ok_at} предложений")
    res["seconds"] = round(time.time() - t0, 1); json.dump(res, open(os.path.join(OUT, "exp58_word_meaning.json"), "w"), indent=1); print(res)
