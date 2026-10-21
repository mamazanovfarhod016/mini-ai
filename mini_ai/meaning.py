"""meaning.py — смысл слова строится САМ из контекстов (exp58, идея Фархода), для каждого нового слова сразу.

Без градиента: базовые коды слов — случайные ±1 (D); контекст слова в предложении = Σ perm_i(base(сосед_i)),
позиция = фиксированная перестановка; sem(w) = Σ контекстов по всем вхождениям. Слова, употребляемые одинаково,
получают близкие коды сами: red≈blue, actor≈actress, january≈march (exp58: 40/40 пар выше 90-го перцентиля шума).
Новое слово понимается с 1-го предложения и уточняется с каждым следующим.

Что даёт ядру:
  • similar(a, b) — похожи ли слова по смыслу (для рамок вопросов: biggest≈largest, town≈city);
  • neighbors(w) — что она «думает» о слове;
  • kind(w) — к какому известному классу слов ближе (continent/country/city/person/number…) — прототипы классов
    = сумма кодов опорных слов класса (опорные слова — закрытый короткий список, не обучение);
  • explain(w) — честная сводка: сколько раз видела, на что похоже, уверенность.
Хранится в state_dir/meaning.npz (инкрементально).
"""
import os, re, collections, numpy as np

D = 4096; K = 3
BASE_NPZ = os.path.expanduser("~/Developer/mini-ai-runs/learning-core/meaning_base.npz")   # базовый смысл слов из Simple Wikipedia (tools/build_meaning.py)
STOP = set("a an the of in on at to for from by with and or as is was are were be been it its this that these those he she they his her their has have had do does did not also which who whom what when where there here than then".split())
SEEDS = {"continent": ["europe", "asia", "africa", "america", "oceania", "antarctica"], "country": ["france", "germany", "russia", "china", "japan", "italy", "spain", "india"],
         "city": ["paris", "london", "berlin", "moscow", "tokyo", "rome", "madrid"], "person": ["actor", "writer", "singer", "scientist", "king", "president", "player", "doctor", "teacher", "student"],
         "number": ["two", "three", "four", "five", "ten", "hundred", "thousand"], "color": ["red", "blue", "green", "yellow", "white", "black"],
         "month": ["january", "february", "march", "april", "june", "july"], "animal": ["dog", "cat", "horse", "bird", "fish", "lion"],
         "water": ["river", "lake", "sea", "ocean", "bay"], "organization": ["company", "university", "union", "party", "club", "league"]}

def toks(s): return re.findall(r"[a-zа-яё]+", s.lower())

class Meaning:
    def __init__(self, state_dir=None):
        self.rng = np.random.default_rng(7); self.base = {}; self.sem = {}; self.n = collections.Counter()
        self.perm = {i: self.rng.permutation(D) for i in range(-K, K + 1) if i}
        self.path = os.path.join(state_dir, "meaning.npz") if state_dir else None; self._dirty = 0; self._delta = set()
        if state_dir and os.path.exists(BASE_NPZ): self.load(BASE_NPZ)
        if self.path and os.path.exists(self.path): self.load(); self._delta = set()
        self._has_base = bool(state_dir and os.path.exists(BASE_NPZ)); self.defs = {}
    def b(self, w):
        if w not in self.base: self.base[w] = np.random.default_rng(abs(hash(w)) % (2**32)).choice([-1, 1], size=D).astype(np.int8)
        return self.base[w]
    def read(self, text):
        n = 0
        for sent in re.split(r"(?<=[.!?])\s+", text):
            ws = [w for w in toks(sent) if w not in STOP]
            if len(ws) < 2: continue
            # ОПРЕДЕЛЕНИЕ: «X is a/an <class> …» / «X is the <class> of …» → X получает смысл класса сразу (вес = 8 контекстов)
            md = re.match(r"^(?:the\s+)?([a-z][\w\-]*(?:\s+[a-z][\w\-]*)?)\s+(?:is|are|was|were)\s+(?:a|an|the)\s+(?:[a-z\-]+\s+){0,2}?([a-z]+)\b", sent.lower().strip())
            if md:
                x, cls = md.group(1).split()[-1], md.group(2)
                if x not in STOP and cls not in STOP and cls != x and cls in self.sem and self.n[cls] >= 20:
                    cv = self.sem[cls].astype(np.float32)
                    self.sem[x] = self.sem.get(x, np.zeros(D, np.float32)).astype(np.float32) + 8.0 * cv / (np.linalg.norm(cv) + 1e-9) * np.sqrt(D) / 4
                    self.n[x] += 1; self._delta.add(x); self.defs[x] = cls
            B = [self.b(w) for w in ws]
            for j, w in enumerate(ws):
                ctx = np.zeros(D, np.float32)
                for i in range(-K, K + 1):
                    if i and 0 <= j + i < len(ws): ctx += B[j + i][self.perm[i]]
                v = self.sem.get(w); self.sem[w] = (ctx if v is None else v.astype(np.float32) + ctx); self.n[w] += 1; n += 1; self._delta.add(w)
        self._dirty += n; return n
    def vec(self, w):
        v = self.sem.get(w.lower())
        if v is None: return None
        v = v.astype(np.float32); return v / (np.linalg.norm(v) + 1e-9)
    def sent_vec(self, words):
        """код фразы = сумма смыслов её слов (idf-веса не нужны: редкие слова и так далеки от всех)"""
        vs = [self.vec(w) for w in words]; vs = [v for v in vs if v is not None]
        if not vs: return None
        v = np.sum(vs, axis=0); return v / (np.linalg.norm(v) + 1e-9)
    def phrase_sim(self, words_a, words_b):
        a, b = self.sent_vec(words_a), self.sent_vec(words_b)
        return None if a is None or b is None else float(a @ b)
    def similar(self, a, b):
        va, vb = self.vec(a), self.vec(b); return None if va is None or vb is None else float(va @ vb)
    def neighbors(self, w, k=5, min_n=20):
        vw = self.vec(w)
        if vw is None: return []
        words = [x for x in self.sem if self.n[x] >= min_n and x != w.lower()]
        if not words: return []
        M = np.stack([self.vec(x) for x in words]); s = M @ vw; idx = np.argsort(-s)[:k]
        return [(words[i], round(float(s[i]), 3)) for i in idx]
    def kind(self, w):
        """класс слова по близости к прототипам из опорных слов (только тех, что она уже видела)"""
        vw = self.vec(w)
        if vw is None: return None, 0.0
        d = self.defs.get(w.lower())
        if d:                                                        # класс по определению: doctor → kind(doctor) = person; continent → continent
            if d in SEEDS: return d, 0.9
            for k, seeds in SEEDS.items():
                if d in seeds: return k, 0.9
            kd, sd = self._kind_ctx(d)
            if kd and sd > 0.2: return kd, round(sd, 3)
        return self._kind_ctx(w)
    def _kind_ctx(self, w):
        vw = self.vec(w)
        if vw is None: return None, 0.0
        best, bs = None, 0.0
        for k, seeds in SEEDS.items():
            vs = [self.vec(s) for s in seeds if self.vec(s) is not None and s != w.lower()]
            if len(vs) < 2: continue
            p = np.sum(vs, axis=0); p /= np.linalg.norm(p) + 1e-9; sc = float(vw @ p)
            if sc > bs: best, bs = k, sc
        return best, round(bs, 3)
    def explain(self, w):
        w = w.lower(); n = self.n.get(w, 0)
        if not n: return f"I have never seen the word «{w}»."
        k, sc = self.kind(w); nb = self.neighbors(w)
        conf = "sure" if n >= 20 and sc > 0.25 else ("fairly sure" if n >= 5 else "not sure yet")
        d = self.defs.get(w)
        if d: conf = "sure" if n >= 3 else "fairly sure"
        return f"«{w}»: seen {n} time{'s' if n != 1 else ''}; " + (f"you told me it is a {d}; " if d else "") + "used like " + ", ".join(x for x, _ in nb[:4]) + (f"; looks like a {k} ({sc})" if k else "") + f" — I'm {conf}."
    def save(self, force=False):
        """в state_dir — только слова, выученные сверх базы (дельта), чтобы не писать сотни МБ на каждый learn"""
        if not self.path or (not self._dirty and not force): return
        words = [w for w in self.sem if w in self._delta] if self._has_base else list(self.sem)
        if not words: self._dirty = 0; return
        np.savez_compressed(self.path, words=np.array(words), sem=np.stack([self.sem[w] for w in words]).astype(np.float16), n=np.array([self.n[w] for w in words])); self._dirty = 0
    def load(self, path=None):
        z = np.load(path or self.path, allow_pickle=False)
        for w, v, n in zip(z["words"], z["sem"], z["n"]):
            w = str(w); prev = self.sem.get(w)
            self.sem[w] = v.astype(np.float16) if prev is None else (prev.astype(np.float32) + v.astype(np.float32)).astype(np.float16); self.n[w] += int(n)   # float16 в памяти: 68k слов = 0.55 ГБ вместо 1.1
