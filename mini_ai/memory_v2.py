"""Mini AI — ядро памяти v2 как библиотека (21.10.2026).

ВСЁ без градиента. Три части (ARCHITECT.md §2, ARCHITECT_v2.md):
  Sieve     — слово ↔ канонический код C(w) (случайный фиксированный, cos≈0 между словами) + IDF-вес
              (exp20: редкое слово информативнее) + cleanup (вектор → ближайшее известное слово).
  Working   — рабочая память: ОДИН вектор S, факты = суперпозиция bind(X, ρ, Y) (exp22, ~50 фактов).
  LongTerm  — долговременная память: список записей (exp4/23: 100% на 1000, 99.9% на 4000, не забывает).
              Два вида записей: ФАКТЫ (ключ=bind(X,ρ), значение=C(Y)) и ПРЕДЛОЖЕНИЯ (ключ=IDF-мешок слов,
              значение=текст) — чтобы читать сырой текст из интернета (exp20) и отвечать по нему.
Сохранение на диск (npz + json): между запусками ничего не теряется — требование «не забывает».
"""
import os, re, json, zlib, time
import numpy as np

D = 2048
_C, _FC = {}, {}
C_CACHE_MAX = 60000          # exp33: 150k слов × (float+complex 32 КБ) = 5 ГБ → OOM. Кэш ограничен, лишнее пересчитывается.


def unit(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def C(w):
    if w not in _C:
        if len(_C) >= C_CACHE_MAX: _C.clear(); _FC.clear()
        v = unit(np.random.default_rng(zlib.crc32(w.encode()) & 0xffffffff).standard_normal(D))
        _C[w] = v
    return _C[w]


def FC(w):
    if w not in _FC:
        if len(_FC) >= C_CACHE_MAX: _FC.clear()
        _FC[w] = np.fft.fft(C(w))
    return _FC[w]


def bind(*ws):
    z = np.ones(D, dtype=complex)
    for w in ws: z = z * (FC(w) if isinstance(w, str) else np.fft.fft(w))
    return np.real(np.fft.ifft(z))


def unbind(s, *ws):
    z = np.fft.fft(s)
    for w in ws: z = z * np.conj(FC(w) if isinstance(w, str) else np.fft.fft(w))
    return np.real(np.fft.ifft(z))


STOP = set("the a an to of is are was were in on at as by for with from and or but that this which who whom "
           "it its he she they their his her them not no into over under between after before during has have "
           "had be been being will would can could may might also than then there these those such about one "
           "two three first second most many some all when where what how did do does there back".split())


def toks(s): return re.findall(r"[a-zA-Z0-9']+", s.lower())
def content_words(ws): return [w for w in ws if w not in STOP and len(w) > 1]


class Sieve:
    """Словарь + IDF + cleanup. Растёт сам: новое слово = новая строка."""
    def __init__(self):
        self.words, self.df, self.n_docs = [], {}, 0
        self._M = None

    def add(self, *ws):
        new = [w for w in dict.fromkeys(ws) if w not in self.df]
        for w in new: self.df[w] = 0; self.words.append(w)
        if new: self._M = None

    def count_doc(self, ws):
        self.n_docs += 1
        for w in set(ws): self.df[w] = self.df.get(w, 0) + 1

    def idf(self, w): return float(np.log((1 + self.n_docs) / (1 + self.df.get(w, 0))) + 1.0)

    def M(self):
        if self._M is None: self._M = np.stack([C(w) for w in self.words]) if self.words else np.zeros((0, D))
        return self._M

    def cleanup(self, v, exclude=()):
        if not self.words: return None, 0.0
        s = self.M() @ unit(v); order = np.argsort(-s)
        for i in order:
            if self.words[i] not in exclude:
                gap = float(s[i] - (s[order[1]] if len(order) > 1 else 0.0))
                return self.words[i], gap
        return None, 0.0

    def bag(self, ws):
        """IDF-взвешенный мешок канонических кодов — ключ предложения (exp20)."""
        cw = content_words(ws)
        if not cw: return np.zeros(D)
        return unit(sum(self.idf(w) * C(w) for w in cw))


class Working:
    """Рабочая память — один вектор (exp22). Ёмкость ~50 фактов при D=2048."""
    def __init__(self): self.S = np.zeros(D); self.n = 0
    def add(self, x, role, y): self.S = self.S + bind(x, role, y); self.n += 1
    def remove(self, x, role, y): self.S = self.S - bind(x, role, y); self.n -= 1
    def query(self, x, role): return unbind(self.S, x, role)


class LongTerm:
    """Долговременная память: список. Факты и предложения. Никогда не удаляет (не забывает)."""
    def __init__(self):
        self.fk, self.fv, self.fmeta = [], [], []        # факты: ключ bind(X,ρ) → значение C(Y)
        self.sk, self.stext, self.smeta = [], [], []     # предложения: ключ IDF-мешок → текст
        self._FK = self._SK = None

    def add_fact(self, x, role, y, meta=None):
        self.fk.append(bind(x, role)); self.fv.append(C(y)); self.fmeta.append(dict(x=x, role=role, y=y, **(meta or {}))); self._FK = None

    def add_sentence(self, key_vec, text, meta=None):
        self.sk.append(key_vec); self.stext.append(text); self.smeta.append(meta or {}); self._SK = None

    def query_fact(self, x, role):
        if not self.fk: return None
        if self._FK is None: self._FK = np.stack(self.fk)
        s = self._FK @ bind(x, role); top = s.max()
        cands = np.where(s >= top - 1e-6)[0]
        return self.fv[int(cands[-1])] if top > 0.3 else None     # recency при равных ключах

    def search_sentences(self, qvec, topk=3):
        if not self.sk: return []
        if self._SK is None: self._SK = np.stack(self.sk)
        s = self._SK @ qvec; order = np.argsort(-s)[:topk]
        return [(self.stext[i], float(s[i]), self.smeta[i]) for i in order]

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        np.savez_compressed(os.path.join(path, "ltm.npz"),
                            fk=np.stack(self.fk) if self.fk else np.zeros((0, D)),
                            fv=np.stack(self.fv) if self.fv else np.zeros((0, D)),
                            sk=np.stack(self.sk) if self.sk else np.zeros((0, D)))
        json.dump(dict(fmeta=self.fmeta, stext=self.stext, smeta=self.smeta),
                  open(os.path.join(path, "ltm.json"), "w"), ensure_ascii=False)

    def load(self, path):
        f = os.path.join(path, "ltm.npz")
        if not os.path.exists(f): return False
        z = np.load(f); j = json.load(open(os.path.join(path, "ltm.json")))
        self.fk, self.fv, self.sk = list(z["fk"]), list(z["fv"]), list(z["sk"])
        self.fmeta, self.stext, self.smeta = j["fmeta"], j["stext"], j["smeta"]
        self._FK = self._SK = None
        return True


class Memory:
    """Фасад: сито + рабочая + долговременная. read_text() читает сырой текст; ask() отвечает."""
    def __init__(self, path=None):
        self.sieve, self.W, self.L = Sieve(), Working(), LongTerm()
        self.path = path
        if path and self.L.load(path):
            j = json.load(open(os.path.join(path, "sieve.json")))
            self.sieve.words, self.sieve.df, self.sieve.n_docs = j["words"], j["df"], j["n_docs"]

    def save(self):
        if not self.path: return
        self.L.save(self.path)
        json.dump(dict(words=self.sieve.words, df=self.sieve.df, n_docs=self.sieve.n_docs),
                  open(os.path.join(self.path, "sieve.json"), "w"), ensure_ascii=False)

    # ---- чтение сырого текста (интернет) ----
    @staticmethod
    def split_sentences(text):
        text = re.sub(r"\s+", " ", text).strip()
        parts = re.split(r"(?<=[.!?])\s+(?=[A-ZА-Я0-9])", text)
        out = []
        for p in parts:
            p = p.strip()
            if not (25 < len(p) < 400): continue
            if sum(c.isdigit() for c in p) / len(p) > 0.15: continue
            if re.match(r"^(Retrieved|Archived|ISBN|doi|pp\.|Jump up)", p): continue
            out.append(p)
        return out

    def read_text(self, text, source=""):
        n = 0
        for s in self.split_sentences(text):
            ws = toks(s); self.sieve.add(*content_words(ws)); self.sieve.count_doc(ws)
            self.L.add_sentence(self.sieve.bag(ws), s, dict(src=source, t=time.time()))
            n += 1
        self.L._SK = None
        return n

    def ask(self, question, topk=3):
        qv = self.sieve.bag(toks(question))
        return self.L.search_sentences(qv, topk)

    # ---- факты (структурные) ----
    def remember(self, x, role, y):
        self.sieve.add(x, y)
        self.W.add(x, role, y); self.L.add_fact(x, role, y)

    def recall(self, x, role):
        w, gap = self.sieve.cleanup(self.W.query(x, role), exclude={x})
        if gap >= 0.05: return w, "working"
        v = self.L.query_fact(x, role)
        if v is None: return None, "none"
        return self.sieve.cleanup(v, exclude={x})[0], "longterm"
