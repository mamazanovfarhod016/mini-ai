"""Эксперимент 11: ЧТЕНИЕ РЕАЛЬНОГО ТЕКСТА без градиента (раунд 2, экспериментатор-А).

Дыра из раунда 1 (критик): факты клались в память ГОТОВЫМИ кортежами, ни одно слово реального предложения
память не читала. Здесь вход — сырые строки data/corpus/{ru,uz}.txt (+ предложения из вопросов «вставь слово»
exam_*.jsonl, чтобы было что искать). Память САМА:
  1) режет строку на слова (нижний регистр, апострофы, знаки — как clean_question),
  2) кодирует каждое слово ситом letter_code (хэш букв, D=1024, ничему не учится),
  3) выводит ТИПЫ слов из собственных счётчиков (как field_learned в exp9, только тут типы — из частот):
       func    — слово встречается в ≥1% прочитанных предложений (служебное: «и», «в», «она», «va», «bilan»…)
       num     — есть цифра
       content — остальное; вес = idf = log(N/df) (редкое слово важнее)
  4) записывает за ОДИН шаг запись: ключ = код предложения, значение = само предложение (сырая строка).
Два варианта ключа:
  sum_typed  — Σ idf·code(w) по содержательным словам (служебные выброшены)
  pair_typed — то же + Σ bind(code(w_i), сдвиг(code(w_{i+1}))) по соседним содержательным словам (порядок)
Контроль: sum_raw (все слова, без типов и весов); базы: BM25, TF-IDF (точные слова, numpy), случайный.

Тесты:
  (1) вопросы по словам из exam_*.jsonl (meaning/synonym/antonym): найдено ли предложение, где есть это слово
      (R@1, R@5); запрос — весь текст вопроса и отдельно только слово. «fill» — найдено ли исходное предложение
      по вопросу с пропуском и 3 кандидатами, и верно ли слово-ответ (кандидат, который есть в найденном).
  (2) искажённый запрос: убрать 30% слов / перемешать / убрать+перемешать / заменить слово синонимом
      из exam-файлов — найдено ли исходное (R@1).
  (3) мост (bridge): 200 пар предложений A,B с общим редким словом (df=2). Запрос — A без слова-моста и без 30%
      других слов. Шаг 1: найти A. Шаг 2: память ЧИТАЕТ найденное A своим же ситом и ищет следующее
      (кроме A) — дошла ли до B. Отдельно: прямой запрос → B (без цепочки) и шаг 2 от истинного A (оракул).
Запуск: cd ~/Developer/mini-ai-memory/wt && python3 tools/exp11_read_text.py
"""
import sys, os, json, re, random, time, math, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_core import unit

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = 1024
N_LINES = 4000            # строк корпуса на язык
FUNC_FRAC = 0.01          # слово в ≥1% предложений → служебное
SEED = 11
LOG = None


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True)
    if LOG: LOG.write(s + "\n"); LOG.flush()


# ---------- сито: строка → слова → коды ----------
APOS = "`’‘ʻʼ´"


def clean(text):
    t = str(text).lower().replace("ё", "е")
    for a in APOS: t = t.replace(a, "'")
    t = re.sub(r"[^\w' ]+", " ", t)
    t = re.sub(r"(?<!\w)'|'(?!\w)", " ", t)      # апостроф только внутри слова (o'rta)
    return t.replace("_", " ").split()


_cc = {}


def letter_code(word, d=D, ns=(2, 3)):
    """Сито из tools/exp3_wordcode.py один в один (хэш crc32 n-грамм букв → ±1 в d измерений).
    Скопировано, т.к. импорт exp3 запускает весь его эксперимент (36 с); слово уже очищено clean()."""
    w = f"<{word}>"
    v = np.zeros(d)
    for n in ns:
        for i in range(len(w) - n + 1):
            h = zlib.crc32(w[i:i + n].encode())
            v[h % d] += 1.0 if (h >> 16) & 1 else -1.0
    return v


def code(w, d=D):
    k = (w, d)
    if k not in _cc: _cc[k] = unit(letter_code(w, d))
    return _cc[k]


def bind(a, b): return np.real(np.fft.ifft(np.fft.fft(a) * np.fft.fft(b)))


# ---------- память ----------
class TextMemory:
    """Список записей. Читает сырые строки; типы и веса слов — из собственных счётчиков df."""
    def __init__(self, variant, d=D):
        self.variant = variant; self.d = d; self.S = []; self.toks = []; self.df = {}; self.K = None; self.N = 0

    def read(self, sentences):
        for s in sentences:
            t = clean(s); self.S.append(s); self.toks.append(t)
            for w in set(t): self.df[w] = self.df.get(w, 0) + 1
        self.N = len(self.S)
        self.K = np.stack([self.key(t) for t in self.toks])

    def observe(self, texts):
        """Память прочла тексты (например, вопросы) только для счётчиков слов — записей не делает.
        Так «что значит слово» становится служебным, как «и» или «на»."""
        for s in texts:
            for w in set(clean(s)): self.df[w] = self.df.get(w, 0) + 1
            self.N += 1

    def wtype(self, w):
        if any(c.isdigit() for c in w): return "num"
        if self.df.get(w, 0) >= FUNC_FRAC * self.N: return "func"
        return "content"

    def weight(self, w):
        df = self.df.get(w, 0)
        return math.log((self.N + 1) / (df + 1)) + 1.0            # незнакомое слово (df=0) — как самое редкое

    def key(self, toks):
        d = self.d
        if self.variant == "sum_raw":
            k = sum((code(w, d) for w in toks), np.zeros(d)); return unit(k)
        cont = [w for w in toks if self.wtype(w) != "func"]
        k = sum((self.weight(w) * code(w, d) for w in cont), np.zeros(d))
        if self.variant == "pair_typed" and len(cont) > 1:
            pk = sum((bind(code(a, d), np.roll(code(b, d), 1)) for a, b in zip(cont, cont[1:])), np.zeros(d))
            k = unit(k) + unit(pk)
        return unit(k)

    def score(self, text): return self.K @ self.key(clean(text))

    def score_residual(self, found_text, query_text):
        """Шаг 2 цепочки по ВЕКТОРАМ: из ключа найденного предложения вычитаем то, что уже было в запросе
        (проекция), остаток = «новые» слова (в т.ч. слово-мост) → ищем следующее предложение."""
        ka = self.key(clean(found_text)); kq = self.key(clean(query_text))
        r = ka - (ka @ kq) * kq
        return self.K @ unit(r)


class BM25:
    """BM25 / TF-IDF по точным словам; матрица «предложение×слово» разреженная (scipy), иначе медленно."""
    def __init__(self, k1=1.5, b=0.75, tfidf=False): self.k1, self.b, self.tfidf = k1, b, tfidf; self.S = []

    def read(self, sentences):
        import scipy.sparse as sp
        self.S = list(sentences); self.toks = [clean(s) for s in sentences]; self.N = len(self.S)
        self.vocab = {}
        for t in self.toks:
            for w in set(t): self.vocab.setdefault(w, len(self.vocab))
        V = len(self.vocab)
        rows = [i for i, t in enumerate(self.toks) for _ in t]; cols = [self.vocab[w] for t in self.toks for w in t]
        M = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(self.N, V)); M.sum_duplicates()
        df = np.asarray((M > 0).sum(0)).ravel(); self.df = df
        self.idf = np.log((self.N - df + 0.5) / (df + 0.5) + 1.0)
        dl = np.asarray(M.sum(1)).ravel(); avg = dl.mean()
        M = M.tocoo()
        if self.tfidf:
            data = M.data * np.log((self.N + 1) / (df[M.col] + 1))
            W = sp.csr_matrix((data, (M.row, M.col)), shape=(self.N, V))
            nrm = np.sqrt(np.asarray(W.multiply(W).sum(1)).ravel()); self.W = sp.diags(1 / np.maximum(nrm, 1e-9)) @ W
        else:
            data = M.data * (self.k1 + 1) / (M.data + self.k1 * (1 - self.b + self.b * dl[M.row] / avg))
            self.W = sp.csr_matrix((data, (M.row, M.col)), shape=(self.N, V))
        self.W = self.W.tocsr()

    def score(self, text):
        q = np.zeros(len(self.vocab), dtype=np.float32)
        for w in clean(text):
            if w in self.vocab: q[self.vocab[w]] += 1
        if self.tfidf:
            q = q * np.log((self.N + 1) / (self.df + 1)); q /= max(np.linalg.norm(q), 1e-9)
            return np.asarray(self.W @ q).ravel()
        return np.asarray(self.W @ (q * self.idf)).ravel()

    def score_residual(self, found_text, query_text):
        """Для базы остаток — по словам: слова найденного предложения, которых не было в запросе."""
        qs = set(clean(query_text)); return self.score(" ".join(w for w in clean(found_text) if w not in qs))


class RandomMem:
    def __init__(self): self.rng = np.random.default_rng(SEED)

    def read(self, sentences): self.S = list(sentences); self.N = len(self.S)

    def score(self, text): return self.rng.random(self.N)

    def score_residual(self, a, q): return self.score(a)


# ---------- данные ----------
def stem(w):
    return w if len(w) <= 4 else (w[:-1] if len(w) <= 6 else w[:-2])


def has_word(toks, word):
    s = stem(word); return any(t.startswith(s) for t in toks)


def load(lang):
    lines = [l.strip() for l in open(f"{ROOT}/data/corpus/{lang}.txt", encoding="utf-8")]
    lines = [l for l in lines if 4 <= len(clean(l)) <= 40]
    rng = random.Random(SEED); rng.shuffle(lines)
    sents = []; seen = set()
    for l in lines:
        k = " ".join(clean(l))
        if k not in seen: seen.add(k); sents.append(l)
        if len(sents) >= N_LINES: break
    exam = [json.loads(l) for l in open(f"{ROOT}/data/words/exam_{lang}.jsonl", encoding="utf-8")]
    fills = [e for e in exam if e["kind"] == "fill"]
    for e in fills:                                        # предложения «вставь слово» тоже читаем как текст
        k = " ".join(clean(e["a"]))
        if k not in seen: seen.add(k); sents.append(e["a"].strip())
    rng.shuffle(sents)
    return sents, exam


def synonyms(exam, lang):
    syn = {}
    for e in exam:
        if e["kind"] != "synonym": continue
        a = e["a"]
        if lang == "ru":
            ws = re.findall(r"«([^»]+)»", a)[1:]
        else:
            ws = [x.strip(" .") for x in a.split("—")[-1].split(",")]
        ws = [clean(w)[0] for w in ws if clean(w) and len(clean(w)) == 1]
        if ws: syn[clean(e["word"])[0]] = ws[0]
    return syn


# ---------- метрики ----------
def rank_of(scores, target_idx, exclude=None):
    s = scores.copy()
    if exclude is not None: s[exclude] = -np.inf
    return int((s > s[target_idx]).sum())        # 0 = первое место


def topk(scores, k, exclude=None):
    s = scores.copy()
    if exclude is not None: s[exclude] = -np.inf
    return np.argpartition(-s, k)[:k] if k < len(s) else np.arange(len(s))


def run_lang(lang):
    sents, exam = load(lang)
    toks = [clean(s) for s in sents]; N = len(sents)
    log(f"\n=== {lang}: прочитано {N} предложений (корпус {min(N_LINES, N)} + предложения «вставь слово»), "
        f"слов всего {sum(len(t) for t in toks)}, уникальных {len({w for t in toks for w in t})}")
    mems = {}
    for name, var, d in (("sum_raw", "sum_raw", D), ("sum_typed", "sum_typed", D), ("pair_typed", "pair_typed", D),
                         ("sum_typed4k", "sum_typed", 4096)):
        t0 = time.time(); m = TextMemory(var, d); m.read(sents); mems[name] = m
        log(f"  {name:11} D={d} записано за {time.time()-t0:.1f} s")
    for name, m in (("bm25", BM25()), ("tfidf", BM25(tfidf=True)), ("random", RandomMem())):
        t0 = time.time(); m.read(sents); mems[name] = m; log(f"  {name:11} индекс за {time.time()-t0:.1f} s")
    m0 = mems["sum_typed"]
    func = sorted([w for w in m0.df if m0.wtype(w) == "func"], key=lambda w: -m0.df[w])
    log(f"  типы из памяти: служебных {len(func)} (топ: {' '.join(func[:15])}), "
        f"числовых {sum(m0.wtype(w)=='num' for w in m0.df)}, содержательных {sum(m0.wtype(w)=='content' for w in m0.df)}")
    res = {"N": N, "func_words": func[:30], "n_func": len(func)}

    # ---- (1) вопросы по словам ----
    qs = [e for e in exam if e["kind"] in ("meaning", "synonym", "antonym")]
    rel = {}
    for e in qs:
        w = clean(e["word"]); w = w[0] if w else ""
        if not w or len(w) < 3: continue
        idx = [i for i, t in enumerate(toks) if has_word(t, w)]
        if idx: rel[e["q"]] = (e, idx)
    log(f"\n(1) вопросы по словам: {len(rel)} вопросов из {len(qs)}, чьё слово есть хоть в одном прочитанном предложении")
    res["t1"] = {}
    for qmode in ("question", "word", "question_observed"):
        if qmode == "question_observed":           # память прочла сами вопросы (только счётчики слов, записей нет)
            for m in mems.values():
                if isinstance(m, TextMemory): m.observe([e["q"] for e in qs])
            m1 = mems["sum_typed"]; frame = [w for w in ("что", "значит", "слово", "синоним", "антоним", "so'zi", "nimani", "anglatadi") if w in m1.df]
            log(f"  (память прочла {len(qs)} вопросов как текст; теперь служебные: " +
                " ".join(f"{w}={m1.wtype(w)}" for w in frame) + ")")
        res["t1"][qmode] = {}
        for name, m in mems.items():
            if qmode == "question_observed" and not isinstance(m, TextMemory): continue
            r1 = r5 = 0
            for q, (e, idx) in rel.items():
                s = m.score(e["word"] if qmode == "word" else q); top = topk(s, 5); best = int(s.argmax())
                r1 += best in idx; r5 += bool(set(top.tolist()) & set(idx))
            res["t1"][qmode][name] = {"R1": r1 / len(rel), "R5": r5 / len(rel), "n": len(rel)}
            log(f"  запрос={qmode:18} {name:11} R@1 {r1/len(rel)*100:5.1f}%  R@5 {r5/len(rel)*100:5.1f}%")
    # fill
    fills = [e for e in exam if e["kind"] == "fill"]
    pos = {" ".join(clean(s)): i for i, s in enumerate(sents)}
    fl = [(e, pos[" ".join(clean(e["a"]))]) for e in fills if " ".join(clean(e["a"])) in pos]
    log(f"(1b) «вставь слово»: {len(fl)} вопросов, исходное предложение прочитано памятью")
    res["t1_fill"] = {}
    for name, m in mems.items():
        r1 = r5 = wok = 0
        for e, ti in fl:
            s = m.score(e["q"]); best = int(s.argmax()); r1 += best == ti; r5 += ti in topk(s, 5)
            cands = [clean(c)[0] for c in re.findall(r"\(([^)]*)\)", e["q"])[0].split(",") if clean(c)]
            gold = [c for c in cands if c in toks[ti]]            # кандидат, стоящий в исходном предложении (форма слова)
            inbest = [c for c in cands if c in toks[best]]        # кандидаты, которые есть в НАЙДЕННОМ предложении
            wok += len(gold) == 1 and inbest == gold
        res["t1_fill"][name] = {"R1": r1 / len(fl), "R5": r5 / len(fl), "word_acc": wok / len(fl), "n": len(fl)}
        log(f"  {name:11} R@1 {r1/len(fl)*100:5.1f}%  R@5 {r5/len(fl)*100:5.1f}%  слово верно {wok/len(fl)*100:5.1f}%")

    # ---- (2) искажённые запросы ----
    rng = random.Random(SEED)
    cand = [i for i in range(N) if len(toks[i]) >= 6]; sample = rng.sample(cand, 500)
    syn = synonyms(exam, lang)
    log(f"\n(2) искажённые запросы: 500 предложений (≥6 слов); словарь синонимов из exam: {len(syn)} слов")

    def perturb(i, kind, rr):
        t = list(toks[i])
        if kind == "exact": return " ".join(t)
        if kind == "drop30":
            drop = set(rr.sample(range(len(t)), round(len(t) * 0.3))); return " ".join(w for j, w in enumerate(t) if j not in drop)
        if kind == "shuffle": rr.shuffle(t); return " ".join(t)
        if kind == "drop30_shuffle":
            keep = [w for j, w in enumerate(t) if j not in set(rr.sample(range(len(t)), round(len(t) * 0.3)))]; rr.shuffle(keep); return " ".join(keep)
        if kind == "first_half": return " ".join(t[:max(2, len(t) // 2)])
        if kind == "synonym":
            for j, w in enumerate(t):
                for lemma, sw in syn.items():
                    if w.startswith(stem(lemma)) and len(lemma) >= 4: t[j] = sw; return " ".join(t)
            return None
    res["t2"] = {}
    for kind in ("exact", "drop30", "shuffle", "drop30_shuffle", "first_half", "synonym"):
        rr = random.Random(SEED + 1); queries = [(i, perturb(i, kind, rr)) for i in sample]
        queries = [(i, q) for i, q in queries if q]
        res["t2"][kind] = {"n": len(queries)}
        line = f"  {kind:15} (n={len(queries):3}) "
        for name, m in mems.items():
            if not queries or (name == "sum_typed4k" and kind not in ("drop30_shuffle", "first_half")): continue
            r1 = sum(int(m.score(q).argmax()) == i for i, q in queries) / len(queries)
            res["t2"][kind][name] = r1; line += f"{name} {r1*100:5.1f}%  "
        log(line)

    # ---- (3) мост: два предложения с общим редким словом ----
    dfw = m0.df
    bridges = sorted(w for w, c in dfw.items() if c == 2 and len(w) >= 5 and w.isalpha() and m0.wtype(w) == "content")
    rng = random.Random(SEED + 2); rng.shuffle(bridges); pairs = []     # sorted → один и тот же набор пар при каждом запуске
    used = set()
    for w in bridges:
        ab = [i for i, t in enumerate(toks) if w in t]
        if len(ab) != 2 or len(toks[ab[0]]) < 5 or len(toks[ab[1]]) < 5 or ab[0] in used or ab[1] in used: continue
        pairs.append((w, ab[0], ab[1])); used.update(ab)
        if len(pairs) >= 200: break
    log(f"\n(3) мост: {len(pairs)} пар предложений с общим редким словом (df=2). "
        f"Запрос = A без слова-моста и без 30% слов → A → (память читает A) → B")
    res["t3"] = {"n": len(pairs)}
    rr = random.Random(SEED + 3)
    queries = []
    for w, a, b in pairs:
        t = [x for x in toks[a] if x != w]; drop = set(rr.sample(range(len(t)), round(len(t) * 0.3)))
        queries.append(" ".join(x for j, x in enumerate(t) if j not in drop))
    for name, m in mems.items():
        hop1 = hop2 = hop2_5 = direct = oracle = bridge_ok = resid = resid5 = 0
        for (w, a, b), q in zip(pairs, queries):
            s = m.score(q); a1 = int(s.argmax()); hop1 += a1 == a
            direct += int(s.argmax()) == b
            s2 = m.score(sents[a1]); b1 = int(np.where(np.arange(N) == a1, -np.inf, s2).argmax())
            hop2 += (a1 == a and b1 == b); hop2_5 += (a1 == a and b in topk(s2, 6, exclude=a1))
            if a1 == a and b1 == b:
                common = [x for x in set(toks[a1]) & set(toks[b1]) if m0.wtype(x) == "content"]
                bridge_ok += (w in common)
            sr = m.score_residual(sents[a1], q); br = int(np.where(np.arange(N) == a1, -np.inf, sr).argmax())
            resid += (a1 == a and br == b); resid5 += (a1 == a and b in topk(sr, 6, exclude=a1))
            so = m.score(sents[a]); oracle += int(np.where(np.arange(N) == a, -np.inf, so).argmax()) == b
        n = len(pairs)
        res["t3"][name] = {"hop1_R1": hop1 / n, "chain_R1": hop2 / n, "chain_R5": hop2_5 / n,
                           "chain_residual_R1": resid / n, "chain_residual_R5": resid5 / n,
                           "direct_B_R1": direct / n, "oracle_hop2_R1": oracle / n, "bridge_word_found": bridge_ok / n}
        log(f"  {name:11} шаг1 (A) {hop1/n*100:5.1f}%  цепочка A→B R@1 {hop2/n*100:5.1f}%  R@5 {hop2_5/n*100:5.1f}%  "
            f"| остаток (A минус запрос) R@1 {resid/n*100:5.1f}%  R@5 {resid5/n*100:5.1f}%  "
            f"| напрямую в B {direct/n*100:5.1f}%  шаг2 от истинного A {oracle/n*100:5.1f}%")
    return res


if __name__ == "__main__":
    LOG = open(f"{OUT}/exp11.log", "w")
    t0 = time.time(); log(f"exp11: чтение реального текста, D={D}, строк корпуса на язык {N_LINES}, служебное = df≥{FUNC_FRAC*100:.0f}%")
    res = {}
    for lang in ("ru", "uz"):
        res[lang] = run_lang(lang)
    res["sec"] = round(time.time() - t0, 1); log(f"\nвсего {res['sec']} s")
    json.dump(res, open(f"{OUT}/exp11_read_text.json", "w"), indent=1, ensure_ascii=False)
    LOG.close()
