"""Эксперимент 16a: ЧТЕНИЕ без градиента на реальном тексте — учимся синонимам.
Тезис round2 §2.2 (random indexing, Kanerva–Kristofersson–Holst 2000; Sahlgren 2005):
синонимы — разные буквы, значит буквенный код (letter_code) их не сближает; но если слово
получает «контекстный» код = сумма (со знаком направления) соседей по ВСЕМ предложениям,
где оно встретилось, — синонимы (которые встречаются в похожих контекстах) сближаются,
а это чтение: один проход по тексту, сложение векторов, НИКАКОГО backprop.

Данные: data/corpus/ru.txt (13639 предложений, читаем как учебный поток, без разметки),
проверка — data/words/exam_ru.jsonl kind=synonym (522 пар «слово → синоним(ы)», написано
человеком, не нами). Это первый тест ядра на НАСТОЯЩЕМ тексте и настоящей лингвистической
задаче (а не на синтетике exp3-exp15).

Три кода слова сравниваем:
  canon   — только канонический код C(w) (случайный, не учится) — должен быть ~случайным по синонимам
  letters — буквенный хэш L(w) (как в exp3/exp5) — должен путать ПОХОЖИЕ ПО БУКВАМ, не по смыслу
  ctx     — контекстный код после чтения корпуса: ctx(w) = unit(Σ_occurrences Σ_{|Δ|≤K} sign(Δ)·Π^{|Δ|}·C(соседΔ))
            (Π — фиксированная перестановка координат = «направление»; без обучения, один проход)
  mix     — [αС(w) ; (1−α) ctx(w)] — слово узнаётся по букве, смысл — по контексту (round2 формула)
Метрика на каждую пару (word, synonym) из exam_ru: ранг synonym среди всех слов словаря корпуса
по косинусу с word (1 = первый, т.е. нашёлся сразу); медианный ранг, доля top-10, top-50, vs random.
"""
import sys, os, json, re, time, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_core import unit
from mini_ai.transformer import clean_question

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
D = 512; K = 2         # окно контекста: 2 слова влево/вправо
rng = np.random.default_rng(0)


def letter_code(word, d=192, ns=(2, 3)):
    w = f"<{clean_question(word)}>"; v = np.zeros(d)
    for n in ns:
        for i in range(len(w) - n + 1):
            h = zlib.crc32(w[i:i + n].encode()); v[h % d] += 1.0 if (h >> 16) & 1 else -1.0
    return v


P = rng.standard_normal((192, D)) / np.sqrt(192)
_L = {}
def code_letters(w):
    if w not in _L: _L[w] = unit(letter_code(w) @ P)
    return _L[w]

_C = {}
def code_canon(w):
    if w not in _C: _C[w] = unit(np.random.default_rng(zlib.crc32(w.encode())).standard_normal(D))
    return _C[w]

# фиксированная перестановка координат = «направление/расстояние» (Sahlgren–Holst–Kanerva 2008)
_perm = rng.permutation(D)
def Pi(v, k):
    if k == 0: return v
    idx = _perm.copy()
    for _ in range(abs(k) - 1): idx = idx[_perm]
    return v[idx] if k > 0 else v[np.argsort(idx)]


def words(s): return [w for w in re.findall(r"[\w']+", s.lower()) if w]


def read_corpus(path, limit=None):
    """Один проход по тексту: для каждого слова копим контекстный вектор. Без обучения — сложение."""
    ctx = {}
    n_sent = 0
    for line in open(path, encoding="utf-8"):
        s = words(line)
        if not s: continue
        n_sent += 1
        if limit and n_sent > limit: break
        cs = [code_canon(w) for w in s]
        for i, w in enumerate(s):
            acc = ctx.get(w)
            if acc is None: acc = np.zeros(D)
            for dlt in range(-K, K + 1):
                j = i + dlt
                if dlt == 0 or j < 0 or j >= len(s): continue
                acc = acc + np.sign(dlt) * Pi(cs[j], abs(dlt))
            ctx[w] = acc
    return {w: unit(v) for w, v in ctx.items() if np.linalg.norm(v) > 1e-9}, n_sent


def extract_syn(a): return re.findall(r"[«\"]([^»\"]+)[»\"]", a)[1:]


def run():
    t0 = time.time()
    ctx, n_sent = read_corpus("data/corpus/ru.txt")
    t_read = time.time() - t0
    rows = [json.loads(l) for l in open("data/words/exam_ru.jsonl", encoding="utf-8")]
    syn_rows = [r for r in rows if r["kind"] == "synonym"]
    vocab = sorted(set(ctx.keys()) | {r["word"] for r in syn_rows} | {s for r in syn_rows for s in extract_syn(r["a"])})
    V = len(vocab)
    print(f"прочитано {n_sent} предложений ru.txt, слов с контекстом: {len(ctx)}, общий словарь теста: {V}, время чтения {t_read:.1f}s", flush=True)

    Ccanon = np.stack([code_canon(w) for w in vocab])
    Cletters = np.stack([code_letters(w) for w in vocab])
    Cctx = np.stack([ctx.get(w, np.zeros(D)) for w in vocab])
    alpha = 0.3
    Cmix = unit(np.hstack([alpha * Ccanon, (1 - alpha) * (Cctx if Cctx.any() else Ccanon)]))

    def eval_scheme(name, M, qvec_fn, min_occ=None, occ=None):
        pairs = 0; ranks = []; hit10 = hit50 = 0
        for r in syn_rows:
            w = r["word"]; syns = [s for s in extract_syn(r["a"]) if s in vocab and s != w]
            if not syns or w not in vocab: continue
            if min_occ is not None and occ.get(w, 0) < min_occ: continue
            qi = vocab.index(w); q = qvec_fn(qi)
            s = M @ q; order = np.argsort(-s)
            rank_pos = {vocab[idx]: pos for pos, idx in enumerate(order)}
            for syn in syns:
                if min_occ is not None and occ.get(syn, 0) < min_occ: continue
                rk = rank_pos.get(syn)
                if rk is None: continue
                pairs += 1; ranks.append(rk + 1)
                hit10 += rk < 10; hit50 += rk < 50
        med = float(np.median(ranks)) if ranks else float("nan")
        print(f"  {name:10} пар {pairs:4}  медианный ранг {med:7.1f} / {V}  top10 {hit10/max(1,pairs)*100:5.1f}%  top50 {hit50/max(1,pairs)*100:5.1f}%", flush=True)
        return dict(pairs=pairs, median_rank=med, V=V, top10=hit10/max(1, pairs), top50=hit50/max(1, pairs))

    import collections
    occ = collections.Counter()
    for line in open("data/corpus/ru.txt", encoding="utf-8"):
        for w in words(line): occ[w] += 1
    cov = np.mean([occ.get(r["word"], 0) >= 3 for r in syn_rows])
    print(f"доля проверочных слов, встретившихся в корпусе >=3 раз: {cov*100:.1f}%", flush=True)

    res = {}
    res["canon"] = eval_scheme("canon", Ccanon, lambda i: Ccanon[i])
    res["letters"] = eval_scheme("letters", Cletters, lambda i: Cletters[i])
    res["ctx"] = eval_scheme("ctx (прочитано)", Cctx, lambda i: Cctx[i])
    res["ctx_covered"] = eval_scheme("ctx (>=3 вхожд.)", Cctx, lambda i: Cctx[i], min_occ=3, occ=occ)
    res["mix"] = eval_scheme("mix (letters+ctx)", Cmix, lambda i: Cmix[i])
    rng2 = np.random.default_rng(1)
    rand_ranks = []
    for r in syn_rows:
        syns = [s for s in extract_syn(r["a"]) if s in vocab]
        if not syns: continue
        for _ in syns: rand_ranks.append(rng2.integers(0, V))
    print(f"  {'random':10} ожидаемый медианный ранг (контроль) ~ {V/2:.0f} / {V}  (реальный {np.median(rand_ranks):.0f})", flush=True)
    res["random_expected_median"] = V / 2
    json.dump(res, open(f"{OUT}/exp16_core_synonyms.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    run()
