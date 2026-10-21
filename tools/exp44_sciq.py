"""Эксперимент 44: ШКОЛЬНАЯ ПРОГРАММА — SciQ (Welbl et al. 2017), 1000 test-вопросов, 4 варианта.
Память = учебные абзацы из TRAIN (11 679 абзацев, ~60k предложений). Test-вопросы при чтении не видны.
Три способа ответить (все без градиента), от простого к рассуждению:
  A. МЕШОК: вопрос+вариант → ближайшее предложение памяти; вариант с максимальным сходством. (контроль)
  B. ФАКТЫ: читатель → факты; вариант, чьи слова покрыты фактами, в которых есть слова вопроса.
  C. СВЯЗЬ ВОПРОС→ВАРИАНТ через память: для каждого варианта ищем предложение, где вместе встречаются
     слова вопроса И вариант (совместная встречаемость = связь). Счёт = IDF-вес общих слов вопроса в этом
     предложении. Это «рассуждение в 1 шаг с проверкой связи», не просто похожесть.
  D. C + 2-й шаг: если ни один вариант не связан напрямую — через промежуточное предложение (мост):
     предложение с словами вопроса → его редкие слова → предложение с вариантом и этими словами.
Метрики: точность на 1000; случайный выбор = 25%. Литература: без обучения (BM25) ~50–60%, модели 90%+.
"""
import sys, os, json, time, re, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from mini_ai.memory_v2 import toks, content_words, C, unit, D
from mini_ai.reader import Reader

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
train = pd.read_parquet("data/school/sciq/data/train-00000-of-00001.parquet")
test = pd.read_parquet("data/school/sciq/data/test-00000-of-00001.parquet")

# ---- память: предложения учебника ----
sents = []
for sup in train["support"].dropna():
    for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", str(sup)):
        s = s.strip()
        if 20 < len(s) < 400: sents.append(s)
sents = list(dict.fromkeys(sents)); N = len(sents)
tok_s = [content_words(toks(s)) for s in sents]
dfc = collections.Counter(w for ws in tok_s for w in set(ws)); idf = {w: float(np.log((1 + N) / (1 + c)) + 1) for w, c in dfc.items()}
inv = collections.defaultdict(set)
for i, ws in enumerate(tok_s):
    for w in set(ws): inv[w].add(i)
print(f"учебник: {len(train)} абзацев → {N} предложений, словарь {len(dfc)}", flush=True)

def bagv(ws): return unit(sum(idf.get(w, 1.0) * C(w) for w in ws)) if ws else np.zeros(D)
V = np.stack([bagv(ws) for ws in tok_s])

# ---- факты ----
r = Reader(); facts = []
for i, s in enumerate(sents):
    fs, _ = r.read(s, "")
    for X, rel, Y in fs: facts.append((set(content_words(toks(X))), set(content_words(toks(Y))), i))
print(f"фактов {len(facts)}", flush=True)

def w_idf(ws): return sum(idf.get(w, 1.0) for w in ws)

def method_A(q, opts):
    qw = content_words(toks(q)); sc = []
    for o in opts:
        v = bagv(qw + content_words(toks(o))); sc.append(float((V @ v).max()))
    return int(np.argmax(sc))

def method_B(q, opts):
    qw = set(content_words(toks(q))); sc = []
    for o in opts:
        ow = set(content_words(toks(o))); best = 0.0
        for X, Y, i in facts:
            if ow & (X | Y) and qw & (X | Y): best = max(best, w_idf(qw & (X | Y)) + w_idf(ow & (X | Y)))
        sc.append(best)
    return int(np.argmax(sc)) if max(sc) > 0 else method_A(q, opts)

def joint_score(qw, ow, q_toks=None, opt_toks=None):
    """лучшее предложение, где есть И слова варианта, И слова вопроса.
    Счёт = ПОКРЫТИЕ вопроса этим предложением (доля IDF слов вопроса, найденных в нём) ^2 × доля слов варианта.
    Квадрат: предложение, содержащее почти весь вопрос, — это и есть ответ; оно должно давить остальные.
    + позиция: если в предложении слова вопроса слева от варианта идут в том же порядке — бонус (рамка)."""
    cand = set.union(*[inv.get(w, set()) for w in ow]) if ow else set()
    best = 0.0; qidf = w_idf(qw) or 1.0
    for i in cand:
        sw = set(tok_s[i]); hit_o = ow & sw
        if not hit_o: continue
        cov = w_idf(qw & sw) / qidf
        s = (cov ** 2) * (len(hit_o) / len(ow))
        if q_toks and opt_toks:                       # позиционный бонус: вариант стоит после слов вопроса (как в вопросе «... is called ?»)
            st = toks(sents[i]); pos_o = next((k for k in range(len(st)) if st[k:k + len(opt_toks)] == opt_toks), -1)
            if pos_o > 0 and sum(1 for w in q_toks[-4:] if w in st[max(0, pos_o - 6):pos_o]) >= 2: s *= 1.5
        best = max(best, s)
    return best

def method_C(q, opts):
    qw = set(content_words(toks(q))); qt = content_words(toks(q))
    sc = [joint_score(qw, set(content_words(toks(o))), qt, toks(o)) for o in opts]
    return int(np.argmax(sc)) if max(sc) > 0 else method_A(q, opts)

def method_D(q, opts):
    qw = set(content_words(toks(q))); sc = [joint_score(qw, set(content_words(toks(o)))) for o in opts]
    if max(sc) > 0 and sorted(sc)[-1] > 1.3 * sorted(sc)[-2]: return int(np.argmax(sc))
    # мост: предложения с ≥2 словами вопроса → их редкие слова (IDF > 3) → связь с вариантом
    qsents = collections.Counter()
    for w in qw:
        for i in inv.get(w, ()): qsents[i] += idf.get(w, 1.0)
    bridge = set()
    for i, _ in qsents.most_common(5): bridge |= {w for w in tok_s[i] if idf.get(w, 0) > 3.0 and w not in qw}
    sc2 = [sc[k] + 0.5 * joint_score(bridge, set(content_words(toks(o)))) for k, o in enumerate(opts)]
    return int(np.argmax(sc2)) if max(sc2) > 0 else method_A(q, opts)

rng = np.random.default_rng(0); res = {}
for name, fn in (("A мешок", method_A), ("B факты", method_B), ("C связь", method_C), ("D связь+мост", method_D)):
    ok = 0; t0 = time.time()
    for _, row in test.iterrows():
        opts = [row["correct_answer"], row["distractor1"], row["distractor2"], row["distractor3"]]
        perm = rng.permutation(4); shuffled = [opts[j] for j in perm]; gold = int(np.where(perm == 0)[0][0])
        ok += fn(row["question"], shuffled) == gold
    res[name] = ok / len(test)
    print(f"  {name:14} точность {ok/len(test)*100:.1f}%  ({(time.time()-t0)/len(test)*1000:.0f} мс/вопрос)", flush=True)
print("случайный выбор: 25%")
json.dump(dict(n=len(test), sentences=N, facts=len(facts), **{k: v for k, v in res.items()}), open(f"{OUT}/exp44_sciq.json", "w"), ensure_ascii=False, indent=1)
