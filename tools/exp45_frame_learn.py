"""Эксперимент 45: ПОНИМАНИЕ ВОПРОСА через рамку + «не знаю → прочитал → знаю».
SciQ 1000 test. Три режима:
  1. рамка (question.py) → факт с обязательным покрытием → ответ; нет факта → ОТКАЗ (честно «не знаю»);
  2. рамка, а при отказе — старый метод C (связь по мешку);
  3. ЦИКЛ ОБУЧЕНИЯ: вопросы, на которые отказ, — агент читает support этого вопроса (как ученик открывает
     учебник на нужной странице) → память растёт → тот же вопрос снова. Считаем: доля отказов до/после,
     точность на тех, где раньше был отказ. Это «знает мало → покажет → научится».
Метрики: точность, доля отказов, точность СРЕДИ ответов (precision — когда отвечает, насколько права).
"""
import sys, os, json, time, re, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from mini_ai.memory_v2 import toks, content_words
from mini_ai.reader import Reader
from mini_ai.question import parse_question, answer_by_frame

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
train = pd.read_parquet("data/school/sciq/data/train-00000-of-00001.parquet")
test = pd.read_parquet("data/school/sciq/data/test-00000-of-00001.parquet")

def split(sup): return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", str(sup)) if 20 < len(s.strip()) < 400]
sents = list(dict.fromkeys(s for sup in train["support"].dropna() for s in split(sup)))
tok_s = [content_words(toks(s)) for s in sents]
dfc = collections.Counter(w for ws in tok_s for w in set(ws)); N = len(sents)
idf = {w: float(np.log((1 + N) / (1 + c)) + 1) for w, c in dfc.items()}
inv = collections.defaultdict(set)
for i, ws in enumerate(tok_s):
    for w in set(ws): inv[w].add(i)

facts_idx = []; r = Reader()
def add_sentence(s, i):
    for X, rel, Y in r.read(s, "")[0]:
        facts_idx.append((set(content_words(toks(X))), rel, set(content_words(toks(Y))), X, Y, i))
for i, s in enumerate(sents): add_sentence(s, i)
print(f"учебник {N} предложений, фактов {len(facts_idx)}", flush=True)
by_word = collections.defaultdict(list)
for k, f in enumerate(facts_idx):
    for w in f[0] | f[2]: by_word[w].append(k)

def w_idf(ws): return sum(idf.get(w, 1.0) for w in ws)
def frame_answer(q):
    fr = parse_question(q)
    known = set(fr["Y"] if fr["slot"] == "X" else fr["X"])
    if not known: return None, fr
    cand = set.intersection(*[set(by_word.get(w, [])) for w in known]) if known else set()   # все обязательные слова
    if not cand: return None, fr
    res = answer_by_frame(fr, [facts_idx[k] for k in cand], idf)
    return (res[0] if res else None), fr

def pick(q, opts):
    """→ (индекс варианта или None, как)"""
    best, fr = frame_answer(q)
    if best:
        s, ans, fact = best; aw = set(toks(ans)) | set(toks(fact[0] + " " + fact[2]))
        sc = [len(set(content_words(toks(o))) & aw) / max(1, len(content_words(toks(o)))) for o in opts]
        if max(sc) > 0: return int(np.argmax(sc)), "рамка"
    return None, "не знаю"

def method_C(q, opts):
    qw = set(content_words(toks(q))); qidf = w_idf(qw) or 1; sc = []
    for o in opts:
        ow = set(content_words(toks(o))); cand = set.union(*[inv.get(w, set()) for w in ow]) if ow else set(); best = 0.0
        for i in cand:
            sw = set(tok_s[i]); ho = ow & sw
            if ho: best = max(best, (w_idf(qw & sw) / qidf) ** 2 * len(ho) / len(ow))
        sc.append(best)
    return int(np.argmax(sc)) if max(sc) > 0 else 0

rng = np.random.default_rng(0); rows = []
for _, row in test.iterrows():
    opts = [row["correct_answer"], row["distractor1"], row["distractor2"], row["distractor3"]]
    perm = rng.permutation(4); sh = [opts[j] for j in perm]; gold = int(np.where(perm == 0)[0][0])
    rows.append((row["question"], sh, gold, str(row["support"])))

def evaluate(label):
    ok = ans = okC = 0; refuse = []
    for k, (q, sh, gold, sup) in enumerate(rows):
        p, how = pick(q, sh)
        if p is None: refuse.append(k); c = method_C(q, sh); okC += c == gold
        else: ans += 1; ok += p == gold; okC += p == gold
    n = len(rows)
    print(f"  {label}: отвечает {ans}/{n} ({ans/n*100:.0f}%), из них верно {ok} (precision {ok/max(1,ans)*100:.1f}%), отказов {len(refuse)}; "
          f"рамка+мешок при отказе: {okC/n*100:.1f}%", flush=True)
    return refuse, ok, ans, okC

print("--- 1. до дочитывания ---"); refuse1, ok1, ans1, okC1 = evaluate("рамка")
# ЦИКЛ: на отказах — читаем support вопроса (ученик открыл учебник), память растёт
t0 = time.time(); added = 0
for k in refuse1:
    for s in split(rows[k][3]):
        if s not in sents:
            sents.append(s); tok_s.append(content_words(toks(s))); i = len(sents) - 1
            for w in set(tok_s[i]): inv[w].add(i)
            n0 = len(facts_idx); add_sentence(s, i)
            for j in range(n0, len(facts_idx)):
                for w in facts_idx[j][0] | facts_idx[j][2]: by_word[w].append(j)
            added += 1
print(f"--- 2. дочитал {added} предложений по {len(refuse1)} отказам ({time.time()-t0:.1f}s) ---")
refuse2, ok2, ans2, okC2 = evaluate("рамка после чтения")
learned = [k for k in refuse1 if k not in refuse2]
ok_learned = sum(pick(rows[k][0], rows[k][1])[0] == rows[k][2] for k in learned)
print(f"  из {len(refuse1)} отказов теперь отвечает на {len(learned)}, верно {ok_learned} ({ok_learned/max(1,len(learned))*100:.0f}%)")
print("--- примеры: не знал → прочитал → ответил ---")
for k in learned[:5]:
    q, sh, gold, sup = rows[k]; p, how = pick(q, sh); best, fr = frame_answer(q)
    print(f"  Q: {q}\n     рамка: slot={fr['slot']} X={fr['X']} rel={fr['rel']} Y={fr['Y']}\n     факт: {best[2][0]} —{best[2][1]}→ {best[2][2][:60]}  → «{sh[p]}» {'✓' if p == gold else '✗ (gold: ' + sh[gold] + ')'}")
json.dump(dict(before=dict(ok=ok1, answered=ans1, refused=len(refuse1), with_fallback=okC1), after=dict(ok=ok2, answered=ans2, refused=len(refuse2), with_fallback=okC2),
               learned=len(learned), ok_learned=ok_learned, n=len(rows)), open(f"{OUT}/exp45_frame_learn.json", "w"), indent=1)
