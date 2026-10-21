"""Эксперимент 26: РЕЧЬ БЕЗ ГРАДИЕНТА — память следующего слова в том же векторном пространстве.
Плато: ядро отвечает чужими предложениями. Новый механизм: ключ = bind(w₋₂, w₋₁) (порядок важен —
bind с позиционными ролями), значение = C(w). Запись: один проход по тексту; чтение: ключ → список →
cleanup по словарю сита. Та же память, что для фактов (LongTerm.add_fact) — слова и факты в одном месте.
Плюс «направление»: при генерации ответа на вопрос прибавляем к значению вектор темы (мешок вопроса),
чтобы из нескольких продолжений выбирать то, что про тему.
Метрики (честные, held-out): top-1 следующего слова на 10% отложенных предложений; сравнение с
обычной таблицей 3-грамм (словарь Python) — должны совпасть по точности (это и есть доказательство, что
вектор хранит то же, что таблица, но в одном пространстве с фактами). Генерация: 5 примеров текстом.
"""
import sys, os, json, time, re, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_v2 import Memory, toks, C, bind, unit, D

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
mem = Memory(path=f"{OUT}/wiki_memory")
sents = [toks(s) for s in mem.L.stext]
sents = [["<s>", "<s>"] + s + ["</s>"] for s in sents if len(s) >= 4]
rng = np.random.default_rng(0); rng.shuffle(sents)
n_test = len(sents) // 10; test, train = sents[:n_test], sents[n_test:]
print(f"предложений: train {len(train)}, test {len(test)}")

# ---- вариант А: таблица 3-грамм (контроль) ----
table = collections.defaultdict(collections.Counter)
for s in train:
    for i in range(2, len(s)): table[(s[i-2], s[i-1])][s[i]] += 1

# ---- вариант Б: векторная память следующего слова ----
# ключ: bind(C(w₋₂), ρ₂) + bind(C(w₋₁), ρ₁) — порядок через роли; суперпозиция даёт частичное совпадение
# при неизвестной паре (ключ (a,b) частично похож на (x,b) — «back-off» бесплатно, из геометрии)
P1, P2 = "__pos1__", "__pos2__"
def key(w2, w1): return unit(bind(w2, P2) + bind(w1, P1))
vocab = {}
keys, vals = {}, {}          # агрегируем повторы: ключ-строка → (вектор ключа, сумма значений)
t0 = time.time()
for s in train:
    for i in range(2, len(s)):
        k = (s[i-2], s[i-1]); vocab[s[i]] = 1
        if k not in keys: keys[k] = key(*k); vals[k] = np.zeros(D)
        vals[k] += C(s[i])
K = np.stack(list(keys.values())); V = np.stack([unit(v) for v in vals.values()]); klist = list(keys)
words = list(vocab); WM = np.stack([C(w) for w in words])
print(f"векторная память: {len(klist)} ключей, словарь {len(words)}, запись {time.time()-t0:.1f}s")

def predict_vec(w2, w1, topk=1, bias=None):
    q = key(w2, w1); s = K @ q; top = np.argsort(-s)[:5]
    if s[top[0]] > 0.9: v = V[top[0]]                      # точный ключ
    else: v = V[top].T @ np.maximum(s[top], 0)             # back-off: смесь похожих ключей (из геометрии)
    if bias is not None: v = v + 0.3 * bias
    sc = WM @ unit(v); return [words[j] for j in np.argsort(-sc)[:topk]]

def predict_table(w2, w1):
    c = table.get((w2, w1));
    return c.most_common(1)[0][0] if c else None

# ---- оценка top-1 на held-out ----
n = okA = okB = seenA = 0; t0 = time.time()
for s in test:
    for i in range(2, len(s)):
        n += 1; w2, w1, gold = s[i-2], s[i-1], s[i]
        a = predict_table(w2, w1); okA += a == gold; seenA += a is not None
        okB += predict_vec(w2, w1)[0] == gold
print(f"held-out next-word top-1: таблица 3-грамм {okA/n*100:.1f}% (ключ известен в {seenA/n*100:.0f}% случаев) | "
      f"векторная память {okB/n*100:.1f}%  ({n} предсказаний, {time.time()-t0:.1f}s)")

# ---- генерация ----
def generate(prompt, max_len=25, bias_q=None):
    ws = ["<s>", "<s>"] + toks(prompt); bias = mem.sieve.bag(toks(bias_q)) if bias_q else None
    for _ in range(max_len):
        cands = predict_vec(ws[-2], ws[-1], topk=3, bias=bias)
        nxt = next((c for c in cands if c not in ws[-6:]), cands[0])   # не зацикливаться
        if nxt == "</s>": break
        ws.append(nxt)
    return " ".join(ws[2:])

print("--- генерация (ядро говорит само) ---")
for p, q in [("samarkand is", None), ("the capital of uzbekistan", None), ("timur", None),
             ("the economy", None), ("tashkent is", "what is the population of tashkent")]:
    print(f"  «{p}» → {generate(p, bias_q=q)}")
json.dump(dict(train=len(train), test=len(test), keys=len(klist), vocab=len(words),
               table_top1=okA/n, vec_top1=okB/n), open(f"{OUT}/exp26_speech.json", "w"), indent=1)
