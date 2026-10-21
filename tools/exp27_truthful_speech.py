"""Эксперимент 27: ПРАВДИВАЯ РЕЧЬ — факт из памяти + генерация с принуждением (без градиента).
exp26 показал: 3-граммная векторная речь грамматична, но врёт («samarkand is a member of the soviet
union»). Причина: следующий слово выбирается по частоте, факт не участвует.

Новая схема (3 части, всё в одном пространстве D=2048):
  1. РАМКА из предложения (exp18 на Википедии): «X is/was (a|an|the) ... Y» → факт (X, is, Y-фраза);
     «X ... in Y», «X ... of Y» → (X, in, Y). Субъект = первые слова до связки. Факт пишется
     в LongTerm как bind(X, rel) → мешок слов Y-фразы (значение — не одно слово, а фраза).
  2. ВОПРОС → (X, rel): «what/who is X» → (X, is); «where is X» → (X, in). Чтение из памяти → вектор
     ответа A = мешок слов правильной фразы.
  3. РЕЧЬ с принуждением: следующее слово = argmax по словарю от [3-граммная память] + λ·A.
     λ велик (1.0) — слова факта притягиваются; ограничение: грамматика из 3-грамм, факт из памяти.
Метрика: 20 вопросов по прочитанному (ответ-факт известен из рамки), успех = сгенерированный ответ
содержит ключевое слово факта. Контроль: exp26 без принуждения (λ=0).
"""
import sys, os, json, time, re, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_v2 import Memory, toks, content_words, C, bind, unit, D, STOP

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
mem = Memory(path=f"{OUT}/wiki_memory")
raw = mem.L.stext
print(f"знаний: {len(raw)} предложений")

# ---------- 1. РАМКА ----------
COPULA = re.compile(r"^(?P<x>[A-Z][\w'\-]*(?:\s+[A-Z][\w'\-]*){0,3})\s+(?:is|was|are|were)\s+(?:a|an|the|one of the)?\s*(?P<y>[^,.;]{3,80})")
facts = []                       # (x_words, rel, y_words, source sentence)
for s in raw:
    m = COPULA.match(s)
    if not m: continue
    x = toks(m.group("x")); y = content_words(toks(m.group("y")))
    if x and y and x[0] not in STOP: facts.append((x, "is", y, s))
print(f"рамка нашла фактов «X is Y»: {len(facts)}")

REL = "__is__"
FK, FV, FT = [], [], []
for x, rel, y, s in facts:
    FK.append(unit(sum(C(w) for w in x))); FV.append(unit(sum(C(w) for w in y))); FT.append((x, y, s))
FK, FV = np.stack(FK), np.stack(FV)

def fact_vec(subject_words, hint_words=()):
    """Факт по субъекту; если у субъекта много фактов — выбираем ближайший к подсказке (теме вопроса)."""
    q = unit(sum(C(w) for w in subject_words)); sc = FK @ q
    idx = np.where(sc > 0.5)[0]
    if len(idx) == 0: return None, None, 0.0
    if hint_words and len(idx) > 1:
        h = unit(sum(C(w) for w in hint_words)); j = idx[int(np.argmax(FV[idx] @ h))]
    else: j = idx[int(np.argmax(sc[idx]))]
    return FV[j], FT[j], float(sc[j])

# ---------- 3-граммная речь (из exp26) ----------
sents = [["<s>", "<s>"] + toks(s) + ["</s>"] for s in raw if len(toks(s)) >= 4]
P1, P2 = "__pos1__", "__pos2__"
def key(w2, w1): return unit(bind(w2, P2) + bind(w1, P1))
keys, vals = {}, {}; vocab = {}
for s in sents:
    for i in range(2, len(s)):
        k = (s[i-2], s[i-1]); vocab[s[i]] = 1
        if k not in keys: keys[k] = key(*k); vals[k] = np.zeros(D)
        vals[k] += C(s[i])
K = np.stack(list(keys.values())); V = np.stack([unit(v) for v in vals.values()])
words = list(vocab); WM = np.stack([C(w) for w in words]); widx = {w: i for i, w in enumerate(words)}

def next_scores(w2, w1):
    q = key(w2, w1); s = K @ q; top = np.argsort(-s)[:5]
    v = V[top[0]] if s[top[0]] > 0.9 else V[top].T @ np.maximum(s[top], 0)
    return WM @ unit(v)

def generate(prompt_words, A=None, lam=1.0, max_len=14):
    ws = ["<s>", "<s>"] + list(prompt_words)
    bias = (WM @ A) if A is not None else 0
    for _ in range(max_len):
        sc = next_scores(ws[-2], ws[-1]) + lam * bias
        for w in ws[-5:]:
            if w in widx: sc[widx[w]] -= 1.0          # не повторяться
        nxt = words[int(np.argmax(sc))]
        if nxt == "</s>": break
        ws.append(nxt)
    return " ".join(ws[2:])

# ---------- вопросы: 20 РАЗНЫХ субъектов; вопрос = «what is X» + 1 слово-подсказка темы (как человек:
# «what is uzbekistan, про хлопок?») — подсказка ОДНО слово факта, ответ проверяется по ОСТАЛЬНЫМ словам ----------
rng = np.random.default_rng(1)
cand = [f for f in FT if len(f[1]) >= 3 and len(f[0]) <= 2 and f[0][0] not in ("it", "this", "there", "he", "she")]
rng.shuffle(cand); qs, seen = [], set()
for f in cand:
    if tuple(f[0]) in seen: continue
    seen.add(tuple(f[0])); qs.append(f)
    if len(qs) == 20: break
okF = okC = 0; rows = []
print(f"--- {len(qs)} вопросов (разные субъекты): «what is X (hint)?» → речь с принуждением (λ=1) vs без (λ=0) ---")
for x, y, src in qs:
    hint = [y[0]]; rest = [w for w in y[1:] if len(w) > 3][:4] or y[1:]
    A, ft, sc = fact_vec(x, hint)
    prompt = x + ["is"]
    gen_c = generate(prompt, None, 0.0)
    gen_f = generate(prompt, A, 1.0) if A is not None else gen_c
    hitC = any(w in toks(gen_c) for w in rest); hitF = any(w in toks(gen_f) for w in rest)
    okC += hitC; okF += hitF
    print(f"  what is {' '.join(x)} ({hint[0]})?  факт: {' '.join(y[:6])}\n"
          f"     без:  {gen_c}\n     с:    {gen_f}   {'OK' if hitF else 'MISS'}")
    rows.append(dict(x=x, y=y, control=gen_c, forced=gen_f, hitC=hitC, hitF=hitF))
print(f"ИТОГО: содержит слово факта (кроме подсказки) — без принуждения {okC}/{len(qs)}, с принуждением {okF}/{len(qs)}")
json.dump(dict(facts=len(facts), control=okC, forced=okF, rows=rows), open(f"{OUT}/exp27_truthful_speech.json", "w"),
          ensure_ascii=False, indent=1)
