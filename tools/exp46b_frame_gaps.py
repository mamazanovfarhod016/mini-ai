import sys, os; sys.path.insert(0, '.')
src = open("tools/exp45_frame_learn.py").read().split("rng = np.random.default_rng(0); rows = []")[0]
exec(src)
from mini_ai.question import parse_question
import random; random.seed(2)
rows = [(r["question"], r["correct_answer"], str(r["support"])) for _, r in test.iterrows()]
cat = collections.Counter(); shown = 0
for q, gold, sup in random.sample(rows, 300):
    fr = parse_question(q); known = set(fr["Y"] if fr["slot"] == "X" else fr["X"])
    # есть ли в памяти факт, содержащий gold И хоть что-то из вопроса?
    gw = set(content_words(toks(gold))); qw = set(content_words(toks(q)))
    has_fact = any(gw & (f[0] | f[2]) and len(qw & (f[0] | f[2])) >= 2 for f in facts_idx)
    cand = set.intersection(*[set(by_word.get(w, [])) for w in known]) if known else set()
    if not known: c = "рамка пустая (не разобрал вопрос)"
    elif not cand: c = "нет факта со ВСЕМИ словами вопроса" + (" (но факт с gold есть)" if has_fact else " (и факта с gold нет)")
    else:
        res = answer_by_frame(fr, [facts_idx[k] for k in cand], idf); c = "ответил" if res else "кандидаты есть, рамка отвергла"
    cat[c] += 1
    if "но факт с gold есть" in c and shown < 8:
        shown += 1; f = next(f for f in facts_idx if gw & (f[0] | f[2]) and len(qw & (f[0] | f[2])) >= 2)
        print(f"Q: {q}\n   рамка: slot={fr['slot']} known={sorted(known)}\n   факт с gold: {f[3]} —{f[1]}→ {f[4][:70]}\n   не хватает слов: {sorted(known - (f[0] | f[2]))}\n")
print(dict(cat))
