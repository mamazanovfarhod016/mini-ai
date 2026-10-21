"""Эксперимент 44b: разбор ошибок метода C (связь) на SciQ — глазами, 30 штук, по категориям."""
import sys, os, json, re, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
exec(open(os.path.join(os.path.dirname(__file__), "exp44_sciq.py")).read().split("rng = np.random.default_rng(0); res = {}")[0])

rng = np.random.default_rng(0); errs = []; nq = 0
for _, row in test.iterrows():
    opts = [row["correct_answer"], row["distractor1"], row["distractor2"], row["distractor3"]]
    perm = rng.permutation(4); shuffled = [opts[j] for j in perm]; gold = int(np.where(perm == 0)[0][0])
    qw = set(content_words(toks(row["question"]))); sc = [joint_score(qw, set(content_words(toks(o)))) for o in shuffled]
    pred = int(np.argmax(sc)) if max(sc) > 0 else -1
    if pred != gold: errs.append((row["question"], shuffled, gold, pred, sc, row["support"]))
print(f"ошибок {len(errs)}/{len(test)}")
cat = collections.Counter()
for q, opts, gold, pred, sc, sup in errs[:30]:
    gold_in_mem = sc[gold] > 0; zero = max(sc) == 0
    gw = set(content_words(toks(opts[gold]))); sup_w = set(content_words(toks(str(sup))))
    gold_in_support = bool(gw & sup_w)
    c = "нет связи ни у одного" if zero else ("gold не связан, дистрактор связан" if not gold_in_mem else "оба связаны, дистрактор сильнее")
    cat[c] += 1
    print(f"\nQ: {q}\n   gold: «{opts[gold]}» (счёт {sc[gold]:.1f})   выбрано: «{opts[pred] if pred>=0 else '-'}» (счёт {max(sc):.1f})   [{c}]\n   support: {str(sup)[:160]}")
print("\nКАТЕГОРИИ:", dict(cat))
