"""Эксперимент 47: ДВА ШАГА на школьных фактах — мост через общий термин, с проверкой связности.
Рамка (1 шаг) требует все слова вопроса в одном факте: precision 86%, покрытие 15%.
Два шага: вопрос → ЧАСТЬ слов покрыта фактом F1 → F1 содержит термин T (не из вопроса, редкий) →
          факт F2 содержит T и ОСТАЛЬНЫЕ слова вопроса → ответ из F2 (или F1).
Условие связности: F1 ∪ F2 покрывают ВСЕ обязательные слова вопроса (как 1 шаг, но суммарно),
и мост T — содержательное слово (IDF > 2.5), общее для F1 и F2. Это та же логика exp9 (две записи,
общая сущность), но на открытом тексте.
Сравнение: 1 шаг / 1+2 шага / + мешок при отказе. Метрики те же: precision, покрытие, общая точность.
"""
import sys, os, json, time, re, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
exec(open(os.path.join(os.path.dirname(__file__), "exp45_frame_learn.py")).read().split("rng = np.random.default_rng(0); rows = []")[0])
OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"

def fact_words(f): return f[0] | f[2]
def cover(ws, known): return sum(idf.get(w, 1.0) for w in known & ws)

def two_step(q, max_f1=60, max_t=8):
    fr = parse_question(q); known = set(fr["Y"] if fr["slot"] == "X" else fr["X"])
    if len(known) < 2: return None
    kidf = sum(idf.get(w, 1.0) for w in known)
    # F1: факты, покрывающие ≥ 40% веса вопроса (но не все — иначе это 1 шаг)
    cnt = collections.Counter()
    for w in known:
        for k in by_word.get(w, []): cnt[k] += idf.get(w, 1.0)
    f1s = [(c / kidf, k) for k, c in cnt.most_common(max_f1) if 0.4 <= c / kidf < 0.99]
    best = None
    for c1, k1 in f1s:
        F1 = facts_idx[k1]; w1 = fact_words(F1); rest = known - w1
        bridges = sorted((t for t in w1 - known if idf.get(t, 0) > 2.5), key=lambda t: -idf[t])[:max_t]
        for t in bridges:
            for k2 in by_word.get(t, []):
                if k2 == k1: continue
                F2 = facts_idx[k2]; w2 = fact_words(F2)
                if not rest <= w2: continue                       # F2 должен закрыть остаток вопроса
                # ответ: часть F2, где нет слов вопроса и нет моста; иначе часть F1
                for F, part_x, part_y in ((F2, F2[3], F2[4]), (F1, F1[3], F1[4])):
                    for ans, aw in ((part_x, F[0]), (part_y, F[2])):
                        if aw and not (aw & known) and ans.lower() not in ("it", "this", "they", "these"):
                            s = c1 + (cover(w2, rest) / kidf) + 0.2 * (idf[t] / 6)
                            if not best or s > best[0]: best = (s, ans, t, F1, F2)
    return best

def pick2(q, opts):
    b1, fr = frame_answer(q)
    if b1:
        s, ans, fact = b1; aw = set(toks(ans)) | set(toks(fact[0] + " " + fact[2]))
        sc = [len(set(content_words(toks(o))) & aw) / max(1, len(content_words(toks(o)))) for o in opts]
        if max(sc) > 0: return int(np.argmax(sc)), "1 шаг"
    b2 = two_step(q)
    if b2:
        s, ans, t, F1, F2 = b2; aw = set(toks(ans)) | fact_words(F2) | fact_words(F1)
        aw_ans = set(toks(ans))
        sc = [len(set(content_words(toks(o))) & aw_ans) / max(1, len(content_words(toks(o)))) for o in opts]
        if max(sc) == 0: sc = [len(set(content_words(toks(o))) & aw) / max(1, len(content_words(toks(o)))) for o in opts]
        if max(sc) > 0: return int(np.argmax(sc)), f"2 шага через «{t}»"
    return None, "не знаю"

rng = np.random.default_rng(0); rows = []
for _, row in test.iterrows():
    opts = [row["correct_answer"], row["distractor1"], row["distractor2"], row["distractor3"]]
    perm = rng.permutation(4); sh = [opts[j] for j in perm]; gold = int(np.where(perm == 0)[0][0])
    rows.append((row["question"], sh, gold, str(row["support"])))

stat = collections.Counter(); okC = 0; t0 = time.time(); shown = 0
for q, sh, gold, sup in rows:
    p, how = pick2(q, sh); key = how.split(" через")[0]
    if p is None: c = method_C(q, sh); okC += c == gold; stat["отказ"] += 1
    else:
        stat[key + " всего"] += 1; stat[key + " верно"] += p == gold; okC += p == gold
        if key == "2 шага" and shown < 6:
            shown += 1; b2 = two_step(q)
            print(f"  {'✓' if p == gold else '✗'} {q}\n      F1: {b2[3][3]} —{b2[3][1]}→ {b2[3][4][:60]}\n      мост «{b2[2]}» → F2: {b2[4][3]} —{b2[4][1]}→ {b2[4][4][:60]}\n      → «{sh[p]}»{'' if p == gold else '  (gold: ' + sh[gold] + ')'}")
n = len(rows)
print(f"\n1 шаг: {stat['1 шаг верно']}/{stat['1 шаг всего']} (precision {stat['1 шаг верно']/max(1,stat['1 шаг всего'])*100:.0f}%)")
print(f"2 шага: {stat['2 шага верно']}/{stat['2 шага всего']} (precision {stat['2 шага верно']/max(1,stat['2 шага всего'])*100:.0f}%)")
print(f"отказов {stat['отказ']} → мешок; ИТОГО {okC}/{n} = {okC/n*100:.1f}% (exp45 1 шаг + мешок: 65.3%), {(time.time()-t0)/n*1000:.0f} мс/вопрос")
json.dump(dict(stat), open(f"{OUT}/exp47_two_step.json", "w"), ensure_ascii=False, indent=1)
