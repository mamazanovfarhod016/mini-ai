"""Эксперимент 7: смысл = связи. Без жёсткого плана: запрос — слияние кодов слов; память находит, с чем
связана каждая часть, подставляет найденное на место части, сливает снова — пока не дойдёт до ответа.
Выведенное записывает обратно как НОВЫЙ факт (которого никто не говорил) и потом читает его в один шаг.
Факты как в exp6: «X dost → Y», «X obj → N». Запрос: [X, dost, obj] = «сколько obj у друга X» —
такой записи нет; «dost dost» — друг друга.
"""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_core import unit
from tools.exp3_wordcode import letter_code
from tools.facts_unique import make_unique, OUT

OBJ = ["olma", "kitob", "qalam"]
code = lambda w: unit(letter_code(w))


class Memory:
    def __init__(self): self.K, self.V, self.S, self.KW = [], [], [], []
    def write(self, words, answer):
        self.K.append(unit(sum(code(w) for w in words))); self.V.append(code(answer)); self.S.append(answer); self.KW.append(tuple(words))
    def best(self, words):
        """Ближайшая запись к слиянию слов: (сходство, слова записи, ответ)."""
        s = np.stack(self.K) @ unit(sum(code(w) for w in words)); i = int(s.argmax())
        return float(s[i]), self.KW[i], self.S[i]


def solve(mem, words, thr=0.9, max_steps=5):
    """Нет плана. Пока слияние всего запроса не совпадает с записью (сходство < thr):
    ищем, какая ПАРА слов запроса совпадает с записью лучше всего, заменяем пару её ответом, сливаем снова."""
    words = list(words); trace = []
    for _ in range(max_steps):
        s, kw, ans = mem.best(words)
        if s >= thr: trace.append((tuple(words), ans)); return ans, trace
        cand = []
        for i in range(len(words)):
            for j in range(len(words)):
                if i < j:
                    s2, kw2, a2 = mem.best([words[i], words[j]])
                    cand.append((s2, i, j, a2))
        s2, i, j, a2 = max(cand)
        if s2 < thr: return None, trace
        trace.append(((words[i], words[j]), a2))
        rest = [w for k, w in enumerate(words) if k not in (i, j)]
        words = [a2] + rest
    return None, trace


res = {}
for N in (100, 300, 1000):
    facts = make_unique(N); names = [f["name"] for f in facts]
    rng = random.Random(7); perm = names[:]; rng.shuffle(perm); friend = dict(zip(names, perm))
    count = {(f["name"], o): str(rng.randrange(2, 99)) for f in facts for o in OBJ}
    mem = Memory()
    for a, b in friend.items(): mem.write([a, "dost"], b)
    for (a, o), n in count.items(): mem.write([a, o], n)
    base = len(mem.S)
    ok2 = ok3 = new_ok = 0; steps2 = []
    derived = []
    for a in names:
        o = rng.choice(OBJ)
        ans, tr = solve(mem, [a, "dost", o]); ok2 += ans == count[(friend[a], o)]; steps2.append(len(tr))
        if ans is not None: derived.append(([a, "dost", o], ans, count[(friend[a], o)]))
        ans3, _ = solve(mem, [a, "dost", "dost", o]); ok3 += ans3 == count[(friend[friend[a]], o)]
    for words, ans, gold in derived: mem.write(words, ans)          # познанное → в память
    for words, ans, gold in derived:                                # теперь читается в ОДИН шаг?
        s, kw, a1 = mem.best(words); new_ok += (a1 == gold and s >= 0.9)
    r = {"records": base, "hop2_noplan": ok2 / N, "hop3_noplan": ok3 / N, "avg_steps2": float(np.mean(steps2)),
         "derived_written": len(derived), "derived_1step": new_ok / max(1, len(derived)), "old_intact": None}
    # старые факты не испорчены новыми записями?
    r["old_intact"] = float(np.mean([mem.best([a, o])[2] == count[(a, o)] for a in names for o in OBJ]))
    res[N] = r
    print(f"N={N:4} ({base} записей): 2 шага без плана {r['hop2_noplan']*100:5.1f}% (ср. {r['avg_steps2']:.1f} шага)  "
          f"3 шага {r['hop3_noplan']*100:5.1f}%  | записано новых {len(derived)}, читаются в 1 шаг {r['derived_1step']*100:5.1f}%, старые целы {r['old_intact']*100:5.1f}%", flush=True)
json.dump(res, open(f"{OUT}/exp7_meaning_links.json", "w"), indent=1)
