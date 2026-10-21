"""Эксперимент 6: цепочка вопросов к памяти (multi-hop) без градиента.
Факты: «друг X — Y» (отношение) и «у X N предметов» (число). Вопрос «сколько olma у друга Бека?»
не лежит в памяти целиком: нужно спросить дважды. Ответ 1-го шага (имя) через сито наоборот → строка,
из строки сито → новый ключ → 2-й шаг. Так же 3 шага («друг друга»).
Ключ = буквы (exp5: сеть не нужна). Значение = буквы ответа. Чтение softmax β=100 (exp5).
Контроллер (кто решает, что спросить) здесь — жёсткий план из 2–3 шагов; измеряем только, доживает ли
сигнал до конца цепочки, когда каждый шаг проходит через сито туда и обратно.
"""
import sys, os, json, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_core import unit
from tools.exp3_wordcode import letter_code
from tools.facts_unique import make_unique, OUT

OBJ = ["olma", "kitob", "qalam"]


class Memory:
    def __init__(self): self.K, self.V, self.S = [], [], []
    def write(self, key_words, answer):
        self.K.append(unit(sum(unit(letter_code(w)) for w in key_words))); self.V.append(unit(letter_code(answer))); self.S.append(answer)
    def read(self, key_words, beta=100):
        K = np.stack(self.K); q = unit(sum(unit(letter_code(w)) for w in key_words))
        s = K @ q; w = np.exp(beta * (s - s.max())); w /= w.sum()
        return w @ np.stack(self.V)


def sieve_back(v, vocab, table): return vocab[int((unit(v) @ table.T).argmax())]


res = {}
for N in (100, 300, 1000):
    facts = make_unique(N); names = [f["name"] for f in facts]
    rng = random.Random(7); perm = names[:]; rng.shuffle(perm)
    friend = {a: b for a, b in zip(names, perm)}                   # друг X — Y (перестановка, без неподвижных точек не требуем)
    count = {(f["name"], o): str(rng.randrange(2, 99)) for f in facts for o in OBJ}
    mem = Memory()
    for a, b in friend.items(): mem.write([a, "dost"], b)
    for (a, o), n in count.items(): mem.write([a, o], n)
    vocab_names = names; T_names = np.stack([unit(letter_code(n)) for n in vocab_names])
    vocab_nums = [str(i) for i in range(2, 99)]; T_nums = np.stack([unit(letter_code(n)) for n in vocab_nums])
    ok1 = ok2 = ok3 = 0
    for a in names:
        o = rng.choice(OBJ)
        h1 = sieve_back(mem.read([a, "dost"]), vocab_names, T_names)            # шаг 1: кто друг?
        ok1 += h1 == friend[a]
        h2 = sieve_back(mem.read([h1, o]), vocab_nums, T_nums)                  # шаг 2: сколько у него?
        ok2 += h2 == count[(friend[a], o)]
        f2 = sieve_back(mem.read([h1, "dost"]), vocab_names, T_names)           # шаг 3: друг друга → сколько у него?
        h3 = sieve_back(mem.read([f2, o]), vocab_nums, T_nums)
        ok3 += h3 == count[(friend[friend[a]], o)]
    r = {"records": len(mem.S), "hop1": ok1 / N, "hop2": ok2 / N, "hop3": ok3 / N}
    res[N] = r
    print(f"N={N:4} имён, записей {r['records']:4}: 1 шаг {r['hop1']*100:5.1f}%  2 шага {r['hop2']*100:5.1f}%  3 шага {r['hop3']*100:5.1f}%", flush=True)
json.dump(res, open(f"{OUT}/exp6_multihop.json", "w"), indent=1)
