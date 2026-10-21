"""Эксперимент 5: «сито» — слово ↔ числа в обе стороны, без генератора.
Ключи как в exp4 (буквы + сеть). ЗНАЧЕНИЕ — не номер записи, а вектор ответа:
  (a) letters: буквенный хэш строки ответа (то же сито, что для ключей; ничему не учится)
  (b) wte:     сумма эмбеддингов токенов ответа из таблицы сети (сито сети)
Чтение: память отдаёт вектор v̂ (softmax-взвешенная сумма значений — как делала бы сеть, не argmax),
обратный шаг сита: ближайший из словаря ответов (все числа 2…98 → 97 кандидатов) → строка.
Меряем точность «вопрос → строка ответа» при N = 100/300/1000, для списка и для одной Hebb/delta-матрицы.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.transformer import load_model
from mini_ai.memory_core import Encoder, unit
from tools.exp3_wordcode import letter_code
from tools.facts_unique import make_unique, MODEL, OUT

p, cfg, tok, meta = load_model(MODEL)
facts = make_unique(1000)
enc = Encoder(p, cfg, tok, 1)
cache = f"{OUT}/exp5_keys.npz"
if os.path.exists(cache):
    z = np.load(cache); KF, KQ = z["KF"], z["KQ"]
else:
    t0 = time.time()
    KF = np.stack([enc.key(f["fact"]) for f in facts]); KQ = np.stack([enc.key(f["q"]) for f in facts])
    np.savez(cache, KF=KF, KQ=KQ); print(f"encode 2000 фраз: {time.time()-t0:.0f} s", flush=True)
LF = np.stack([unit(letter_code(f["name"])) + unit(letter_code(f["obj"])) for f in facts])

# словарь ответов и два сита
answers = [str(i) for i in range(2, 99)]
wte = np.asarray(p["wte"], dtype=np.float64)


def sieve_letters(s): return unit(letter_code(s))
def sieve_wte(s): return unit(wte[tok.encode(s)].sum(0))


def decode(vhat, table):            # обратный шаг сита: ближайшая строка словаря
    return (unit(vhat) @ table.T).argmax(-1)


res = {}
for name, sieve in (("letters", sieve_letters), ("wte", sieve_wte)):
    table = np.stack([sieve(a) for a in answers])                       # (97, dv)
    cos_tab = (table @ table.T - np.eye(len(answers))).max()
    print(f"\n[{name}] dv={table.shape[1]}, макс косинус между разными ответами {cos_tab:.2f}")
    for N in (100, 300, 1000):
        kf, kq = KF[:N].copy(), KQ[:N].copy()
        for lang in ("uz", "ru", "en"):
            m = np.array([f["lang"] == lang for f in facts[:N]]); kf[m] -= kf[m].mean(0); kq[m] -= kq[m].mean(0)
        A = unit(np.hstack([unit(kf), unit(LF[:N])])); B = unit(np.hstack([unit(kq), unit(LF[:N])]))
        V = np.stack([sieve(f["gold"]) for f in facts[:N]])                 # значения — векторы ответов
        gold = np.array([answers.index(f["gold"]) for f in facts[:N]])
        S = B @ A.T                                                         # (N запросов, N записей)
        out = {}
        for beta in (10, 30, 100):                                          # список: softmax-чтение → v̂
            w = np.exp(beta * (S - S.max(1, keepdims=True))); w /= w.sum(1, keepdims=True)
            out[f"list_b{beta}"] = (decode(w @ V, table) == gold).mean()
        W = V.T @ A                                                         # Hebb: одна матрица
        out["hebb"] = (decode((W @ B.T).T, table) == gold).mean()
        W = np.zeros_like(W)                                                # delta-rule, 3 прохода
        for _ in range(3):
            for i in range(N): W += 0.5 * np.outer(V[i] - W @ A[i], A[i])
        out["delta"] = (decode((W @ B.T).T, table) == gold).mean()
        res[f"{name}_{N}"] = {k: float(v) for k, v in out.items()}
        print(f"N={N:4}: " + "  ".join(f"{k}={v*100:5.1f}%" for k, v in out.items()), flush=True)
json.dump(res, open(f"{OUT}/exp5_sieve.json", "w"), indent=1)
