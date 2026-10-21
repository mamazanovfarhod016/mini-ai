"""Эксперимент 4: настоящая ёмкость памяти, когда потолок генератора = 100 % (уникальные имена).
Имена генерируем из слогов (seed), чтобы 1000 фактов были различимы. Ключ = буквы + сеть (победитель exp3).
Сравниваем list (softmax/argmax по косинусу — «modern Hopfield») и hebb (одна матрица) при N = 100…1000.
Плюс: средний косинус между ключами и зазор Δ (правильный − лучший неправильный) — то, что просил THEORY.md.
"""
import sys, os, json, time, random, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.transformer import load_model
from mini_ai.memory_core import Encoder, unit
from tools.exp3_wordcode import letter_code
from train_transformer import clean_question

MODEL = "/home/hermes/Developer/mini-ai/model_transformer_school.json"
OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
p, cfg, tok, meta = load_model(MODEL)
SYL = ["ba", "zo", "ra", "mi", "do", "sha", "ke", "nu", "li", "to", "far", "jon", "gul", "nur", "sher", "bek", "din", "ol", "ser", "ka"]


def name_at(i, rng):
    n = "".join(rng.choice(SYL) for _ in range(rng.choice([2, 3])))
    return n[0].upper() + n[1:]


def make_unique(N, seed=11):
    rng = random.Random(seed); seen = set(); facts = []
    while len(facts) < N:
        nm = name_at(len(facts), rng)
        if nm in seen: continue
        seen.add(nm)
        lang = ["uz", "ru", "en"][len(facts) % 3]; num = rng.randrange(2, 99)
        obj = rng.choice(["olma", "kitob", "qalam"])
        fact = {"uz": f"{nm}da {num} ta {obj} bor.", "ru": f"У {nm} {num} {obj}.", "en": f"{nm} has {num} {obj}."}[lang]
        q = {"uz": f"{nm}da nechta {obj} bor?", "ru": f"Сколько {obj} у {nm}?", "en": f"How many {obj} does {nm} have?"}[lang]
        facts.append({"name": nm, "obj": obj, "fact": fact, "q": q, "gold": str(num), "lang": lang})
    return facts


facts = make_unique(1000)
enc = Encoder(p, cfg, tok, 1)
t0 = time.time()
KF = np.stack([enc.key(f["fact"]) for f in facts]); KQ = np.stack([enc.key(f["q"]) for f in facts])
print(f"encode 2000 фраз: {time.time()-t0:.0f} s", flush=True)
LF = np.stack([unit(letter_code(f["name"])) + unit(letter_code(f["obj"])) for f in facts])
LQ = LF.copy()  # имя и предмет в вопросе те же — код слова не зависит от фразы
res = {}
for N in (100, 300, 1000):
    kf, kq = KF[:N].copy(), KQ[:N].copy()
    for lang in ("uz", "ru", "en"):
        m = np.array([f["lang"] == lang for f in facts[:N]])
        kf[m] -= kf[m].mean(0); kq[m] -= kq[m].mean(0)
    A = unit(np.hstack([unit(kf), unit(LF[:N])])); B = unit(np.hstack([unit(kq), unit(LQ[:N])]))
    S = A @ B.T                                  # list: softmax/argmax по косинусу
    pred = S.argmax(0); acc_list = (pred == np.arange(N)).mean()
    # зазор Δ: правильный − лучший неправильный
    diag = S[np.arange(N), np.arange(N)]; S2 = S.copy(); S2[np.arange(N), np.arange(N)] = -9
    delta = diag - S2.max(0)
    # средний косинус между ключами (анизотропия)
    KK = A @ A.T; mean_cos = (KK.sum() - N) / (N * (N - 1))
    # Hebb: W = Σ e_i k_iᵀ → читаем W·q — то же самое, что список, пока значения one-hot. Честный Hebb —
    # когда ЗНАЧЕНИЕ тоже вектор: W = Σ v_i k_iᵀ, v_i = случайный код ответа; читаем v̂ = W q, ищем ближайший код.
    rng = np.random.default_rng(0); Vcodes = unit(rng.standard_normal((N, A.shape[1])))
    W = Vcodes.T @ A                             # (dv, dk)
    vhat = (W @ B.T).T                           # (N, dv)
    pred_h = (unit(vhat) @ Vcodes.T).argmax(1); acc_hebb = (pred_h == np.arange(N)).mean()
    res[N] = {"list": acc_list, "hebb": acc_hebb, "mean_cos": mean_cos, "delta_mean": float(delta.mean()), "delta_min": float(delta.min())}
    print(f"N={N:4}: list={acc_list*100:5.1f}%  hebb={acc_hebb*100:5.1f}%  | cos между ключами {mean_cos:.3f}, зазор Δ ср {delta.mean():.3f} мин {delta.min():.3f}", flush=True)
json.dump(res, open(f"{OUT}/exp4_capacity.json", "w"), indent=1)
