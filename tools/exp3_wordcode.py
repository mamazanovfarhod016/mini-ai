"""Эксперимент 3: «вес слова как конструкция, не как градиент» (идея Фархода 20.10).
Вектор слова собираем БЕЗ обучения:
  letters  — хэш символьных n-грамм (2–3 буквы) в d измерений (Bloom/hashing trick, как fastText-subword)
  contrast — hidden-state сети минус средний «шаблон» (содержание без формы) — из exp2
  concat   — склейка обоих (буквенный код + смысл из сети)
Ключ факта = код имени (+ код предмета). Ключ вопроса — то же. Память — список/Hebb. Экзамен 100 фактов.
Отдельно: сколько уникальных имён — чтобы знать потолок, и acc при N=300 (интерференция).
"""
import sys, os, json, time, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.transformer import load_model
from mini_ai.oneshot_bench import make_facts
from mini_ai.memory_core import Encoder, unit
from train_transformer import clean_question

MODEL = "/home/hermes/Developer/mini-ai/model_transformer_school.json"
OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
p, cfg, tok, meta = load_model(MODEL)
D = 192


def letter_code(word, d=D, ns=(2, 3)):
    """Хэш n-грамм букв → вектор d (±1 по хэшу). Разные слова → почти ортогональные векторы,
    похожие по буквам — похожие векторы. Никакого обучения."""
    w = f"<{clean_question(word)}>"
    v = np.zeros(d)
    for n in ns:
        for i in range(len(w) - n + 1):
            h = zlib.crc32(w[i:i + n].encode())
            v[h % d] += 1.0 if (h >> 16) & 1 else -1.0
    return v


def content_words(f):
    """Слова-носители содержания: имя (+ предмет для count). Вопрос содержит те же слова — проверяем честно."""
    name = f["name"]
    obj = None
    if f["kind"] == "count":
        # предмет — последнее «вещественное» слово факта
        obj = {"uz": f["fact"].split()[-1].strip("."), "ru": f["fact"].split()[-1].strip("."),
               "en": f["fact"].split()[-1].strip(".")}[f["lang"]]
    return name, obj


def key_letters(text, words):
    v = np.zeros(D)
    for w in words:
        if w and clean_question(w) in clean_question(text):
            v += unit(letter_code(w))
    return v


def run(N, seed=7):
    facts = make_facts(N, seed)
    enc = Encoder(p, cfg, tok, 1)
    # contrast-ключи по всей фразе (из exp2: лучший «сетевой» признак)
    KF = np.stack([enc.key(f["fact"], "mean") for f in facts]); KQ = np.stack([enc.key(f["q"], "mean") for f in facts])
    for lang in ("uz", "ru", "en"):
        for kind in ("count", "city"):
            m = np.array([f["lang"] == lang and f["kind"] == kind for f in facts])
            if m.sum() > 1:
                KF[m] -= KF[m].mean(0); KQ[m] -= KQ[m].mean(0)
    LF = np.stack([key_letters(f["fact"], content_words(f)) for f in facts])
    LQ = np.stack([key_letters(f["q"], content_words(f)) for f in facts])
    out = {}
    for label, A, B in (("contrast", KF, KQ), ("letters", LF, LQ),
                        ("concat", np.hstack([unit(KF), unit(LF)]), np.hstack([unit(KQ), unit(LQ)]))):
        S = unit(A) @ unit(B).T
        pred = S.argmax(0)
        acc = (pred == np.arange(N)).mean()
        # Hebb: та же матрица, но суммируем ключи одинаковых «значений» — здесь значения уникальны → равно списку
        out[label] = acc
    uniq = len({(f["lang"], f["name"], content_words(f)[1]) for f in facts})
    return out, uniq


res = {}
for N in (30, 100, 300):
    t0 = time.time(); out, uniq = run(N)
    res[N] = {**out, "uniq_keys": uniq}
    print(f"N={N:3}: " + "  ".join(f"{k}={v*100:5.1f}%" for k, v in out.items()) +
          f"  | уникальных (имя+предмет) {uniq}/{N} = потолок {uniq/N*100:.0f}%  ({time.time()-t0:.0f} s)", flush=True)
json.dump(res, open(f"{OUT}/exp3_wordcode.json", "w"), indent=1)
