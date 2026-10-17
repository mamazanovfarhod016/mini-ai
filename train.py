"""
Обучение Mini AI.

    .venv/bin/python train.py

1. Читает data/intents.json (фразы-примеры для каждого намерения).
2. Делит фразы на train (80%) и test (20%) — тест сеть НЕ видит при обучении.
3. Добавляет к train «искусственные опечатки» (аугментация).
4. Обучает MLP (backprop + Adam), печатает точность на тесте и матрицу ошибок.
5. Переобучает на ВСЕХ фразах и сохраняет веса + словарь в model.json
   (этот файл потом едет в телефон).
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

from mini_ai.features import fnv1a, normalize, tokens, vectorize
from mini_ai.mlp import MLP, Adam

ROOT = Path(__file__).resolve().parent
DIM = 2048           # длина входного вектора (ячеек хеша)
HIDDEN = 32          # нейронов в скрытом слое
EPOCHS = 30
BATCH = 32
LR = 0.003
L2 = 1e-4
THRESHOLD = 0.45     # ниже этой уверенности — «не знаю»
SCALE = 1000         # веса храним целыми числами: round(w * 1000)
SEED = 42

RU_KEYS = "йцукенгшщзхфывапролджэячсмитьбю"
LAT_KEYS = "qwertyuiopasdfghjklzxcvbnm"


def typo(phrase, rng):
    """Одна случайная опечатка: пропуск, повтор, замена или перестановка буквы."""
    s = list(phrase)
    letters = [i for i, c in enumerate(s) if c.isalpha()]
    if len(letters) < 3:
        return phrase
    i = rng.choice(letters)
    kind = rng.randrange(4)
    alphabet = RU_KEYS if "а" <= s[i] <= "я" else LAT_KEYS
    if kind == 0:
        del s[i]
    elif kind == 1:
        s.insert(i, s[i])
    elif kind == 2:
        s[i] = rng.choice(alphabet)
    elif i + 1 < len(s) and s[i + 1].isalpha():
        s[i], s[i + 1] = s[i + 1], s[i]
    return "".join(s)


def drop_word(phrase, rng):
    """Выкидываем одно случайное слово: сеть учится не зависеть от одного слова."""
    w = phrase.split()
    if len(w) < 3:
        return phrase
    del w[rng.randrange(len(w))]
    return " ".join(w)


FILLERS_RU = ["ну", "а", "слушай", "бот", "друг", "пожалуйста", "эй", "скажи"]
FILLERS_UZ = ["bot", "do'stim", "iltimos", "hoy", "ayt", "endi"]


def add_filler(phrase, rng):
    """Добавляем «слово-паразит» в начало или конец: «ну как дела», «salom bot»."""
    lat = sum("a" <= c <= "z" for c in phrase) > sum("а" <= c <= "я" for c in phrase)
    f = rng.choice(FILLERS_UZ if lat else FILLERS_RU)
    return f + " " + phrase if rng.random() < 0.5 else phrase + " " + f


def augment(samples, rng, n_copies=3):
    out = list(samples)
    for text, y in samples:
        for _ in range(n_copies):
            out.append((typo(text, rng), y))
        out.append((drop_word(text, rng), y))
        out.append((add_filler(text, rng), y))
    return out


def known_hashes(texts):
    """Хеши всех признаков, встреченных при обучении (без опечаток).
    Нужны, чтобы понять, что фраза совсем незнакомая → честное «не знаю»."""
    return sorted({fnv1a(f) for t in texts for f in tokens(t)})


def to_matrix(samples, dim=None):
    dim = dim or DIM
    X = np.array([vectorize(t, dim) for t, _ in samples])
    y = np.array([lbl for _, lbl in samples])
    return X, y


def train(samples, n_classes, seed=SEED, verbose=True):
    X, y = to_matrix(samples)
    net = MLP(X.shape[1], HIDDEN, n_classes, seed=seed)
    opt = Adam(net.params, lr=LR)
    rng = np.random.default_rng(seed)
    for epoch in range(1, EPOCHS + 1):
        idx = rng.permutation(len(X))          # перемешиваем каждую эпоху
        total = 0.0
        for s in range(0, len(X), BATCH):
            b = idx[s:s + BATCH]
            loss, cache = net.loss(X[b], y[b], L2)      # прямой проход
            grads = net.backward(cache, y[b], L2)       # обратный проход
            opt.step(net.params, grads)                 # шаг Adam
            total += loss * len(b)
        if verbose and (epoch % 10 == 0 or epoch == 1):
            acc = (net.predict_proba(X).argmax(1) == y).mean()
            print(f"  эпоха {epoch:3d}  loss={total / len(X):.4f}  train acc={acc:.3f}")
    return net


def split(intents, rng):
    """Стратифицированный сплит: из каждого намерения 20% фраз — в тест."""
    train_s, test_s = [], []
    for k, it in enumerate(intents):
        ps = sorted(set(normalize(p) for p in it["patterns"]))
        rng.shuffle(ps)
        n_test = max(2, round(len(ps) * 0.2))
        test_s += [(p, k) for p in ps[:n_test]]
        train_s += [(p, k) for p in ps[n_test:]]
    return train_s, test_s


def confusion(net, X, y, tags):
    K = len(tags)
    M = np.zeros((K, K), dtype=int)
    for t, p in zip(y, net.predict_proba(X).argmax(1)):
        M[t, p] += 1
    return M


def main():
    data = json.loads((ROOT / "data/intents.json").read_text(encoding="utf-8"))
    intents = data["intents"]
    tags = [it["tag"] for it in intents]
    rng = random.Random(SEED)

    # ---------- 0. точность на 3 разных случайных сплитах (честнее, чем один) ----------
    accs, chat_accs = [], []
    for seed in (7, 1):
        tr, te = split(intents, random.Random(seed))
        n = train(augment(tr, random.Random(seed)), len(tags), seed=seed, verbose=False)
        Xs, ys = to_matrix(te)
        ps = n.predict_proba(Xs).argmax(1)
        accs.append(float((ps == ys).mean()))
        chat = ys != tags.index("out_of_scope")
        chat_accs.append(float((ps[chat] == ys[chat]).mean()))

    # ---------- 1. честная проверка на отложенных фразах ----------
    train_s, test_s = split(intents, rng)
    train_aug = augment(train_s, rng)
    print(f"Намерений: {len(tags)}, фраз train: {len(train_s)} (+опечатки = {len(train_aug)}), test: {len(test_s)}")
    print(f"Вход: {DIM} (хеш слов + триграмм) → скрытый слой {HIDDEN} ReLU → {len(tags)} softmax")
    net = train(train_aug, len(tags))
    Xt, yt = to_matrix(test_s)
    pred = net.predict_proba(Xt).argmax(1)
    test_acc = float((pred == yt).mean())
    print(f"\nТОЧНОСТЬ НА ТЕСТЕ (сплит seed={SEED}): {test_acc * 100:.1f}%  ({int((pred == yt).sum())}/{len(yt)})")
    accs.insert(0, test_acc)
    chat = yt != tags.index("out_of_scope")
    chat_accs.insert(0, float((pred[chat] == yt[chat]).mean()))
    chat_mean = sum(chat_accs) / len(chat_accs)
    mean_acc = sum(accs) / len(accs)
    print("Точность на 3 разных сплитах: " + ", ".join(f"{a * 100:.1f}%" for a in accs) + f"  → среднее {mean_acc * 100:.1f}%")
    print("  только «разговорные» намерения (без out_of_scope): " + ", ".join(f"{a * 100:.1f}%" for a in chat_accs)
          + f"  → среднее {chat_mean * 100:.1f}%")

    M = confusion(net, Xt, yt, tags)
    print("\nМатрица ошибок (строка = правильный ответ, столбец = ответ сети; показаны только ошибки):")
    errors = []
    for i in range(len(tags)):
        for j in range(len(tags)):
            if i != j and M[i, j]:
                errors.append(f"  {tags[i]:>15} → {tags[j]:<15} ×{M[i, j]}")
    print("\n".join(errors) or "  ошибок нет")
    print("\nОшибочные фразы:")
    for (t, k), p in zip(test_s, pred):
        if k != p:
            print(f"  «{t}»: нужно {tags[k]}, сеть сказала {tags[p]}")

    # Полная матрица — в файл (для README / анализа)
    lines = ["true\\pred," + ",".join(tags)]
    lines += [tags[i] + "," + ",".join(map(str, M[i])) for i in range(len(tags))]
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs/confusion_matrix.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---------- 2. финальная модель на всех фразах ----------
    all_s = train_s + test_s
    all_aug = augment(all_s, random.Random(SEED + 1))
    print(f"\nФинальное обучение на всех {len(all_s)} фразах (+опечатки = {len(all_aug)})")
    net = train(all_aug, len(tags), verbose=False)

    # Квантизация: храним round(w*1000) — целые числа короче в JSON.
    # Перед использованием делим обратно на SCALE (и Python, и JS).
    r = lambda a: [int(round(float(v) * SCALE)) for v in np.asarray(a).ravel()]
    model = {
        "format": "mini-ai/1",
        "dim": DIM,
        "hidden": HIDDEN,
        "scale": SCALE,
        "threshold": THRESHOLD,
        "test_accuracy": round(mean_acc, 4),
        "test_accuracy_splits": [round(a, 4) for a in accs],
        "test_accuracy_chat": round(chat_mean, 4),
        "n_params": net.n_params(),
        "intents": tags,
        "responses": {it["tag"]: it["responses"] for it in intents},
        "labels": {it["tag"]: it["description"] for it in intents},
        "known": known_hashes(t for t, _ in all_s),
        "W1": r(net.params["W1"]),
        "b1": r(net.params["b1"]),
        "W2": r(net.params["W2"]),
        "b2": r(net.params["b2"]),
    }
    text = json.dumps(model, ensure_ascii=False, separators=(",", ":"))
    (ROOT / "model.json").write_text(text, encoding="utf-8")
    # Копия для приложения (JS читает её из assets, офлайн)
    (ROOT / "app/www/model.js").write_text("window.MINI_AI_MODEL=" + text + ";\n", encoding="utf-8")
    print(f"Параметров: {net.n_params():,}. model.json: {len(text.encode()) / 1024:.0f} КБ")
    return chat_mean


if __name__ == "__main__":
    acc = main()
    sys.exit(0 if acc >= 0.9 else 1)
