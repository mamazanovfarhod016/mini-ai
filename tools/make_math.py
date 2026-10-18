"""
Примеры по арифметике, чтобы Mini AI училась СЧИТАТЬ сама (а не через калькулятор).

    .venv/bin/python tools/make_math.py      # → data/math_pairs.json

Что внутри (всё детерминировано, seed=0):
  add   — сложение a + b, a,b = 0..99           (10 000 пар)
  sub   — вычитание a − b, 0 ≤ b ≤ a ≤ 99        ( 5 050 пар)
  mul   — таблица умножения 0..10 × 0..10        (   121 пара)
  div   — деление нацело из таблицы (56 / 8)     (   110 пар)
  check — «23 + 45 = 68» → «Верно, 68.» / «Нет, будет 68.» (ru, uz, en)
Языки: русский, узбекский, английский — одни и те же примеры в разных формулировках.

ЧЕСТНЫЙ ТЕСТ: 20% пар (a, b) для add/sub целиком спрятаны — модель НИ РАЗУ не видит
этот пример ни в одной формулировке. Если она их решает — она поняла, как складывать
по разрядам, а не зазубрила ответы. Для mul/div прячем один порядок множителей
(видела 8 * 7 — экзамен на 7 * 8): таблицу умножения выводом не получишь, только
перестановкой — это и проверяем.
"""
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FORMS = {
    "ru": ["{a} {op} {b}", "{a} {op} {b} =", "сколько будет {a} {op} {b}", "{a} {op} {b} сколько", "посчитай {a} {op} {b}"],
    "uz": ["{a} {op} {b} necha bo'ladi", "{a} {op} {b} nechchi", "{a} {op} {b} hisobla", "{a} {op} {b} qancha"],
    "en": ["what is {a} {op} {b}", "{a} {op} {b} equals", "how much is {a} {op} {b}", "calculate {a} {op} {b}"],
}


def forms(rng, k):
    pool = [(f, l) for l, fs in FORMS.items() for f in fs]
    return rng.sample(pool, k)


def main():
    rng = random.Random(0)
    train, test = [], []

    def put(bucket, kind, a, op, b, ans, k):
        for f, lang in forms(rng, k):
            bucket.append({"q": f.format(a=a, op=op, b=b), "a": str(ans), "kind": kind, "lang": lang})

    pairs_add = [(a, b) for a in range(100) for b in range(100)]
    pairs_sub = [(a, b) for a in range(100) for b in range(a + 1)]
    for kind, pairs, op, fn in (("add", pairs_add, "+", lambda a, b: a + b), ("sub", pairs_sub, "−", lambda a, b: a - b)):
        rng.shuffle(pairs)
        n_test = len(pairs) // 5
        for a, b in pairs[n_test:]:
            put(train, kind, a, op, b, fn(a, b), 2)
        for a, b in pairs[:n_test]:
            put(test, kind, a, op, b, fn(a, b), 1)

    mul = [(a, b) for a in range(11) for b in range(11)]
    off = [(a, b) for a, b in mul if a < b and a >= 2]
    held_mul = set(rng.sample(off, 12))                      # прячем «a * b», «b * a» остаётся в обучении
    for a, b in mul:
        put(test if (a, b) in held_mul else train, "mul", a, "*", b, a * b, 1 if (a, b) in held_mul else 4)
    div = [(a * b, b, a) for a in range(11) for b in range(1, 11)]
    held_div = set(rng.sample([d for d in div if d[1] != d[2] and d[2] >= 2], 12))
    for n, d, r in div:
        put(test if (n, d, r) in held_div else train, "div", n, "/", d, r, 1 if (n, d, r) in held_div else 4)

    # проверка чужого ответа: «23 + 45 = 68» → «Верно, 68.» / «Нет, будет 68.»
    def checks(bucket, pairs, op, fn, count):
        for a, b in rng.sample(pairs, count):
            right = fn(a, b)
            claim = right if rng.random() < 0.5 else right + rng.choice([-10, -2, -1, 1, 2, 10])
            if claim < 0:
                claim = right + 1
            r = rng.random()
            if r < 0.55:
                q, ans = f"{a} {op} {b} = {claim}", (f"Верно, {right}." if claim == right else f"Нет, будет {right}.")
                lang = "ru"
            elif r < 0.8:
                q, ans = f"is {a} {op} {b} = {claim} right", (f"Correct, {right}." if claim == right else f"No, it is {right}.")
                lang = "en"
            else:
                q, ans = f"{a} {op} {b} = {claim} to'g'rimi", (f"To'g'ri, {right}." if claim == right else f"Yo'q, {right} bo'ladi.")
                lang = "uz"
            bucket.append({"q": q, "a": ans, "kind": "check", "lang": lang})

    n_add_test = len(pairs_add) // 5
    checks(train, pairs_add[n_add_test:], "+", lambda a, b: a + b, 3000)
    checks(test, pairs_add[:n_add_test], "+", lambda a, b: a + b, 300)

    out = ROOT / "data/math_pairs.json"
    out.write_text(json.dumps({"train": train, "test": test}, ensure_ascii=False), encoding="utf-8")
    by = lambda xs: {k: sum(1 for x in xs if x["kind"] == k) for k in ("add", "sub", "mul", "div", "check")}
    print(f"{out}: train {len(train)} {by(train)}, test {len(test)} {by(test)}")


if __name__ == "__main__":
    main()
