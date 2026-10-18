"""
Пошаговые РАССУЖДЕНИЯ по школьной математике на трёх языках (uz, ru, en) —
чтобы Mini AI училась показывать шаги и САМА их проверять, а не угадывать ответ.

    .venv/bin/python tools/make_reasoning.py      # → data/reasoning.json

Все шаги и ответы считает Python (fractions.Fraction, math) — ответ `a`,
который должна выучить модель, это ПОЛНЫЙ текст рассуждения:
  шаги вычисления, строка «Проверка/Tekshirish/Check: ...», строка «Ответ/Javob/Answer: ...».
server/reasoncheck.py разбирает этот текст и проверяет арифметику каждого
уравнения «X op Y = Z» — без ИИ, чистый скрипт.

ЧЕСТНЫЙ ЭКЗАМЕН: как и в make_school_math.py — для каждой темы ~17% наборов
чисел целиком спрятаны в test по crc32(kind:key), одинаково для всех 3 языков.
Если модель решает их — она поняла способ, а не выучила число наизусть.

Темы (kind)                                             класс
  add_pv    сложение по разрядам (десятки, единицы)      2-3
  sub_pv    вычитание по разрядам (десятки, единицы)      2-3
  mul21     двузначное × однозначное, распределение      3
  divrem    деление с остатком + проверка q*b+r=a         3
  order     порядок действий a+b*c / a-b*c                3-4
  word      задачи со словами (1 действие)                2-4
  frac_a    дроби: общий знаменатель (НОК), сложение      5
  pct       процент от числа через 1%                     5-6
  lineq     линейное уравнение a*x + b = c                6
  prop      пропорция a : b = c : x                       6
"""
import json
import math
import random
import zlib
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LANGS = ("uz", "ru", "en")
TEST_FRAC = 0.17

KW = {
    "ru": {"check": "Проверка", "answer": "Ответ"},
    "uz": {"check": "Tekshirish", "answer": "Javob"},
    "en": {"check": "Check", "answer": "Answer"},
}
REM_WORD = {"ru": "остаток", "uz": "qoldiq", "en": "remainder"}

NAMES = {"uz": ["Ali", "Malika", "Sardor", "Dilnoza"], "ru": ["Аня", "Тимур", "Саша", "Лола"],
         "en": ["Tom", "Anna", "Sam", "Lily"]}
ITEMS = {"uz": ["olma", "qalam", "kitob", "konfet"], "ru": ["яблок", "карандашей", "книг", "конфет"],
         "en": ["apples", "pencils", "books", "candies"]}
WORD_Q = {
    "add": {"uz": "{n}da {a} ta {i} bor edi, yana {b} ta berishdi. Endi nechta",
            "ru": "у {n} было {a} {i}, дали ещё {b}. сколько стало",
            "en": "{n} had {a} {i} and got {b} more. how many now"},
    "sub": {"uz": "{n}da {a} ta {i} bor edi, {b} tasini berdi. Nechta qoldi",
            "ru": "у {n} было {a} {i}, {b} отдал. сколько осталось",
            "en": "{n} had {a} {i} and gave away {b}. how many are left"},
    "mul": {"uz": "{a} ta qutida {b} tadan {i} bor. Hammasi nechta",
            "ru": "в {a} коробках по {b} {i}. сколько всего",
            "en": "there are {a} boxes with {b} {i} each. how many in total"},
    "div": {"uz": "{a} ta {i}ni {b} kishiga teng bo'lishdi. Har biriga nechta",
            "ru": "{a} {i} разделили поровну на {b} человек. сколько каждому",
            "en": "{a} {i} were shared equally among {b} people. how many each"},
}


def is_test(kind, key):
    """Детерминированно: этот набор чисел спрятан для экзамена? (одинаково на всех языках)."""
    return zlib.crc32(f"{kind}:{key}".encode()) % 1000 < TEST_FRAC * 1000


def frac(f):
    f = Fraction(f)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def steps_text(steps):
    return ", ".join(steps)


def build_a(lang, steps, check_eqs, answer_str):
    kw = KW[lang]
    return (f"{steps_text(steps)}. {kw['check']}: {steps_text(check_eqs)} ✓. "
            f"{kw['answer']}: {answer_str}")


def main():
    rng = random.Random(0)
    train, test = [], []
    seen = set()

    def put(kind, grade, key, q_by_lang, steps, check_eqs, answer_str, answer_val=None):
        """q_by_lang: {lang: question text}. Числовая часть steps/check_eqs/answer одна для всех языков."""
        if (kind, key) in seen:
            return
        seen.add((kind, key))
        hide = is_test(kind, key)
        a_val = answer_val if answer_val is not None else answer_str
        for lang in LANGS:
            rec = {
                "q": q_by_lang[lang],
                "a": build_a(lang, steps, check_eqs, answer_str),
                "kind": kind,
                "lang": lang,
                "grade": grade,
                "steps": list(steps),
                "answer": a_val,
            }
            (test if hide else train).append(rec)

    # --- сложение/вычитание по разрядам (десятки, единицы), 2-3 класс ---
    for _ in range(6000):
        a, b = rng.randint(10, 899), rng.randint(10, 899)
        bt, bo = b - b % 10, b % 10
        s1 = a + bt
        ans = s1 + bo
        steps = [f"{a} + {bt} = {s1}", f"{s1} + {bo} = {ans}"]
        check = [f"{ans} − {b} = {a}"]
        q = {l: f"{a} + {b}" for l in LANGS}
        put("add_pv", 2, (a, b), q, steps, check, str(ans))

        x, y = max(a, b), min(a, b)
        yt, yo = y - y % 10, y % 10
        s1 = x - yt
        ans = s1 - yo
        steps = [f"{x} − {yt} = {s1}", f"{s1} − {yo} = {ans}"]
        check = [f"{ans} + {y} = {x}"]
        q = {l: f"{x} − {y}" for l in LANGS}
        put("sub_pv", 2, (x, y), q, steps, check, str(ans))

    # --- двузначное × однозначное, распределение, 3 класс ---
    for a in range(11, 100):
        for b in range(2, 10):
            at, ao = a - a % 10, a % 10
            m1, m2 = at * b, ao * b
            ans = m1 + m2
            steps = [f"{at} * {b} = {m1}", f"{ao} * {b} = {m2}", f"{m1} + {m2} = {ans}"]
            check = [f"{ans} / {b} = {a}"]
            q = {l: f"{a} * {b}" for l in LANGS}
            put("mul21", 3, (a, b), q, steps, check, str(ans))

    # --- деление с остатком, 3 класс ---
    for _ in range(4000):
        b = rng.randint(2, 9)
        a = rng.randint(b + 1, 999)
        q, r = divmod(a, b)
        qb = q * b
        steps = [f"{q} * {b} = {qb}", f"{a} − {qb} = {r}"]
        check = [f"{q} * {b} = {qb}", f"{qb} + {r} = {a}"]
        qd = {l: f"{a} / {b}" for l in LANGS}
        # ответ разный по словам на каждом языке -> соберём вручную, не через put()
        if (("divrem"), (a, b)) in seen:
            continue
        seen.add(("divrem", (a, b)))
        hide = is_test("divrem", (a, b))
        for lang in LANGS:
            ans_str = f"{q}, {REM_WORD[lang]} {r}"
            rec = {"q": qd[lang], "a": build_a(lang, steps, check, ans_str), "kind": "divrem",
                   "lang": lang, "grade": 3, "steps": list(steps), "answer": ans_str}
            (test if hide else train).append(rec)

    # --- порядок действий, 3-4 класс ---
    for _ in range(4000):
        a, b, c = rng.randint(1, 80), rng.randint(2, 12), rng.randint(2, 12)
        bc = b * c
        if rng.random() < 0.5:
            ans = a + bc
            steps = [f"{b} * {c} = {bc}", f"{a} + {bc} = {ans}"]
            check = [f"{ans} − {bc} = {a}"]
            q = {l: f"{a} + {b} * {c}" for l in LANGS}
            put("order", 3, ("+", a, b, c), q, steps, check, str(ans))
        else:
            if a < bc:
                a += bc
            ans = a - bc
            steps = [f"{b} * {c} = {bc}", f"{a} − {bc} = {ans}"]
            check = [f"{ans} + {bc} = {a}"]
            q = {l: f"{a} − {b} * {c}" for l in LANGS}
            put("order", 3, ("-", a, b, c), q, steps, check, str(ans))

    # --- задачи со словами, 2-4 класс (1 действие) ---
    for _ in range(5000):
        op = rng.choice(["add", "sub", "mul", "div"])
        ni, ii = rng.randrange(4), rng.randrange(4)
        if op == "add":
            a, b = rng.randint(2, 60), rng.randint(2, 40)
            ans = a + b
            steps = [f"{a} + {b} = {ans}"]
            check = [f"{ans} − {b} = {a}"]
        elif op == "sub":
            a = rng.randint(5, 90)
            b = rng.randint(1, a - 1)
            ans = a - b
            steps = [f"{a} − {b} = {ans}"]
            check = [f"{ans} + {b} = {a}"]
        elif op == "mul":
            a, b = rng.randint(2, 9), rng.randint(2, 12)
            ans = a * b
            steps = [f"{a} * {b} = {ans}"]
            check = [f"{ans} / {b} = {a}"]
        else:
            b = rng.randint(2, 9)
            ans = rng.randint(2, 12)
            a = b * ans
            steps = [f"{a} / {b} = {ans}"]
            check = [f"{ans} * {b} = {a}"]
        q = {l: WORD_Q[op][l].format(n=NAMES[l][ni], i=ITEMS[l][ii], a=a, b=b) for l in LANGS}
        put("word", 2, (op, a, b), q, steps, check, str(ans))

    # --- дроби: общий знаменатель (НОК), сложение/вычитание, 5 класс ---
    for _ in range(4000):
        d, e = rng.randint(2, 15), rng.randint(2, 15)
        L = d * e // math.gcd(d, e)
        if L > 300:
            continue
        a = rng.randint(1, d * 2)
        c = rng.randint(1, e * 2)
        sub = rng.random() < 0.5
        A, C = a * (L // d), c * (L // e)
        if sub:
            if A < C:          # первая дробь должна быть не меньше второй (a/d − c/e)
                a, d, A, c, e, C = c, e, C, a, d, A
            S = A - C
            op_sym = "−"
        else:
            S = A + C
            op_sym = "+"
        g = math.gcd(S, L) if S else L
        p, qd2 = (S // g, L // g) if S else (0, 1)
        res_str = str(p) if qd2 == 1 else f"{p}/{qd2}"
        steps = [f"{a}/{d} = {A}/{L}", f"{c}/{e} = {C}/{L}", f"{A} {op_sym} {C} = {S}"]
        check = [f"{p} * {g} = {S}"]
        q = {l: f"{a}/{d} {op_sym} {c}/{e}" for l in LANGS}
        put("frac_a", 5, (d, e, a, c, sub), q, steps, check, res_str)

    # --- процент от числа через 1%, 5-6 класс ---
    for p in (1, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 80, 90):
        for b in range(10, 1001, 10):
            if (p * b) % 100 != 0:
                continue
            onepct = Fraction(b, 100)
            val = Fraction(p * b, 100)
            if val.denominator != 1:
                continue
            val = val.numerator
            onepct_s = frac(onepct)
            steps = [f"{b}/100 = {onepct_s}", f"{onepct_s} * {p} = {val}"]
            check = [f"{val} * 100 = {p} * {b}"]
            q = {l: f"{p}% {b}" for l in LANGS}
            put("pct", 5, (p, b), q, steps, check, str(val))

    # --- линейное уравнение a*x + b = c, 6 класс ---
    for _ in range(4000):
        a = rng.randint(2, 12)
        x = rng.randint(1, 20)
        b = rng.randint(0, 50)
        d = a * x
        c = d + b
        steps = [f"{c} − {b} = {d}", f"{d} / {a} = {x}"]
        check = [f"{a} * {x} = {d}", f"{d} + {b} = {c}"]
        q = {l: f"{a} * x + {b} = {c}" for l in LANGS}
        put("lineq", 6, (a, b, c), q, steps, check, str(x))

    # --- пропорция a : b = c : x, 6 класс ---
    for _ in range(4000):
        a, k = rng.randint(1, 12), rng.randint(2, 9)
        b, c = rng.randint(1, 12) * a, a * k
        m = b * c
        if m % a != 0:
            continue
        x = m // a
        steps = [f"{b} * {c} = {m}", f"{m} / {a} = {x}"]
        check = [f"{a} * {x} = {m}"]
        q = {l: f"{a}/{b} = {c}/x" for l in LANGS}
        put("prop", 6, (a, b, c), q, steps, check, str(x))

    out = ROOT / "data/reasoning.json"
    out.write_text(json.dumps({"train": train, "test": test}, ensure_ascii=False), encoding="utf-8")
    kinds = sorted({x["kind"] for x in train + test})
    print(f"{out}: train {len(train)}, test {len(test)}")
    for k in kinds:
        for l in LANGS:
            tr = sum(x["kind"] == k and x["lang"] == l for x in train)
            te = sum(x["kind"] == k and x["lang"] == l for x in test)
            print(f"  {k:7s} {l}  train {tr:6d}  test {te:5d}")


if __name__ == "__main__":
    main()
