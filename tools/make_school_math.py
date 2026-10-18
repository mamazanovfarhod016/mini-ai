"""
Школьная математика 1–6 класс на трёх языках (uz, ru, en) — чтобы Mini AI училась РЕШАТЬ, а не зубрить.

    .venv/bin/python tools/make_school_math.py      # → data/school_math.json

Все ответы считает Python (fractions.Fraction, math.gcd) — правильные всегда.
Формат как у data/math_pairs.json: {"train": [...], "test": [...]}, у каждого {q, a, kind, lang, grade}.

ЧЕСТНЫЙ ЭКЗАМЕН: для каждой темы 20% наборов чисел целиком спрятаны в test —
модель ни разу не видит эти числа ни на одном языке. Решит → поняла правило.

Темы (kind)                                класс
  cmp      сравнение чисел (>, <, =)          1–2
  add3     сложение до 1000                   2–3
  sub3     вычитание до 1000                  2–3
  mul21    двузначное × однозначное           3
  divrem   деление с остатком                 3
  order    порядок действий (a + b * c)       3–4
  word     задачи со словами (+ − × ÷)        1–4
  round    округление до десятков/сотен       4
  units    перевод единиц (м↔см, кг↔г, ч↔мин) 2–5
  frac_s   сократить дробь                    5
  frac_a   сложить/вычесть дроби (общий знам.) 5
  frac_c   сравнить дроби                     5
  dec      десятичные дроби (+, −)            5
  pct      процент от числа                   5–6
  gcd      НОД                                6
  lcm      НОК                                6
  divis    делится ли на 2, 3, 5, 9, 10       6
  neg      отрицательные числа (+, −)         6
  prop     пропорция (a : b = c : x)          6
"""
import json
import math
import random
import zlib
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LANGS = ("uz", "ru", "en")
TEST_FRAC = 0.2


def is_test(kind, key):
    """Детерминированно: этот набор чисел спрятан для экзамена? (одинаково на всех языках)"""
    return zlib.crc32(f"{kind}:{key}".encode()) % 100 < TEST_FRAC * 100


def frac(f):
    f = Fraction(f)
    return str(f.numerator) if f.denominator == 1 else f"{f.numerator}/{f.denominator}"


def dec(x):
    s = f"{x:.2f}".rstrip("0").rstrip(".")
    return s if s != "-0" else "0"


# --------------------------------------------------------------------------------------
# Формулировки: {lang: [шаблоны]}. Ответ — только результат (его проверяет скрипт).
# --------------------------------------------------------------------------------------
T = {
    "cmp": {"uz": ["{a} va {b} qaysi katta", "{a} ? {b} solishtir"],
            "ru": ["сравни {a} и {b}", "что больше {a} или {b}"],
            "en": ["compare {a} and {b}", "which is bigger {a} or {b}"]},
    "calc": {"uz": ["{e} necha bo'ladi", "{e} hisobla"],
             "ru": ["сколько будет {e}", "посчитай {e}"],
             "en": ["what is {e}", "calculate {e}"]},
    "divrem": {"uz": ["{a} ni {b} ga qoldiqli bo'l"],
               "ru": ["раздели {a} на {b} с остатком"],
               "en": ["divide {a} by {b} with remainder"]},
    "round10": {"uz": ["{a} ni o'nlikkacha yaxlitla"], "ru": ["округли {a} до десятков"], "en": ["round {a} to the nearest ten"]},
    "round100": {"uz": ["{a} ni yuzlikkacha yaxlitla"], "ru": ["округли {a} до сотен"], "en": ["round {a} to the nearest hundred"]},
    "frac_s": {"uz": ["{a}/{b} kasrni qisqartir"], "ru": ["сократи дробь {a}/{b}"], "en": ["simplify the fraction {a}/{b}"]},
    "frac_c": {"uz": ["{a} va {b} kasrlarni solishtir"], "ru": ["сравни дроби {a} и {b}"], "en": ["compare the fractions {a} and {b}"]},
    "pct": {"uz": ["{b} ning {p}% i qancha"], "ru": ["сколько будет {p}% от {b}", "найди {p}% от {b}"],
            "en": ["what is {p}% of {b}", "find {p}% of {b}"]},
    "gcd": {"uz": ["{a} va {b} ning EKUB i"], "ru": ["найди нод {a} и {b}"], "en": ["find the gcd of {a} and {b}"]},
    "lcm": {"uz": ["{a} va {b} ning EKUK i"], "ru": ["найди нок {a} и {b}"], "en": ["find the lcm of {a} and {b}"]},
    "divis": {"uz": ["{a} soni {b} ga bo'linadimi"], "ru": ["делится ли {a} на {b}"], "en": ["is {a} divisible by {b}"]},
    "prop": {"uz": ["{a} : {b} = {c} : x x ni top"], "ru": ["{a} : {b} = {c} : x найди x"], "en": ["{a} : {b} = {c} : x find x"]},
}
YES = {"uz": "Ha", "ru": "Да", "en": "Yes"}
NO = {"uz": "Yo'q", "ru": "Нет", "en": "No"}
REM = {"uz": "{q}, qoldiq {r}", "ru": "{q}, остаток {r}", "en": "{q}, remainder {r}"}

UNITS = [  # (из, в, множитель, {lang: (из, в)})
    ("m", "cm", 100, {"uz": ("metr", "santimetr"), "ru": ("метров", "сантиметров"), "en": ("meters", "centimeters")}),
    ("km", "m", 1000, {"uz": ("kilometr", "metr"), "ru": ("километров", "метров"), "en": ("kilometers", "meters")}),
    ("kg", "g", 1000, {"uz": ("kilogramm", "gramm"), "ru": ("килограммов", "граммов"), "en": ("kilograms", "grams")}),
    ("h", "min", 60, {"uz": ("soat", "daqiqa"), "ru": ("часов", "минут"), "en": ("hours", "minutes")}),
    ("min", "s", 60, {"uz": ("daqiqa", "soniya"), "ru": ("минут", "секунд"), "en": ("minutes", "seconds")}),
]
UNIT_Q = {"uz": "{n} {u1} necha {u2}", "ru": "{n} {u1} это сколько {u2}", "en": "how many {u2} in {n} {u1}"}

NAMES = {"uz": ["Ali", "Malika", "Sardor", "Dilnoza"], "ru": ["Аня", "Тимур", "Саша", "Лола"], "en": ["Tom", "Anna", "Sam", "Lily"]}
ITEMS = {"uz": ["olma", "qalam", "kitob", "konfet"], "ru": ["яблок", "карандашей", "книг", "конфет"],
         "en": ["apples", "pencils", "books", "candies"]}
WORD = {
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


def main():
    rng = random.Random(0)
    train, test = [], []
    seen, seen_q = set(), set()

    def put(kind, grade, key, items, hide_key=None):
        """items: [(lang, вопрос, ответ)] — один набор чисел на всех языках сразу.
        hide_key — по чему прятать в экзамен (по умолчанию key); для задач — только числа, без имён."""
        if (kind, key) in seen or any(q in seen_q for _, q, _ in items):
            return
        seen.add((kind, key))
        seen_q.update(q for _, q, _ in items)
        hide = is_test(kind, key if hide_key is None else hide_key)
        for lang, q, a in items:
            rec = {"q": q, "a": str(a), "kind": kind, "lang": lang, "grade": grade}
            (test if hide else train).append(rec)

    def tpl(name, lang, **kw):
        return rng.choice(T[name][lang]).format(**kw)

    def calc(kind, grade, key, expr, ans):
        put(kind, grade, key, [(l, tpl("calc", l, e=expr), ans) for l in LANGS])

    # --- 1–2 класс: сравнение ---
    for _ in range(1500):
        a, b = rng.randint(0, 999), rng.randint(0, 999)
        if rng.random() < 0.1:
            b = a
        sign = ">" if a > b else "<" if a < b else "="
        put("cmp", 1, (a, b), [(l, tpl("cmp", l, a=a, b=b), f"{a} {sign} {b}") for l in LANGS])

    # --- 2–3 класс: + − до 1000 ---
    for _ in range(3000):
        a, b = rng.randint(100, 899), rng.randint(10, 999)
        if a + b <= 1000:
            calc("add3", 2, (a, b), f"{a} + {b}", a + b)
        x, y = max(a, b), min(a, b)
        calc("sub3", 2, (x, y), f"{x} − {y}", x - y)

    # --- 3 класс: × и ÷ с остатком ---
    for a in range(11, 100):
        for b in range(2, 10):
            calc("mul21", 3, (a, b), f"{a} * {b}", a * b)
    for _ in range(1500):
        b = rng.randint(2, 9)
        a = rng.randint(b + 1, 99)
        q, r = divmod(a, b)
        put("divrem", 3, (a, b), [(l, tpl("divrem", l, a=a, b=b), REM[l].format(q=q, r=r)) for l in LANGS])

    # --- 3–4 класс: порядок действий ---
    for _ in range(1500):
        a, b, c = rng.randint(1, 50), rng.randint(2, 9), rng.randint(2, 9)
        if rng.random() < 0.5:
            calc("order", 3, ("a+b*c", a, b, c), f"{a} + {b} * {c}", a + b * c)
        else:
            if a < b * c:
                a += b * c
            calc("order", 3, ("a-b*c", a, b, c), f"{a} − {b} * {c}", a - b * c)

    # --- 1–4 класс: задачи со словами ---
    for _ in range(2500):
        op = rng.choice(["add", "sub", "mul", "div"])
        ni, ii = rng.randrange(4), rng.randrange(4)
        if op == "add":
            a, b = rng.randint(2, 60), rng.randint(2, 40); ans = a + b
        elif op == "sub":
            a = rng.randint(5, 90); b = rng.randint(1, a - 1); ans = a - b
        elif op == "mul":
            a, b = rng.randint(2, 9), rng.randint(2, 12); ans = a * b
        else:
            b = rng.randint(2, 9); ans = rng.randint(2, 12); a = b * ans
        put("word", 2, (op, a, b, ni, ii),
            [(l, WORD[op][l].format(n=NAMES[l][ni], i=ITEMS[l][ii], a=a, b=b), ans) for l in LANGS],
            hide_key=(op, a, b))

    # --- 4 класс: округление ---
    for _ in range(1500):
        a = rng.randint(11, 9999)
        r10 = int(math.floor(a / 10 + 0.5) * 10)
        put("round", 4, ("10", a), [(l, tpl("round10", l, a=a), r10) for l in LANGS])
        if a >= 100:
            r100 = int(math.floor(a / 100 + 0.5) * 100)
            put("round", 4, ("100", a), [(l, tpl("round100", l, a=a), r100) for l in LANGS])

    # --- 2–5 класс: единицы ---
    for u1, u2, k, names in UNITS:
        for n in range(1, 60):
            put("units", 3, (u1, n), [(l, UNIT_Q[l].format(n=n, u1=names[l][0], u2=names[l][1]), n * k) for l in LANGS])

    # --- 5 класс: дроби ---
    for b in range(2, 25):
        for a in range(1, b * 3):
            g = rng.randint(2, 6)
            if math.gcd(a, b) == 1 and a * g < 100:
                put("frac_s", 5, (a * g, b * g), [(l, tpl("frac_s", l, a=a * g, b=b * g), frac(Fraction(a, b))) for l in LANGS])
    for _ in range(2500):
        d = rng.randint(2, 12)
        a, c = rng.randint(1, d * 2), rng.randint(1, d * 2)
        if rng.random() < 0.5:
            calc("frac_a", 5, ("+", a, c, d), f"{a}/{d} + {c}/{d}", frac(Fraction(a + c, d)))
        else:
            a, c = max(a, c), min(a, c)
            calc("frac_a", 5, ("-", a, c, d), f"{a}/{d} − {c}/{d}", frac(Fraction(a - c, d)))
    for _ in range(2000):
        x = Fraction(rng.randint(1, 11), rng.randint(2, 12))
        y = Fraction(rng.randint(1, 11), rng.randint(2, 12))
        sx, sy = f"{x.numerator}/{x.denominator}", f"{y.numerator}/{y.denominator}"
        sign = ">" if x > y else "<" if x < y else "="
        put("frac_c", 5, (sx, sy), [(l, tpl("frac_c", l, a=sx, b=sy), f"{sx} {sign} {sy}") for l in LANGS])

    # --- 5 класс: десятичные ---
    for _ in range(2500):
        a, b = rng.randint(1, 999) / 100, rng.randint(1, 999) / 100
        if rng.random() < 0.5:
            calc("dec", 5, ("+", a, b), f"{dec(a)} + {dec(b)}", dec(a + b))
        else:
            a, b = max(a, b), min(a, b)
            calc("dec", 5, ("-", a, b), f"{dec(a)} − {dec(b)}", dec(a - b))

    # --- 5–6 класс: проценты ---
    for p in (1, 5, 10, 20, 25, 30, 40, 50, 60, 75, 80, 90, 100, 150, 200):
        for b in range(10, 1001, 10):
            v = Fraction(p * b, 100)
            if v.denominator == 1:
                put("pct", 5, (p, b), [(l, tpl("pct", l, p=p, b=b), v.numerator) for l in LANGS])

    # --- 6 класс: НОД, НОК, делимость ---
    for _ in range(2500):
        g = rng.randint(1, 15)
        a, b = g * rng.randint(1, 12), g * rng.randint(1, 12)
        if a == b:
            continue
        put("gcd", 6, (a, b), [(l, tpl("gcd", l, a=a, b=b), math.gcd(a, b)) for l in LANGS])
        if a * b // math.gcd(a, b) <= 1000:
            put("lcm", 6, (a, b), [(l, tpl("lcm", l, a=a, b=b), a * b // math.gcd(a, b)) for l in LANGS])
    for _ in range(2500):
        d = rng.choice([2, 3, 5, 9, 10])
        a = rng.randint(10, 9999)
        if rng.random() < 0.5:
            a -= a % d
        put("divis", 6, (a, d), [(l, tpl("divis", l, a=a, b=d), (YES if a % d == 0 else NO)[l]) for l in LANGS])

    # --- 6 класс: отрицательные числа ---
    for _ in range(3000):
        a, b = rng.randint(-50, 50), rng.randint(-50, 50)
        sa, sb = (f"−{-a}" if a < 0 else str(a)), (f"−{-b}" if b < 0 else str(b))
        if a >= 0 and b >= 0:
            continue
        op = rng.choice(["+", "−"])
        r = a + b if op == "+" else a - b
        calc("neg", 6, (a, op, b), f"{sa} {op} {sb}", f"−{-r}" if r < 0 else r)

    # --- 6 класс: пропорция a : b = c : x → x = b·c / a ---
    for _ in range(2500):
        a, k = rng.randint(1, 12), rng.randint(2, 9)
        b, c = rng.randint(1, 12) * a, a * k
        x = Fraction(b * c, a)
        if x.denominator == 1:
            put("prop", 6, (a, b, c), [(l, tpl("prop", l, a=a, b=b, c=c), x.numerator) for l in LANGS])

    out = ROOT / "data/school_math.json"
    out.write_text(json.dumps({"train": train, "test": test}, ensure_ascii=False), encoding="utf-8")
    kinds = sorted({x["kind"] for x in train + test})
    print(f"{out}: train {len(train)}, test {len(test)}")
    for k in kinds:
        print(f"  {k:7s} train {sum(x['kind'] == k for x in train):5d}  test {sum(x['kind'] == k for x in test):4d}")


if __name__ == "__main__":
    main()
