"""
Математика на ЛОГИКУ (19.10.2026, Фарход: «чтобы решала сама, зная логику, а не запоминала цифры»).

    .venv/bin/python tools/make_logic_math.py      # → data/logic_math.json

Каждый ответ — рассуждение по шагам + проверка + ответ (как tools/make_reasoning.py, тот же формат,
его проверяет server/reasoncheck.py). Всё считает Python (fractions), ИИ не участвует.
Скобки в вопросах не используем: очистка вопроса (clean_question) их убирает — смысл бы поменялся.

Темы (kind)                                                         класс
  chain     цепочка из 3–5 чисел: 20 + 10 + 1, 50 − 8 + 3 − 10          2–4
  neg       отрицательные: −5 + 3, 4 − 9, −6 * −3, −20 / 4 (с правилом)  6
  negchain  цепочка с отрицательными: −5 + 8 − 12                       6
  order2    порядок действий: 7 + 3 * 4 − 6, 20 − 12 / 4 + 1              3–5
  rect      периметр и площадь прямоугольника/квадрата (формула)         3–5
  speed     путь = скорость × время, и обратно (формула)                 4–5
  cost      стоимость = цена × количество, сдача (2 действия)            3–4
  pct2      процент: скидка/наценка (2 действия)                         6
  lineq2    уравнение a*x + b = c, в т.ч. с отрицательными               6
  word2     задача в 2 действия                                          3–4

ЧЕСТНЫЙ ЭКЗАМЕН: ~17% наборов чисел спрятаны в test по crc32(kind:key), одинаково на всех языках.
"""
import json
import random
import zlib
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LANGS = ("uz", "ru", "en")
TEST_FRAC = 0.17
KW = {"ru": ("Проверка", "Ответ"), "uz": ("Tekshirish", "Javob"), "en": ("Check", "Answer")}
M = "−"   # минус как в normalize_math


def n(v):
    """Число → текст: −5, 12, 7/2."""
    v = Fraction(v)
    s = str(v.numerator) if v.denominator == 1 else f"{abs(v.numerator)}/{v.denominator}"
    if v.denominator != 1 and v < 0:
        s = M + s
    return s.replace("-", M)


def is_test(kind, key):
    return zlib.crc32(f"{kind}:{key}".encode()) % 1000 < TEST_FRAC * 1000


def answer_text(lang, steps, check, ans, rule=None):
    chk, answ = KW[lang]
    head = (rule[lang] + ". ") if rule else ""
    return f"{head}{', '.join(steps)}. {chk}: {', '.join(check)} ✓. {answ}: {ans}"


RULES = {
    "neg_add": {"ru": "минус и плюс: из большего модуля вычитаем меньший, знак большего",
                "uz": "minus va plyus: kattasidan kichigini ayiramiz, ishora kattasiniki",
                "en": "different signs: subtract the smaller size from the bigger, keep its sign"},
    "neg_same": {"ru": "оба минус: складываем модули, ответ с минусом",
                 "uz": "ikkalasi minus: qo'shamiz, javob minus bilan",
                 "en": "both negative: add the sizes, the answer is negative"},
    "mul_signs": {"ru": "минус на минус даёт плюс, минус на плюс даёт минус",
                  "uz": "minus marta minus plyus, minus marta plyus minus",
                  "en": "minus times minus is plus, minus times plus is minus"},
    "sub_neg": {"ru": "вычесть минус — значит прибавить",
                "uz": "minusni ayirish — qo'shish demak",
                "en": "subtracting a negative means adding"},
    "order": {"ru": "сначала умножение и деление, потом сложение и вычитание",
              "uz": "avval ko'paytirish va bo'lish, keyin qo'shish va ayirish",
              "en": "first multiply and divide, then add and subtract"},
    "left": {"ru": "считаем слева направо", "uz": "chapdan o'ngga hisoblaymiz", "en": "we go from left to right"},
    "rect_p": {"ru": "периметр P = 2 * (a + b)", "uz": "perimetr P = 2 * (a + b)", "en": "perimeter P = 2 * (a + b)"},
    "rect_s": {"ru": "площадь S = a * b", "uz": "yuza S = a * b", "en": "area S = a * b"},
    "sq_p": {"ru": "периметр квадрата P = 4 * a", "uz": "kvadrat perimetri P = 4 * a", "en": "square perimeter P = 4 * a"},
    "sq_s": {"ru": "площадь квадрата S = a * a", "uz": "kvadrat yuzasi S = a * a", "en": "square area S = a * a"},
    "s_vt": {"ru": "путь S = v * t", "uz": "yo'l S = v * t", "en": "distance S = v * t"},
    "v_st": {"ru": "скорость v = S / t", "uz": "tezlik v = S / t", "en": "speed v = S / t"},
    "t_sv": {"ru": "время t = S / v", "uz": "vaqt t = S / v", "en": "time t = S / v"},
    "cost": {"ru": "стоимость = цена * количество", "uz": "narx = bitta narxi * soni", "en": "cost = price * amount"},
    "pct": {"ru": "1% это число / 100", "uz": "1% bu son / 100", "en": "1% is the number / 100"},
    "lineq": {"ru": "переносим число вправо, потом делим на коэффициент",
              "uz": "sonni o'ngga o'tkazamiz, keyin koeffitsientga bo'lamiz",
              "en": "move the number to the right, then divide by the coefficient"},
}


def main():
    rng = random.Random(7)
    train, test, seen = [], [], set()

    def put(kind, grade, key, q, steps, check, ans, rule=None):
        if (kind, key) in seen:
            return
        seen.add((kind, key))
        q = q if isinstance(q, dict) else {l: q for l in LANGS}
        for lang in LANGS:
            r = RULES[rule] if rule else None
            rec = {"q": q[lang], "a": answer_text(lang, steps, check, n(ans), r), "kind": kind, "lang": lang,
                   "grade": grade, "steps": steps, "answer": n(ans)}
            (test if is_test(kind, key) else train).append(rec)

    def fold(nums, ops):
        """Слева направо: шаги «a op b = c»."""
        cur, steps = Fraction(nums[0]), []
        for op, x in zip(ops, nums[1:]):
            new = cur + x if op == "+" else cur - x
            steps.append(f"{n(cur)} {op if op == '+' else M} {n(x)} = {n(new)}")
            cur = new
        return cur, steps

    def expr(nums, ops):
        s = n(nums[0])
        for op, x in zip(ops, nums[1:]):
            s += f" {op if op == '+' else M} {n(x)}"
        return s

    # --- chain: 3–5 положительных чисел, + и − (результат может уйти в минус только в negchain) ---
    for _ in range(9000):
        k = rng.choice([3, 3, 4, 4, 5])
        nums = [rng.choice([rng.randint(1, 20), rng.randint(10, 99), rng.randrange(10, 100, 10)]) for _ in range(k)]
        ops = [rng.choice("++-") for _ in range(k - 1)]
        ans, steps = fold(nums, ops)
        if ans < 0 or any("−" in s.split("=")[1] for s in steps):
            continue
        # проверка: обратный ход от ответа
        back = ans
        for op, x in reversed(list(zip(ops, nums[1:]))):
            back = back - x if op == "+" else back + x
        check = [f"{expr(nums, ops)} = {n(ans)}"]
        put("chain", 3, tuple(nums) + tuple(ops), expr(nums, ops), steps, check, ans, "left")

    # --- neg: два числа, хотя бы одно отрицательное ---
    for _ in range(9000):
        a, b = rng.randint(-50, 50), rng.randint(-50, 50)
        if a >= 0 and b >= 0 or a == 0 or b == 0:
            continue
        op = rng.choice("+-*/")
        if op == "/":
            b = b if abs(b) <= 12 else (b % 11 + 2) * (1 if b > 0 else -1)
            a = b * rng.randint(-12, 12)
            if a == 0:
                continue
            ans = Fraction(a, b)
            rule = "mul_signs"
            steps = [f"{abs(a)} / {abs(b)} = {abs(ans)}", f"{'плюс' if ans > 0 else 'минус'}"]
            steps = [f"{abs(a)} / {abs(b)} = {n(abs(ans))}"]
            q = f"{n(a)} / {n(b)}"
            check = [f"{n(ans)} * {n(b)} = {n(a)}"]
        elif op == "*":
            ans = Fraction(a * b)
            rule = "mul_signs"
            steps = [f"{abs(a)} * {abs(b)} = {abs(a * b)}"]
            q = f"{n(a)} * {n(b)}"
            check = [f"{n(ans)} / {n(b)} = {n(a)}"]
        elif op == "+":
            ans = Fraction(a + b)
            rule = "neg_same" if a < 0 and b < 0 else "neg_add"
            steps = ([f"{abs(a)} + {abs(b)} = {abs(a) + abs(b)}"] if rule == "neg_same"
                     else [f"{max(abs(a), abs(b))} {M} {min(abs(a), abs(b))} = {abs(abs(a) - abs(b))}"])
            q = f"{n(a)} + {n(b)}"
            check = [f"{n(ans)} {M} {n(b)} = {n(a)}"]
        else:
            ans = Fraction(a - b)
            if b < 0:
                rule = "sub_neg"
                steps = [f"{n(a)} + {abs(b)} = {n(ans)}"]
            else:
                rule = "neg_same" if a < 0 else "neg_add"
                steps = [f"{n(a)} + {n(-b)} = {n(ans)}"]
            q = f"{n(a)} {M} {n(b)}"
            check = [f"{n(ans)} + {n(b)} = {n(a)}"]
        put("neg", 6, (a, op, b), q, steps, check, ans, rule)

    # --- negchain: цепочка, где проходим через ноль и минус ---
    for _ in range(6000):
        k = rng.choice([3, 3, 4])
        nums = [rng.randint(-30, 30) for _ in range(k)]
        if all(x >= 0 for x in nums) or 0 in nums:
            continue
        ops = [rng.choice("+-") for _ in range(k - 1)]
        cur, steps = Fraction(nums[0]), []
        for op, x in zip(ops, nums[1:]):
            new = cur + x if op == "+" else cur - x
            steps.append(f"{n(cur)} {op if op == '+' else M} {n(x)} = {n(new)}")
            cur = new
        put("negchain", 6, tuple(nums) + tuple(ops), expr(nums, ops), steps,
            [f"{expr(nums, ops)} = {n(cur)}"], cur, "left")

    # --- order2: a ± b * c ± d, a ± b / c ---
    for _ in range(8000):
        a, b, c, d = rng.randint(1, 60), rng.randint(2, 12), rng.randint(2, 12), rng.randint(1, 40)
        form = rng.choice(["a+b*c-d", "a-b*c+d", "b*c+a", "a+b*c+d", "a-b/c+d", "b*c-a*1"])
        if form == "a+b*c-d":
            m = b * c; s1 = a + m; ans = s1 - d
            q, steps = f"{a} + {b} * {c} {M} {d}", [f"{b} * {c} = {m}", f"{a} + {m} = {s1}", f"{s1} {M} {d} = {n(ans)}"]
        elif form == "a-b*c+d":
            m = b * c; s1 = a - m; ans = s1 + d
            q, steps = f"{a} {M} {b} * {c} + {d}", [f"{b} * {c} = {m}", f"{a} {M} {m} = {n(s1)}", f"{n(s1)} + {d} = {n(ans)}"]
        elif form == "b*c+a":
            m = b * c; ans = m + a
            q, steps = f"{b} * {c} + {a}", [f"{b} * {c} = {m}", f"{m} + {a} = {ans}"]
        elif form == "a+b*c+d":
            m = b * c; s1 = a + m; ans = s1 + d
            q, steps = f"{a} + {b} * {c} + {d}", [f"{b} * {c} = {m}", f"{a} + {m} = {s1}", f"{s1} + {d} = {ans}"]
        elif form == "a-b/c+d":
            bb = b * c; m = b; s1 = a - m; ans = s1 + d
            q, steps = f"{a} {M} {bb} / {c} + {d}", [f"{bb} / {c} = {m}", f"{a} {M} {m} = {n(s1)}", f"{n(s1)} + {d} = {n(ans)}"]
        else:
            continue
        put("order2", 4, (form, a, b, c, d), q, steps, [f"{q} = {n(ans)}"], ans, "order")

    # --- rect: формулы прямоугольника и квадрата ---
    T = {"rp": {"ru": "периметр прямоугольника {a} и {b}", "uz": "to'rtburchak perimetri {a} va {b}",
                "en": "perimeter of a rectangle {a} by {b}"},
         "rs": {"ru": "площадь прямоугольника {a} и {b}", "uz": "to'rtburchak yuzasi {a} va {b}",
                "en": "area of a rectangle {a} by {b}"},
         "qp": {"ru": "периметр квадрата со стороной {a}", "uz": "tomoni {a} bo'lgan kvadrat perimetri",
                "en": "perimeter of a square with side {a}"},
         "qs": {"ru": "площадь квадрата со стороной {a}", "uz": "tomoni {a} bo'lgan kvadrat yuzasi",
                "en": "area of a square with side {a}"}}
    for a in range(2, 40):
        for b in range(2, 40):
            if a == b:
                continue
            s = a + b
            put("rect", 4, ("rp", a, b), {l: T["rp"][l].format(a=a, b=b) for l in LANGS},
                [f"{a} + {b} = {s}", f"2 * {s} = {2 * s}"], [f"{2 * s} / 2 = {s}"], 2 * s, "rect_p")
            put("rect", 4, ("rs", a, b), {l: T["rs"][l].format(a=a, b=b) for l in LANGS},
                [f"{a} * {b} = {a * b}"], [f"{a * b} / {b} = {a}"], a * b, "rect_s")
        put("rect", 3, ("qp", a), {l: T["qp"][l].format(a=a) for l in LANGS},
            [f"4 * {a} = {4 * a}"], [f"{4 * a} / 4 = {a}"], 4 * a, "sq_p")
        put("rect", 4, ("qs", a), {l: T["qs"][l].format(a=a) for l in LANGS},
            [f"{a} * {a} = {a * a}"], [f"{a * a} / {a} = {a}"], a * a, "sq_s")

    # --- speed: S = v * t и обратно ---
    TS = {"s": {"ru": "машина едет {v} км/ч {t} часа. какой путь", "uz": "mashina {v} km/soat tezlikda {t} soat yurdi. yo'l qancha",
                "en": "a car goes {v} km/h for {t} hours. what distance"},
          "v": {"ru": "за {t} часа проехали {s} км. какая скорость", "uz": "{t} soatda {s} km yurdi. tezlik qancha",
                "en": "{s} km in {t} hours. what speed"},
          "t": {"ru": "надо проехать {s} км со скоростью {v} км/ч. сколько часов", "uz": "{s} km yo'lni {v} km/soat tezlikda. necha soat",
                "en": "{s} km at {v} km/h. how many hours"}}
    for v in range(5, 121, 5):
        for t in range(2, 10):
            s = v * t
            put("speed", 4, ("s", v, t), {l: TS["s"][l].format(v=v, t=t) for l in LANGS},
                [f"{v} * {t} = {s}"], [f"{s} / {t} = {v}"], s, "s_vt")
            put("speed", 4, ("v", s, t), {l: TS["v"][l].format(s=s, t=t) for l in LANGS},
                [f"{s} / {t} = {v}"], [f"{v} * {t} = {s}"], v, "v_st")
            put("speed", 4, ("t", s, v), {l: TS["t"][l].format(s=s, v=v) for l in LANGS},
                [f"{s} / {v} = {t}"], [f"{v} * {t} = {s}"], t, "t_sv")

    # --- cost: цена * количество, сдача ---
    TC = {"ru": "{k} тетрадей по {p} сум, дали {m}. какая сдача", "uz": "{k} ta daftar {p} so'mdan, {m} berildi. qaytim qancha",
          "en": "{k} notebooks at {p} each, paid {m}. what is the change"}
    for _ in range(5000):
        k, p = rng.randint(2, 9), rng.randrange(500, 5001, 500)
        c = k * p
        m = rng.choice([x for x in (5000, 10000, 20000, 50000, 100000) if x > c] or [c + 1000])
        put("cost", 3, (k, p, m), {l: TC[l].format(k=k, p=p, m=m) for l in LANGS},
            [f"{k} * {p} = {c}", f"{m} {M} {c} = {m - c}"], [f"{m - c} + {c} = {m}"], m - c, "cost")

    # --- pct2: скидка / наценка ---
    TP = {"down": {"ru": "цена {x}, скидка {p}%. сколько стоит", "uz": "narxi {x}, chegirma {p}%. qancha turadi",
                   "en": "price {x}, {p}% off. what is the new price"},
          "up": {"ru": "цена {x}, подорожало на {p}%. сколько стоит", "uz": "narxi {x}, {p}% qimmatladi. qancha turadi",
                 "en": "price {x}, up by {p}%. what is the new price"}}
    for x in range(100, 5001, 100):
        for p in (5, 10, 15, 20, 25, 30, 40, 50):
            one = Fraction(x, 100); part = one * p
            for kind in ("down", "up"):
                ans = x - part if kind == "down" else x + part
                sign = M if kind == "down" else "+"
                put("pct2", 6, (kind, x, p), {l: TP[kind][l].format(x=x, p=p) for l in LANGS},
                    [f"{x} / 100 = {n(one)}", f"{n(one)} * {p} = {n(part)}", f"{x} {sign} {n(part)} = {n(ans)}"],
                    [f"{n(part)} / {n(one)} = {p}"], ans, "pct")

    # --- lineq2: a*x + b = c, коэффициенты и ответы бывают отрицательными ---
    for a in range(-9, 10):
        if a in (0, 1):
            continue
        for xv in range(-12, 13):
            for b in range(-20, 21, 3):
                c = a * xv + b
                bs = f"+ {b}" if b >= 0 else f"{M} {abs(b)}"
                q = f"{n(a)}x {bs} = {n(c)}"
                rhs = c - b
                steps = [f"{n(a)}x = {n(c)} {M if b >= 0 else '+'} {abs(b)}", f"{n(a)}x = {n(rhs)}",
                         f"x = {n(rhs)} / {n(a)}", f"x = {n(xv)}"]
                # steps 1–3 не «чистые» равенства — проверяем только последнее числовое
                steps = [f"{n(c)} {M if b >= 0 else '+'} {abs(b)} = {n(rhs)}", f"{n(rhs)} / {n(a)} = {n(xv)}"]
                check = [f"{n(a)} * {n(xv)} {bs} = {n(c)}"]
                put("lineq2", 6, (a, xv, b), q, steps, check, xv, "lineq")

    # --- word2: две операции подряд ---
    TW = {"ru": "у {nm} было {a} конфет, купил ещё {b} коробки по {c}. сколько стало",
          "uz": "{nm}da {a} ta konfet bor edi, yana {b} quti {c} tadan oldi. nechta bo'ldi",
          "en": "{nm} had {a} candies and bought {b} boxes of {c}. how many now"}
    TW2 = {"ru": "{a} яблок разложили по {b} в пакет и {c} съели. сколько пакетов осталось",
           "uz": "{a} ta olmani {b} tadan paketga soldik, {c} paketini yedik. nechta paket qoldi",
           "en": "{a} apples were put {b} in a bag and {c} bags were eaten. how many bags are left"}
    names = {"ru": "Тимура", "uz": "Sardor", "en": "Sam"}
    for a in range(1, 40):
        for b in range(2, 10):
            for c in range(2, 13):
                m = b * c
                put("word2", 3, ("buy", a, b, c), {l: TW[l].format(nm=names[l], a=a, b=b, c=c) for l in LANGS},
                    [f"{b} * {c} = {m}", f"{a} + {m} = {a + m}"], [f"{a + m} {M} {m} = {a}"], a + m, "order")
    for bags in range(3, 20):
        for b in range(2, 10):
            for c in range(1, bags):
                a = bags * b
                put("word2", 4, ("bags", a, b, c), {l: TW2[l].format(a=a, b=b, c=c) for l in LANGS},
                    [f"{a} / {b} = {bags}", f"{bags} {M} {c} = {bags - c}"], [f"{bags - c} + {c} = {bags}"],
                    bags - c, "order")

    tr_q = {(x["q"], x["lang"]) for x in train}
    test = [x for x in test if (x["q"], x["lang"]) not in tr_q]   # одинаковый вопрос из разных ключей — только в train
    out = ROOT / "data/logic_math.json"
    out.write_text(json.dumps({"train": train, "test": test}, ensure_ascii=False), encoding="utf-8")
    kinds = sorted({x["kind"] for x in train + test})
    print(f"{out}: train {len(train)}, test {len(test)}")
    for k in kinds:
        tr = sum(x["kind"] == k for x in train)
        te = sum(x["kind"] == k for x in test)
        print(f"  {k:9s} train {tr:6d}  test {te:5d}")


if __name__ == "__main__":
    main()
