"""
Проверка примеров и «тренажёр по слабым местам» — только скрипты, без ИИ-наставника.

Всё считает Python, поэтому ответ всегда правильный:
  • parse_math(q)        — понять пример из вопроса («23+45», «сколько будет 7 умножить на 8», «2+2=5»);
  • kind_of(a, op, b)    — тип примера: add_2x2 (двузначное + двузначное), mul_1x3, sub_3x2 …;
  • log_answer(...)      — сервер на КАЖДЫЙ пример из приложения проверяет ответ модели,
                           ошибки копит по типам (server_data/math_stats.json);
  • drill(kinds)         — на слабые типы генерирует 500 новых примеров для обучения и 100
                           спрятанных для экзамена (детерминированно, seed по типу).
"""
import random
import re
import zlib

from server import store

WORD_OPS = [(r"(умножить|умножь|умноженное)\s+на|ko'paytir(ilsa|sa)?|×|x(?=\s*\d)|х(?=\s*\d)", "*"),
            (r"(разделить|раздели|делить|делённое|деленное)\s+на|bo'lin(sa|gan)?|÷|:", "/"),
            (r"плюс|qo'sh(ilsa|sa)?", "+"), (r"минус|ayir(ilsa|sa)?|−|–", "-")]
OPS = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
SYM = {"add": "+", "sub": "−", "mul": "*", "div": "/"}
DRILL_TRAIN, DRILL_TEST = 500, 100
WEAK_MIN_ERRORS = 2          # тип считается слабым после 2+ ошибок…
WEAK_MAX_ACC = 0.8           # …и если верно меньше 80%


def parse_math(text):
    """→ (a, op, b, claim|None) для простого примера «a op b» или None. Как mathAnswer в learn.js."""
    t = str(text).lower().replace("ё", "е")
    for a in "’ʻʼ`":
        t = t.replace(a, "'")
    t = re.sub(r",(?=\d)", ".", t)
    t = re.sub(r"^(а\s+)?(сколько|скок)\s+(будет|равно|получится)\s*|^(посчитай|вычисли|реши)\s*|^(nechchi|necha)\s+(bo'ladi)?\s*", "", t)
    t = re.sub(r"\s*(сколько|nechchi|necha|bo'ladi|будет|to'g'rimi)\s*\??$", "", t)
    for pat, rep in WORD_OPS:
        t = re.sub(pat, rep, t)
    t = re.sub(r"[?!\s]+$", "", t)
    m = re.fullmatch(r"\s*(\d+)\s*([+\-*/])\s*(\d+)\s*(?:=\s*(-?\d+)?)?\s*", t)
    if not m:
        return None
    a, op, b, claim = int(m.group(1)), m.group(2), int(m.group(3)), m.group(4)
    return a, op, b, (int(claim) if claim is not None else None)


def solve(a, op, b):
    if op == "+": return a + b
    if op == "-": return a - b
    if op == "*": return a * b
    if b == 0: return None
    return a / b


def fmt(v):
    v = round(v, 6)
    return str(int(v)) if v == int(v) else str(v).replace(".", ",")


def right_answer(text, lang="ru"):
    """Строка, которую модель ДОЛЖНА была ответить (тот же формат, что в обучении), или None."""
    p = parse_math(text)
    if not p:
        return None
    a, op, b, claim = p
    v = solve(a, op, b)
    if v is None:
        return "Nolga bo'lib bo'lmaydi." if lang == "uz" else "На ноль делить нельзя."
    s = fmt(v)
    if claim is None:
        return s
    if lang == "uz":
        return f"To'g'ri, {s}." if claim == v else f"Yo'q, {s} bo'ladi."
    return f"Верно, {s}." if claim == v else f"Нет, будет {s}."


def kind_of(a, op, b):
    return f"{OPS[op]}_{len(str(a))}x{len(str(b))}"


def log_answer(data, q, model_answer, lang="ru"):
    """Проверить ответ модели на пример и учесть в статистике по типам. → dict или None (не пример)."""
    p = parse_math(q)
    if not p:
        return None
    a, op, b, _ = p
    right = right_answer(q, lang)
    ok = str(model_answer).strip() == right
    kind = kind_of(a, op, b)
    stats = store.read_json(data / "math_stats.json", {})
    s = stats.setdefault(kind, {"ok": 0, "wrong": 0, "examples": []})
    s["ok" if ok else "wrong"] += 1
    if not ok:
        s["examples"] = (s["examples"] + [{"q": q, "got": model_answer, "right": right}])[-5:]
    store.write_json(data / "math_stats.json", stats)
    return {"kind": kind, "ok": ok, "right": right}


def weak_kinds(data):
    stats = store.read_json(data / "math_stats.json", {})
    out = []
    for k, s in stats.items():
        n = s["ok"] + s["wrong"]
        if s["wrong"] >= WEAK_MIN_ERRORS and s["ok"] / n < WEAK_MAX_ACC:
            out.append(k)
    return sorted(out)


def _rand_num(rng, digits):
    return rng.randint(0 if digits == 1 else 10 ** (digits - 1), 10 ** digits - 1)


def drill(kind, n_train=DRILL_TRAIN, n_test=DRILL_TEST):
    """→ (train, test): списки (вопрос, ответ). Спрятанные (test) не пересекаются с train."""
    op_name, dims = kind.split("_")
    da, db = (int(x) for x in dims.split("x"))
    rng = random.Random(zlib.crc32(kind.encode()))
    seen, pairs = set(), []
    tries = 0
    while len(pairs) < n_train + n_test and tries < 50 * (n_train + n_test):
        tries += 1
        a, b = _rand_num(rng, da), _rand_num(rng, db)
        if op_name == "sub" and b > a:
            a, b = b, a
            if len(str(a)) != da or len(str(b)) != db:
                continue
        if op_name == "div":
            if b == 0:
                continue
            a = a - a % b                      # делится нацело
            if len(str(a)) != da:
                continue
        if (a, b) in seen:
            continue
        seen.add((a, b))
        ans = fmt(solve(a, {"add": "+", "sub": "-", "mul": "*", "div": "/"}[op_name], b))
        form = rng.choice(["{a} {op} {b}", "{a} {op} {b} =", "сколько будет {a} {op} {b}", "посчитай {a} {op} {b}"])
        pairs.append((form.format(a=a, op=SYM[op_name], b=b), ans))
    # Маленькие типы (например 2-значное ÷ 1-значное нацело — всего ~300 вариантов): делим 80/20.
    n_te = min(n_test, len(pairs) // 5)
    return pairs[:min(n_train, len(pairs) - n_te)], pairs[len(pairs) - n_te:]
