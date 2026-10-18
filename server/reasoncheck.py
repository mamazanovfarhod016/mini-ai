"""
Проверка пошаговых рассуждений Mini AI (reasoning) — только скрипт, без ИИ.

check(item_or_q, generated_text, lang) -> {"answer_ok", "steps_ok", "check_ok", "errors": [...]}

Разбирает каждое уравнение вида «X op Y = Z» в сгенерированном тексте и
проверяет его арифметику через fractions.Fraction (целые, десятичные,
дроби a/b, отрицательные с «−»). Отдельно ищет строку с ключевым словом
«Проверка/Tekshirish/Check» (должна быть и быть верной) и финальный ответ
после «Ответ/Javob/Answer» (должен совпадать с истинным ответом задачи).
"""
import re
from fractions import Fraction

KW = {
    "ru": {"check": "Проверка", "answer": "Ответ"},
    "uz": {"check": "Tekshirish", "answer": "Javob"},
    "en": {"check": "Check", "answer": "Answer"},
}
ALL_KEYWORDS = {kw for d in KW.values() for kw in d.values()}

NUM_RE = r"-?\d+(?:\.\d+)?(?:/\d+)?"
TOKEN_RE = re.compile(rf"({NUM_RE})|([+\-*/])")
STMT_SPLIT_RE = re.compile(r"[,.]\s*")


def _to_frac(s):
    """«−3», «3/5», «2.5», «-3» → Fraction."""
    s = s.strip().replace("−", "-")
    if "/" in s:
        n, d = s.split("/")
        return Fraction(int(n), int(d))
    if "." in s:
        return Fraction(s)
    return Fraction(int(s))


def _eval_expr(s):
    """Вычислить простое выражение «a op b op c ...» (без скобок, стандартный
    приоритет * / перед + −) и вернуть Fraction, либо None, если не разобралось."""
    s = s.strip().replace("−", "-")
    toks = []
    for num, op in TOKEN_RE.findall(s):
        toks.append(num if num else op)
    # отфильтровать мусор: TOKEN_RE может не покрыть всю строку, проверим склейкой
    rebuilt = "".join(toks)
    if rebuilt.replace(" ", "") != s.replace(" ", ""):
        return None
    if not toks:
        return None
    # учесть унарный минус в начале (уже входит в NUM_RE через -?), но между operator и num тоже
    try:
        # первый проход: * и /
        i = 1
        vals = [_to_frac(toks[0])]
        ops = []
        while i < len(toks):
            op = toks[i]
            num = _to_frac(toks[i + 1])
            if op in ("*", "/"):
                prev = vals.pop()
                if op == "*":
                    vals.append(prev * num)
                else:
                    if num == 0:
                        return None
                    vals.append(prev / num)
            else:
                ops.append(op)
                vals.append(num)
            i += 2
        total = vals[0]
        for op, v in zip(ops, vals[1:]):
            total = total + v if op == "+" else total - v
        return total
    except Exception:
        return None


def _eval_eq_str(eq_str):
    """«a op b = c op d ...» → bool, либо None если не распарсилось."""
    if "=" not in eq_str:
        return None
    lhs, rhs = eq_str.split("=", 1)
    lv, rv = _eval_expr(lhs), _eval_expr(rhs)
    if lv is None or rv is None:
        return None
    return lv == rv


def find_equations(text):
    """Найти все уравнения (разделённые запятыми/точками) и проверить их. → [(eq_str, ok)]."""
    out = []
    for stmt in STMT_SPLIT_RE.split(text):
        stmt = stmt.strip().rstrip("✓").strip()
        if "=" in stmt and re.search(r"\d", stmt):
            ok = _eval_eq_str(stmt)
            if ok is not None:
                out.append((stmt, ok))
    return out


def _find_keyword_line(text, keyword):
    """Содержимое после 'keyword:' до следующей точки/конца строки, либо None."""
    m = re.search(re.escape(keyword) + r"\s*:\s*([^.]*)", text)
    return m.group(1).strip() if m else None


def _true_answer(item_or_q):
    if isinstance(item_or_q, dict):
        return str(item_or_q.get("answer", item_or_q.get("a", ""))).strip()
    return str(item_or_q).strip()


def check(item_or_q, generated_text, lang="ru"):
    """item_or_q: dict-элемент data/reasoning.json (с полем answer) или готовая строка-ответ.
    generated_text: то, что выдала модель. lang: uz|ru|en — какие ключевые слова искать."""
    errors = []
    kw = KW.get(lang, KW["ru"])
    text = str(generated_text)

    eqs = find_equations(text)
    steps_ok = len(eqs) > 0 and all(ok for _, ok in eqs)
    if not eqs:
        errors.append("no_equations_found")
    for eq_str, ok in eqs:
        if not ok:
            errors.append(f"wrong_step: {eq_str}")

    check_line = _find_keyword_line(text, kw["check"])
    check_ok = False
    if check_line is None:
        errors.append("missing_check_line")
    else:
        check_eqs = find_equations(check_line)
        if not check_eqs:
            errors.append("check_line_has_no_equation")
        else:
            check_ok = all(ok for _, ok in check_eqs)
            if not check_ok:
                errors.append("check_line_wrong")

    answer_line = _find_keyword_line(text, kw["answer"])
    answer_ok = False
    true_answer = _true_answer(item_or_q)
    if answer_line is None:
        errors.append("missing_answer_line")
    else:
        got = answer_line.strip().rstrip(" ✓").strip()
        answer_ok = got == true_answer
        if not answer_ok:
            # попытка сравнить как дроби/числа, если текстовое совпадение не прошло
            try:
                if "," in got or "," in true_answer:
                    raise ValueError
                answer_ok = _to_frac(got.split(",")[0]) == _to_frac(true_answer.split(",")[0])
            except Exception:
                answer_ok = False
            if not answer_ok:
                errors.append(f"wrong_answer: got={got!r} expected={true_answer!r}")

    return {"answer_ok": answer_ok, "steps_ok": steps_ok, "check_ok": check_ok, "errors": errors}
