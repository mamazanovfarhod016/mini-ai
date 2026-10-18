"""
Проверка ответа Mini AI на задание «учись по тексту» (data/context_tasks.json).
Чистый скрипт, без ИИ — все ответы для contextcheck уже известны из генератора.

    check(item, generated) -> {"ok": bool, "errors": [str, ...]}

item      — запись из data/context_tasks.json ({context, q, a, kind, lang, answer, ...});
generated — то, что выдала модель (строка).

Правило: финальный ответ ищем ПОСЛЕ ключевого слова (Ответ/Javob/Answer) — так же,
как учим модель отвечать. Сравниваем с item["answer"] без учёта регистра/пробелов/
лишней пунктуации. Для kind == "derive" дополнительно проверяем, что в тексте
модели есть то же арифметическое уравнение («a + b = r»), что в правильном ответе
(если оно там есть) — иначе модель могла угадать финальное число без расчёта.
"""
import re

KEYWORD = {"uz": "Javob", "ru": "Ответ", "en": "Answer"}


def _norm(s):
    """Нижний регистр, один вид апострофа/тире, одиночные пробелы, без хвостовой точки."""
    s = str(s).strip().lower().replace("ё", "е")
    for a in "’ʻʼ`":
        s = s.replace(a, "'")
    s = s.replace("−", "-").replace("–", "-").replace("—", "-")
    s = re.sub(r"[.!?]+$", "", s)
    return " ".join(s.split())


def _extract_final(text, lang):
    """Текст после последнего ключевого слова (Ответ:/Javob:/Answer:), либо весь текст."""
    kw = KEYWORD.get(lang, "Ответ")
    pattern = re.compile(re.escape(kw) + r"\s*[:\-]?\s*(.+)$", re.IGNORECASE | re.DOTALL)
    m = pattern.search(text)
    if m:
        return m.group(1).strip()
    return str(text).strip()


_EQ_RE = re.compile(r"(-?\d+(?:[.,]\d+)?)\s*([+\-−*xх×/÷])\s*(-?\d+(?:[.,]\d+)?)\s*=\s*(-?\d+(?:[.,]\d+)?)")


def _find_equation(text):
    """Первое найденное уравнение «a op b = r» в тексте, либо None. (a, op, b, r) как числа."""
    m = _EQ_RE.search(text)
    if not m:
        return None
    a, op, b, r = m.groups()
    norm_op = {"−": "-", "x": "*", "х": "*", "×": "*", "÷": "/"}.get(op, op)
    try:
        a_f = float(a.replace(",", "."))
        b_f = float(b.replace(",", "."))
        r_f = float(r.replace(",", "."))
    except ValueError:
        return None
    return a_f, norm_op, b_f, r_f


def _eval_eq(a, op, b):
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if op == "/":
        return a / b if b else None
    return None


def check(item, generated):
    """→ {"ok": bool, "errors": [str, ...]}."""
    errors = []
    lang = item.get("lang", "ru")
    kind = item.get("kind", "fact")
    expected_answer = _norm(item.get("answer", ""))

    final = _extract_final(generated, lang)
    got_answer = _norm(final)

    # финальный ответ: разрешаем точное совпадение ИЛИ совпадение по первому "слову/числу"
    # (модель могла дописать лишнее после правильного ответа — сверяем начало).
    ans_ok = got_answer == expected_answer or got_answer.startswith(expected_answer) and expected_answer != ""
    if not ans_ok:
        errors.append(f"финальный ответ «{final}» != ожидаемому «{item.get('answer')}»")

    if kind == "derive":
        expected_eq = _find_equation(item.get("a", ""))
        if expected_eq is not None:
            got_eq = _find_equation(generated)
            if got_eq is None:
                errors.append("в ответе модели нет уравнения (a op b = r), а должно быть")
            else:
                a, op, b, r = got_eq
                calc = _eval_eq(a, op, b)
                if calc is None or abs(calc - r) > 1e-6:
                    errors.append(f"уравнение в ответе модели неверно: {a} {op} {b} != {r}")

    if kind == "not_in_text":
        # для not_in_text важно, чтобы модель НЕ выдумала конкретный факт вместо «в тексте нет»
        if got_answer and got_answer != expected_answer and not errors:
            errors.append("ожидали «в тексте этого нет», получили другой ответ")

    return {"ok": not errors, "errors": errors}
