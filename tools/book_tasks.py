"""Задания из УЧЕБНИКОВ → обучение по четвертям (19.10.2026, решение Фархода: «по книге, 1 четверть 1 класса,
пока не пройдёт — дальше нельзя»; «не создавай сам» — задания только из книг).

    .venv/bin/python tools/book_tasks.py     # → ~/Developer/mini-ai-books/tasks.json + сводка по четвертям

Что берём из текста урока (~/Developer/mini-ai-books/lessons.json, tools/books_to_lessons.py):
  • примеры с пропуском ответа:   «8 − 1 =», «7 + 2 = …», «45 : 5 = ?»  → ответ считает Python (проверка, не выдумка);
  • примеры с напечатанным ответом: «6 + 3 = 9» → берём, только если Python подтверждает, что ответ верный;
  • цепочки:  «5 + 3 − 2 =» — ответ с шагами слева направо.
Порча шрифта в некоторых PDF 1–2 класса: знак минус напечатан как «$» → чиним (только между числами).
Экзамен: ~20% заданий каждой четверти спрятаны (crc32 вопроса) — модель их не видит в обучении.
"""
import json
import re
import zlib
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

BOOKS = Path.home() / "Developer/mini-ai-books"
TEST_FRAC = 0.2
M = "−"
NUM = r"\d{1,6}"
OPS = {"+": "+", "-": M, "–": M, "−": M, "$": M, "×": "*", "·": "*", "*": "*", "x": "*", ":": "/", "÷": "/"}
OP = r"[+\-–−$×·*:÷]"
# a op b [op c [op d]] = [ответ | пусто | … | ? | _]
EXPR = re.compile(rf"(?<![\d/.,])({NUM})\s*({OP})\s*({NUM})(?:\s*({OP})\s*({NUM}))?(?:\s*({OP})\s*({NUM}))?"
                  rf"\s*=\s*({NUM}|\.\.\.|…|_+|\?|□)?(?![\d/.,])")
KW = {"ru": ("Ответ", "по порядку слева направо"), "uz": ("Javob", "chapdan o'ngga")}


def calc(nums, ops):
    """Порядок действий: сначала * и /, потом + и − слева направо. → Fraction или None."""
    vals, sig = [Fraction(nums[0])], []
    for op, x in zip(ops, nums[1:]):
        x = Fraction(x)
        if op in "*/":
            if op == "/" and x == 0:
                return None
            vals[-1] = vals[-1] * x if op == "*" else vals[-1] / x
        else:
            sig.append(op)
            vals.append(x)
    tot = vals[0]
    for op, v in zip(sig, vals[1:]):
        tot = tot + v if op == "+" else tot - v
    return tot


def fmt(v):
    return (str(v.numerator) if v.denominator == 1 else f"{v.numerator}/{v.denominator}").replace("-", M)


def tasks_from_text(text):
    out = []
    for m in EXPR.finditer(text):
        nums = [m.group(i) for i in (1, 3, 5, 7) if m.group(i)]
        ops = [OPS[m.group(i)] for i in (2, 4, 6) if m.group(i)]
        if len(ops) != len(nums) - 1:
            continue
        v = calc(nums, ops)
        if v is None or v.denominator != 1 or v < 0:   # в начальной школе ответы — целые неотрицательные
            continue
        printed = m.group(8)
        if printed and printed.isdigit() and Fraction(printed) != v:
            continue   # в книге напечатан другой ответ (задание «найди ошибку» или кривой распознанный текст) — не берём
        q = nums[0] + "".join(f" {o} {x}" for o, x in zip(ops, nums[1:]))
        out.append((q, v, ops))
    return out


def answer(q, v, ops, lang):
    if len(ops) == 1:
        return fmt(v)
    # цепочка без * и / — шаги слева направо (как считают в книге)
    if all(o in "+" + M for o in ops):
        nums = re.findall(r"\d+", q)
        cur, steps = Fraction(nums[0]), []
        for o, x in zip(ops, nums[1:]):
            new = cur + int(x) if o == "+" else cur - int(x)
            steps.append(f"{fmt(cur)} {o} {x} = {fmt(new)}")
            cur = new
        return f"{', '.join(steps)}. {KW[lang][0]}: {fmt(v)}"
    return fmt(v)


def main():
    lessons = json.loads((BOOKS / "lessons.json").read_text())
    by_q = defaultdict(dict)   # (grade, quarter) → {(q, lang): item}
    for ls in lessons:
        if ls["subj"] not in ("math", "algebra", "geometry"):
            continue
        for q, v, ops in tasks_from_text(ls["text"]):
            key = (q, ls["lang"])
            stage = (ls["grade"], ls["quarter"])
            # одно и то же задание — в самой ранней четверти, где оно встретилось
            if any(key in by_q[s] for s in by_q if s <= stage):
                continue
            hide = zlib.crc32(f"{ls['grade']}:{q}".encode()) % 1000 < TEST_FRAC * 1000
            by_q[stage][key] = {"q": q, "a": answer(q, v, ops, ls["lang"]), "answer": fmt(v), "lang": ls["lang"],
                                "grade": ls["grade"], "quarter": ls["quarter"], "kind": f"book_g{ls['grade']}q{ls['quarter']}",
                                "book": ls["book"], "lesson": ls["lesson"], "test": hide}
    stages = []
    for st in sorted(by_q):
        items = list(by_q[st].values())
        stages.append({"grade": st[0], "quarter": st[1],
                       "train": [x for x in items if not x["test"]], "test": [x for x in items if x["test"]]})
    (BOOKS / "tasks.json").write_text(json.dumps(stages, ensure_ascii=False))
    for s in stages:
        print(f"{s['grade']:2d} кл, {s['quarter']} четв.: учить {len(s['train']):4d}, экзамен {len(s['test']):3d}")


if __name__ == "__main__":
    main()
