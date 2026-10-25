"""Увеличение объёма заданий ТОЛЬКО для текущей четверти (19.10.2026, решение Фархода: «увеличь в несколько раз,
но не всё сразу с 1 по 11 — давать огромные материалы на этот этап, потом удалять, к следующей четверти — заново»).

Правило: виды заданий и диапазон чисел берём ИЗ КНИГИ этой четверти (какие действия, сколько чисел, до какого числа),
подставляем новые числа; правильный ответ считает Python. Темы и порядок — как в учебнике, ничего сверх.
+ ~10% «ошибка → проверка → исправление» (исследование Physics of LMs 2.2: так модель решает точнее).
Задания, совпавшие с экзаменом (спрятанные задания книги), выкидываем — экзамен честный.
"""
import random
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from book_tasks import answer, calc, fmt  # noqa: E402

M = "−"
CHK = {"ru": ("Проверка", "неверно", "Ответ"), "uz": ("Tekshirish", "noto'g'ri", "Javob")}
INV = {"+": M, M: "+"}


def _shape(item):
    """«12 + 7 − 3» → (('+', '−'), [12, 7, 3])."""
    toks = item["q"].split()
    return tuple(toks[1::2]), [int(t) for t in toks[0::2]]


def augment(cur, factor=100, cap=20000, seed=0, banned_q=()):
    rng = random.Random(seed)
    lim = defaultdict(lambda: {"num": [], "res": 0})   # вид → max каждого числа и результата в КНИГЕ
    for x in cur["train"] + cur["test"]:
        ops, nums = _shape(x)
        L = lim[ops]
        L["num"] = [max(a, b) for a, b in zip(L["num"] + [0] * len(nums), nums)] if L["num"] else nums[:]
        L["res"] = max(L["res"], int(Fraction(x["answer"].replace(M, "-"))))
    banned = {x["q"] for x in cur["test"]} | set(banned_q)   # ни одного экзаменационного задания любой четверти
    have = {(x["q"], x["lang"]) for x in cur["train"]}
    langs = sorted({x["lang"] for x in cur["train"]}) or ["ru", "uz"]
    target = min(cap, factor * max(1, len(cur["train"])))
    out, tries = [], 0
    kinds = list(lim)
    while len(out) < target and tries < target * 50:
        tries += 1
        ops = rng.choice(kinds)
        L = lim[ops]
        nums = [rng.randint(0, max(1, m)) for m in L["num"]]
        if "/" in ops:
            # 20.10: раньше делимое бралось случайно → делилось нацело почти только 0 (0 / 1, 0 / 2…), и сеть
            # не видела настоящего деления (экзамен 45 / 9 → 6, 2 попытки по 33%). Теперь делимое = делитель × частное
            # в пределах книги, без нулевых делимых.
            for k, o in enumerate(ops):
                if o == "/":
                    nums[k + 1] = max(1, nums[k + 1])
                    hi = max(1, L["num"][k] // nums[k + 1])
                    nums[k] = nums[k + 1] * rng.randint(1, hi)
        if "/" in ops and nums[1] == 0:
            continue
        v = calc([str(n) for n in nums], list(ops))
        if v is None or v.denominator != 1 or v < 0 or v > max(L["res"], 1):
            continue
        q = str(nums[0]) + "".join(f" {o} {n}" for o, n in zip(ops, nums[1:]))
        if q in banned:
            continue
        lang = rng.choice(langs)
        if (q, lang) in have:
            continue
        have.add((q, lang))
        a = answer(q, v, list(ops), lang)
        if len(ops) == 1 and ops[0] in INV and rng.random() < 0.10:
            chk, bad, ans = CHK[lang]
            wrong = int(v) + rng.choice([-1, 1])
            if wrong >= 0:
                back = wrong - nums[1] if ops[0] == "+" else wrong + nums[1]
                a = (f"{q} = {wrong}? {chk}: {wrong} {INV[ops[0]]} {nums[1]} = {fmt(Fraction(back))} ≠ {nums[0]}, {bad}. "
                     f"{q} = {fmt(v)}. {ans}: {fmt(v)}")
        out.append({"q": q, "a": a, "answer": fmt(v), "lang": lang, "kind": cur["train"][0]["kind"] if cur["train"] else "book"})
    # 25.10: ГРАНИЦА деления — экзамен «81 / 9», «36 / 4 + 21» проваливался 4 попытки подряд даже после x9-
    # повтора того, что было сгенерировано: divisor=max И quotient=max одновременно — это ДВА независимых
    # случайных максимума, при случайной генерации почти никогда не выпадают вместе (видели max 8 раз из 115
    # чистых делений в попытке 5, и НИ РАЗУ вариант 81 / 9). Добираем недостающую клетку сетки «делитель × частное»
    # до верхней границы книги ДЕТЕРМИНИРОВАННО (не случайно) — каждая комбинация хотя бы раз, остальные
    # операнды — на границе книги (max) и на 0, чтобы «81 / 9» и «36 / 4 + 21» точно встретились в обучении.
    for ops, L in lim.items():
        if "/" not in ops:
            continue
        di = ops.index("/")
        max_div = max(1, L["num"][di + 1])
        max_q = max(1, L["res"])
        other = [k for k in range(len(L["num"])) if k not in (di, di + 1)]
        for divisor in range(1, max_div + 1):
            for quotient in range(1, max_q + 1):
                dividend = divisor * quotient
                if dividend > L["num"][di]:
                    continue
                for extreme in ({0} | {L["num"][k] for k in other} if other else {0}):
                    nums = [0] * len(L["num"])
                    nums[di], nums[di + 1] = dividend, divisor
                    for k in other:
                        nums[k] = extreme
                    v = calc([str(n) for n in nums], list(ops))
                    if v is None or v.denominator != 1 or v < 0 or v > max(L["res"], 1):
                        continue
                    q = str(nums[0]) + "".join(f" {o} {n}" for o, n in zip(ops, nums[1:]))
                    if q in banned:
                        continue
                    for lang in langs:
                        if (q, lang) in have:
                            continue
                        have.add((q, lang))
                        a = answer(q, v, list(ops), lang)
                        out.append({"q": q, "a": a, "answer": fmt(v), "lang": lang,
                                    "kind": cur["train"][0]["kind"] if cur["train"] else "book"})
        # 25.10 (2-й разбор): для деления «а / б» сеть должна бы обобщить «частное 9» через факт умножения
        # «б × 9 = а», но умножение в augment ограничено максимальным РЕЗУЛЬТАТОМ из книги (здесь — 63, т.к.
        # в книге умножение дано только до 9 × 7), а деление из той же четверти требует делимое до 81 (9 × 9).
        # Сеть НИКОГДА не видела факта «9 × 9 = 81» ни в одном задании этой четверти — не промах генератора,
        # а разрыв между диапазонами умножения и деления в одной и той же книге. Добавляем факты умножения
        # «делитель × частное = делимое» для ВСЕЙ той же сетки (делитель × частное из деления), без ограничения
        # по старому max умножения — это не новая тема, а прямая подпорка уже изучаемого деления.
        if ops == ("/",):
            for divisor in range(1, max_div + 1):
                for quotient in range(1, max_q + 1):
                    dividend = divisor * quotient
                    if dividend > L["num"][di]:
                        continue
                    for a_, b_ in ((divisor, quotient), (quotient, divisor)):
                        q = f"{a_} * {b_}"
                        if q in banned:
                            continue
                        for lang in langs:
                            if (q, lang) in have:
                                continue
                            have.add((q, lang))
                            a = answer(q, Fraction(dividend), ["*"], lang)
                            out.append({"q": q, "a": a, "answer": fmt(Fraction(dividend)), "lang": lang,
                                        "kind": cur["train"][0]["kind"] if cur["train"] else "book"})
    return out
