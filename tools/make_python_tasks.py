"""
Задачи по программированию на Python для Mini AI: модель пишет КОД (одну функцию),
а не просто ответ-число. Эталоны (`a`) всегда пишет скрипт/шаблон — никогда ИИ.

    .venv/bin/python tools/make_python_tasks.py   # → data/python_tasks.json

Формат как у data/school_math.json: {"train": [...], "test": [...]}, каждая запись —
    {q, a, kind, lang, func_name, tests: [{args, expected}], hidden_tests: [...]}
q — одна и та же задача на uz (латиница, обычный апостроф '), ru, en.
a — код функции (для kind="explain"/"print" — одна строка объяснения/результат вывода).

Стиль эталонов описан в docs/PYTHON_STYLE.md, его же проверяет server/codecheck.py.

ЧЕСТНЫЙ ЭКЗАМЕН: каждая задача параметризована (имя функции, имена переменных,
константы, конкретные числа/слова). Параметры образуют "семью" — набор конкретных
значений. Семья целиком (во всех языках и со всеми тестами) прячется в test по crc32
от ключа семьи — модель никогда не видит эти конкретные параметры на обучении.
Если она решает их на экзамене — она поняла СТРУКТУРУ задачи, а не запомнила код.

ВАЖНО для интеграции (models/transformer.py: clean_question лежит в mini_ai/transformer.py):
clean_question() ЛОМАЕТ код — она lower()-ит текст и выбрасывает пунктуацию/скобки.
Для кодовых задач НЕЛЬЗЯ пропускать вопрос через clean_question(): пропадут регистр
имён (CamelCase), скобки, двоеточия, кавычки — код перестанет быть кодом.
Функция prompt_text(q) ниже — чисто документальная заглушка: она помечает, что строка
`q` для kind из PYTHON_KINDS должна попадать модели СЫРОЙ (без clean_question), в отличие
от школьной математики. Сам clean_question трогать НЕЛЬЗЯ (общий путь для всех задач) —
это дело интегратора на стороне app.py/транспорта, решить, как не чистить код-вопросы.
"""
from __future__ import annotations

import json
import random
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LANGS = ("uz", "ru", "en")
TEST_FRAC = 0.15

PYTHON_KINDS = (
    "sum_list", "count_vowels", "reverse_words", "is_palindrome", "max_no_builtin",
    "fizzbuzz", "factorial", "gcd", "c2f", "average_guard",
    "filter_evens", "word_freq", "group_first_letter", "merge_dicts",
    "fix_bug", "what_prints", "cleanup",
)


def prompt_text(q: str) -> str:
    """Заглушка-документация: для kind из PYTHON_KINDS подавать модели q КАК ЕСТЬ,
    без mini_ai.transformer.clean_question (она убирает регистр/скобки/пунктуацию
    и ломает код). Сюда ничего не передаётся — это маркер для интегратора."""
    return q


def is_hidden(kind: str, family_key) -> bool:
    """Детерминированно: эта семья параметров спрятана в экзамен? Одинаково для всех языков."""
    return zlib.crc32(f"{kind}:{family_key}".encode()) % 100 < TEST_FRAC * 100


# --------------------------------------------------------------------------------------
# Пулы имён для параметризации (разные тексты функций/переменных у разных семей)
# --------------------------------------------------------------------------------------
FUNC_NAMES = {
    "sum_list": ["sum_list", "total_of", "add_all", "list_sum"],
    "count_vowels": ["count_vowels", "vowel_count", "num_vowels"],
    "reverse_words": ["reverse_words", "flip_words", "words_reversed"],
    "is_palindrome": ["is_palindrome", "is_mirror", "reads_same_backward"],
    "max_no_builtin": ["largest", "find_max", "biggest_value"],
    "fizzbuzz": ["fizzbuzz_word", "label_number", "classify_number"],
    "factorial": ["factorial", "fact_of"],
    "gcd": ["gcd_of", "greatest_common_divisor"],
    "c2f": ["celsius_to_fahrenheit", "to_fahrenheit"],
    "average_guard": ["average", "mean_of", "safe_average"],
    "filter_evens": ["filter_evens", "even_numbers", "only_evens"],
    "word_freq": ["word_frequency", "count_words"],
    "group_first_letter": ["group_by_first_letter", "group_by_letter"],
    "merge_dicts": ["merge_dicts", "combine_dicts"],
}

Q = {
    "sum_list": {
        "uz": "{fn}(lst) funksiyasini yoz: sonlar ro'yxatining yig'indisini qaytaradi",
        "ru": "напиши функцию {fn}(lst): возвращает сумму списка чисел",
        "en": "write a function {fn}(lst) that returns the sum of a list of numbers",
    },
    "count_vowels": {
        "uz": "{fn}(s) funksiyasini yoz: satrdagi unlilar sonini qaytaradi ({vowels})",
        "ru": "напиши функцию {fn}(s): возвращает количество гласных в строке ({vowels})",
        "en": "write a function {fn}(s) that returns the number of vowels in a string ({vowels})",
    },
    "reverse_words": {
        "uz": "{fn}(s) funksiyasini yoz: so'zlar tartibini teskari qaytaradi",
        "ru": "напиши функцию {fn}(s): возвращает слова строки в обратном порядке",
        "en": "write a function {fn}(s) that returns the words of a string in reverse order",
    },
    "is_palindrome": {
        "uz": "{fn}(s) funksiyasini yoz: satr palindrom bo'lsa True qaytaradi",
        "ru": "напиши функцию {fn}(s): возвращает True, если строка палиндром",
        "en": "write a function {fn}(s) that returns True if the string is a palindrome",
    },
    "max_no_builtin": {
        "uz": "{fn}(lst) funksiyasini yoz: max() dan foydalanmasdan ro'yxatdagi eng katta sonni toping",
        "ru": "напиши функцию {fn}(lst): найди наибольшее число в списке без max()",
        "en": "write a function {fn}(lst) that finds the largest number in a list without using max()",
    },
    "fizzbuzz": {
        "uz": "{fn}(n) funksiyasini yoz: {d1} ga bo'linsa '{w1}', {d2} ga bo'linsa '{w2}', ikkisiga ham bo'linsa '{w1}{w2}', bo'lmasa son qaytaradi",
        "ru": "напиши функцию {fn}(n): если делится на {d1} — '{w1}', на {d2} — '{w2}', на оба — '{w1}{w2}', иначе само число",
        "en": "write a function {fn}(n) that returns '{w1}' if divisible by {d1}, '{w2}' if by {d2}, '{w1}{w2}' if both, else the number",
    },
    "factorial": {
        "uz": "{fn}(n) funksiyasini yoz: n ning faktorialini qaytaradi",
        "ru": "напиши функцию {fn}(n): возвращает факториал числа n",
        "en": "write a function {fn}(n) that returns the factorial of n",
    },
    "gcd": {
        "uz": "{fn}(a, b) funksiyasini yoz: a va b ning eng katta umumiy bo'luvchisini qaytaradi",
        "ru": "напиши функцию {fn}(a, b): возвращает наибольший общий делитель a и b",
        "en": "write a function {fn}(a, b) that returns the greatest common divisor of a and b",
    },
    "c2f": {
        "uz": "{fn}(c) funksiyasini yoz: Selsiyni Farengeytga aylantiradi",
        "ru": "напиши функцию {fn}(c): переводит градусы Цельсия в Фаренгейт",
        "en": "write a function {fn}(c) that converts Celsius to Fahrenheit",
    },
    "average_guard": {
        "uz": "{fn}(lst) funksiyasini yoz: ro'yxat o'rtachasini qaytaradi, bo'sh bo'lsa 0 qaytaradi",
        "ru": "напиши функцию {fn}(lst): возвращает среднее значение списка, а для пустого списка 0",
        "en": "write a function {fn}(lst) that returns the average of a list, or 0 for an empty list",
    },
    "filter_evens": {
        "uz": "{fn}(lst) funksiyasini yoz: ro'yxatdagi juft sonlarni qaytaradi",
        "ru": "напиши функцию {fn}(lst): возвращает только чётные числа из списка",
        "en": "write a function {fn}(lst) that returns only the even numbers from a list",
    },
    "word_freq": {
        "uz": "{fn}(s) funksiyasini yoz: satrdagi har bir so'z necha marta takrorlanganini lug'at qilib qaytaradi",
        "ru": "напиши функцию {fn}(s): возвращает словарь — сколько раз встречается каждое слово в строке",
        "en": "write a function {fn}(s) that returns a dict with the count of each word in a string",
    },
    "group_first_letter": {
        "uz": "{fn}(words) funksiyasini yoz: so'zlarni birinchi harfi bo'yicha lug'atga guruhlaydi",
        "ru": "напиши функцию {fn}(words): группирует слова в словарь по первой букве",
        "en": "write a function {fn}(words) that groups words into a dict keyed by first letter",
    },
    "merge_dicts": {
        "uz": "{fn}(d1, d2) funksiyasini yoz: ikki lug'atni birlashtiradi (d2 ustunlik qiladi)",
        "ru": "напиши функцию {fn}(d1, d2): объединяет два словаря (d2 имеет приоритет)",
        "en": "write a function {fn}(d1, d2) that merges two dicts (d2 values take priority)",
    },
}

# --------------------------------------------------------------------------------------
# Эталонные решения — код-шаблоны по стилю docs/PYTHON_STYLE.md
# --------------------------------------------------------------------------------------
A = {
    "sum_list": "def {fn}(numbers: list[float]) -> float:\n"
                "    \"\"\"Возвращает сумму чисел списка.\"\"\"\n"
                "    total = 0\n"
                "    for number in numbers:\n"
                "        total += number\n"
                "    return total\n",
    "count_vowels": "def {fn}(text: str) -> int:\n"
                     "    \"\"\"Считает гласные {vowels} в строке.\"\"\"\n"
                     "    vowels = \"{vowels_set}\"\n"
                     "    return sum(1 for char in text.lower() if char in vowels)\n",
    "reverse_words": "def {fn}(text: str) -> str:\n"
                      "    \"\"\"Возвращает слова строки в обратном порядке.\"\"\"\n"
                      "    return \" \".join(reversed(text.split()))\n",
    "is_palindrome": "def {fn}(text: str) -> bool:\n"
                      "    \"\"\"Проверяет, является ли строка палиндромом.\"\"\"\n"
                      "    cleaned = text.lower().replace(\" \", \"\")\n"
                      "    return cleaned == cleaned[::-1]\n",
    "max_no_builtin": "def {fn}(numbers: list[float]) -> float:\n"
                       "    \"\"\"Находит наибольшее число без max().\"\"\"\n"
                       "    best = numbers[0]\n"
                       "    for number in numbers[1:]:\n"
                       "        if number > best:\n"
                       "            best = number\n"
                       "    return best\n",
    "fizzbuzz": "def {fn}(n: int) -> str:\n"
                "    \"\"\"Возвращает метку по делимости на {d1} и {d2}.\"\"\"\n"
                "    div_a, div_b = {d1}, {d2}\n"
                "    label = \"\"\n"
                "    if n % div_a == 0:\n"
                "        label += \"{w1}\"\n"
                "    if n % div_b == 0:\n"
                "        label += \"{w2}\"\n"
                "    return label or str(n)\n",
    "factorial": "def {fn}(n: int) -> int:\n"
                 "    \"\"\"Возвращает факториал n.\"\"\"\n"
                 "    result = 1\n"
                 "    for value in range(2, n + 1):\n"
                 "        result *= value\n"
                 "    return result\n",
    "gcd": "def {fn}(a: int, b: int) -> int:\n"
           "    \"\"\"Возвращает наибольший общий делитель a и b.\"\"\"\n"
           "    while b:\n"
           "        a, b = b, a % b\n"
           "    return a\n",
    "c2f": "def {fn}(celsius: float) -> float:\n"
           "    \"\"\"Переводит градусы Цельсия в Фаренгейт.\"\"\"\n"
           "    SCALE, OFFSET = 9 / 5, 32\n"
           "    return celsius * SCALE + OFFSET\n",
    "average_guard": "def {fn}(numbers: list[float]) -> float:\n"
                      "    \"\"\"Возвращает среднее списка, 0 для пустого списка.\"\"\"\n"
                      "    if not numbers:\n"
                      "        return 0\n"
                      "    return sum(numbers) / len(numbers)\n",
    "filter_evens": "def {fn}(numbers: list[int]) -> list[int]:\n"
                     "    \"\"\"Возвращает только чётные числа списка.\"\"\"\n"
                     "    return [number for number in numbers if number % 2 == 0]\n",
    "word_freq": "def {fn}(text: str) -> dict[str, int]:\n"
                  "    \"\"\"Считает, сколько раз встречается каждое слово.\"\"\"\n"
                  "    counts: dict[str, int] = {{}}\n"
                  "    for word in text.lower().split():\n"
                  "        counts[word] = counts.get(word, 0) + 1\n"
                  "    return counts\n",
    "group_first_letter": "def {fn}(words: list[str]) -> dict[str, list[str]]:\n"
                           "    \"\"\"Группирует слова в словарь по первой букве.\"\"\"\n"
                           "    groups: dict[str, list[str]] = {{}}\n"
                           "    for word in words:\n"
                           "        groups.setdefault(word[0], []).append(word)\n"
                           "    return groups\n",
    "merge_dicts": "def {fn}(first: dict, second: dict) -> dict:\n"
                   "    \"\"\"Объединяет два словаря, второй имеет приоритет.\"\"\"\n"
                   "    merged = dict(first)\n"
                   "    merged.update(second)\n"
                   "    return merged\n",
}

VOWEL_SETS = ["aeiou", "aeiouy", "aeiouAEIOU".lower()]
FIZZ_WORDS = [("fizz", "buzz"), ("ping", "pong"), ("tap", "top"), ("left", "right")]
FIZZ_DIVS = [(3, 5), (2, 7), (3, 4), (2, 5)]


def make_test(kind, **kw):
    """Возвращает список тестов {args, expected} для одной семьи параметров."""
    if kind == "sum_list":
        return [{"args": [kw["lst"]], "expected": sum(kw["lst"])}]
    if kind == "count_vowels":
        vowels = kw["vowels_set"]
        return [{"args": [kw["s"]], "expected": sum(1 for c in kw["s"].lower() if c in vowels)}]
    if kind == "reverse_words":
        return [{"args": [kw["s"]], "expected": " ".join(reversed(kw["s"].split()))}]
    if kind == "is_palindrome":
        cleaned = kw["s"].lower().replace(" ", "")
        return [{"args": [kw["s"]], "expected": cleaned == cleaned[::-1]}]
    if kind == "max_no_builtin":
        return [{"args": [kw["lst"]], "expected": max(kw["lst"])}]
    if kind == "fizzbuzz":
        d1, d2, w1, w2, n = kw["d1"], kw["d2"], kw["w1"], kw["w2"], kw["n"]
        label = (w1 if n % d1 == 0 else "") + (w2 if n % d2 == 0 else "")
        return [{"args": [n], "expected": label or str(n)}]
    if kind == "factorial":
        n = kw["n"]
        r = 1
        for v in range(2, n + 1):
            r *= v
        return [{"args": [n], "expected": r}]
    if kind == "gcd":
        import math
        return [{"args": [kw["a"], kw["b"]], "expected": math.gcd(kw["a"], kw["b"])}]
    if kind == "c2f":
        c = kw["c"]
        return [{"args": [c], "expected": c * 9 / 5 + 32}]
    if kind == "average_guard":
        lst = kw["lst"]
        return [{"args": [lst], "expected": (sum(lst) / len(lst)) if lst else 0}]
    if kind == "filter_evens":
        lst = kw["lst"]
        return [{"args": [lst], "expected": [x for x in lst if x % 2 == 0]}]
    if kind == "word_freq":
        s = kw["s"]
        counts = {}
        for w in s.lower().split():
            counts[w] = counts.get(w, 0) + 1
        return [{"args": [s], "expected": counts}]
    if kind == "group_first_letter":
        words = kw["words"]
        groups = {}
        for w in words:
            groups.setdefault(w[0], []).append(w)
        return [{"args": [words], "expected": groups}]
    if kind == "merge_dicts":
        d1, d2 = kw["d1_v"], kw["d2_v"]
        merged = dict(d1)
        merged.update(d2)
        return [{"args": [d1, d2], "expected": merged}]
    raise ValueError(kind)


def build_basic(rng, kind, train, test):
    """Строит множество параметризованных семей для одного из базовых kind."""
    names = FUNC_NAMES[kind]
    for i in range(500):
        fn = names[i % len(names)]
        if kind == "sum_list":
            lst = [rng.randint(-50, 50) for _ in range(rng.randint(1, 6))]
            q_kw = {"fn": fn}
            fam_key = (fn,)  # q содержит только fn — семья хранится по имени функции
            tests = make_test(kind, lst=lst)
            a = A[kind].format(fn=fn)
        elif kind == "count_vowels":
            vowels = VOWEL_SETS[i % len(VOWEL_SETS)]
            words = ["banana", "stereo", "umbrella", "yogurt", "picture", "orange"]
            s = words[i % len(words)] + str(i)
            q_kw = {"fn": fn, "vowels": vowels}
            fam_key = (fn, vowels)  # vowels виден в q, конкретная строка s — нет
            tests = make_test(kind, s=s, vowels_set=vowels)
            a = A[kind].format(fn=fn, vowels=vowels, vowels_set=vowels)
        elif kind == "reverse_words":
            s = f"word{i} another{i} last{i}"
            q_kw = {"fn": fn}
            fam_key = (fn,)
            tests = make_test(kind, s=s)
            a = A[kind].format(fn=fn)
        elif kind == "is_palindrome":
            base = ["level", "radar", "hello", "kayak", "python", "noon"][i % 6]
            s = base + (base[::-1] if i % 2 else str(i))
            q_kw = {"fn": fn}
            fam_key = (fn,)
            tests = make_test(kind, s=s)
            a = A[kind].format(fn=fn)
        elif kind == "max_no_builtin":
            lst = [rng.randint(-100, 100) for _ in range(rng.randint(2, 6))]
            q_kw = {"fn": fn}
            fam_key = (fn,)
            tests = make_test(kind, lst=lst)
            a = A[kind].format(fn=fn)
        elif kind == "fizzbuzz":
            d1, d2 = FIZZ_DIVS[i % len(FIZZ_DIVS)]
            w1, w2 = FIZZ_WORDS[i % len(FIZZ_WORDS)]
            n = rng.randint(1, 100)
            q_kw = {"fn": fn, "d1": d1, "d2": d2, "w1": w1, "w2": w2}
            fam_key = (fn, d1, d2, w1, w2)  # n не виден в q — не входит в ключ семьи
            tests = make_test(kind, d1=d1, d2=d2, w1=w1, w2=w2, n=n)
            a = A[kind].format(fn=fn, d1=d1, d2=d2, w1=w1, w2=w2)
        elif kind == "factorial":
            n = rng.randint(0, 10)
            q_kw = {"fn": fn}
            fam_key = (fn,)
            tests = make_test(kind, n=n)
            a = A[kind].format(fn=fn)
        elif kind == "gcd":
            a_, b_ = rng.randint(1, 200), rng.randint(1, 200)
            q_kw = {"fn": fn}
            fam_key = (fn,)
            tests = make_test(kind, a=a_, b=b_)
            a = A[kind].format(fn=fn)
        elif kind == "c2f":
            c = rng.randint(-40, 100)
            q_kw = {"fn": fn}
            fam_key = (fn,)
            tests = make_test(kind, c=c)
            a = A[kind].format(fn=fn)
        elif kind == "average_guard":
            lst = [] if i % 10 == 0 else [rng.randint(-50, 50) for _ in range(rng.randint(1, 6))]
            q_kw = {"fn": fn}
            fam_key = (fn,)
            tests = make_test(kind, lst=lst)
            a = A[kind].format(fn=fn)
        elif kind == "filter_evens":
            lst = [rng.randint(-20, 20) for _ in range(rng.randint(1, 8))]
            q_kw = {"fn": fn}
            fam_key = (fn,)
            tests = make_test(kind, lst=lst)
            a = A[kind].format(fn=fn)
        elif kind == "word_freq":
            s = " ".join(rng.choice(["cat", "dog", "cat", "bird", "dog", "dog"]) for _ in range(rng.randint(3, 7))) + str(i)
            q_kw = {"fn": fn}
            fam_key = (fn,)
            tests = make_test(kind, s=s)
            a = A[kind].format(fn=fn)
        elif kind == "group_first_letter":
            pool = ["apple", "ant", "bear", "cat", "car", "dog", "deer", "ant" + str(i)]
            words = rng.sample(pool, k=rng.randint(2, len(pool)))
            q_kw = {"fn": fn}
            fam_key = (fn,)
            tests = make_test(kind, words=words)
            a = A[kind].format(fn=fn)
        elif kind == "merge_dicts":
            d1_v = {f"k{j}": rng.randint(0, 9) for j in range(rng.randint(1, 4))}
            d2_v = {f"k{j}": rng.randint(0, 9) for j in range(i % 3, i % 3 + rng.randint(1, 3))}
            q_kw = {"fn": fn}
            fam_key = (fn,)
            tests = make_test(kind, d1_v=d1_v, d2_v=d2_v)
            a = A[kind].format(fn=fn)
        else:
            raise ValueError(kind)

        hide = is_hidden(kind, fam_key)
        for lang in LANGS:
            rec = {
                "q": Q[kind][lang].format(**q_kw),
                "a": a,
                "kind": kind,
                "lang": lang,
                "func_name": fn,
                "tests": tests,
                "hidden_tests": tests,
            }
            (test if hide else train).append(rec)


# --------------------------------------------------------------------------------------
# "Что выведет этот код" — ответ считает сам Python (запуск в sandbox при генерации)
# --------------------------------------------------------------------------------------
PRINT_SNIPPETS = [
    ("x = {a}\ny = {b}\nprint(x + y)", lambda a, b: a + b),
    ("values = [{a}, {b}, {c}]\nprint(sum(values))", lambda a, b, c: a + b + c),
    ("text = '{w}'\nprint(text[::-1])", lambda w: None),  # вычисляется ниже
    ("n = {a}\nprint(n % {b})", lambda a, b: a % b),
    ("items = [{a}, {b}, {c}]\nprint(len(items))", lambda a, b, c: 3),
]
PRINT_Q = {
    "uz": "bu kod nima chiqaradi:\n{code}",
    "ru": "что выведет этот код:\n{code}",
    "en": "what does this code print:\n{code}",
}


def build_what_prints(rng, train, test):
    words = ["hello", "world", "python", "mini", "code", "exam"]
    for i in range(900):
        a_, b_, c_ = rng.randint(0, 50), rng.randint(1, 50), rng.randint(0, 50)
        snippet_i = i % len(PRINT_SNIPPETS)
        tpl, _ = PRINT_SNIPPETS[snippet_i]
        w = words[i % len(words)]
        code = tpl.format(a=a_, b=b_, c=c_, w=w)
        # честно исполняем код в изолированном namespace, чтобы получить правильный вывод
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(compile(code, "<snippet>", "exec"), {})
        answer = buf.getvalue().strip()
        fam_key = (snippet_i, a_, b_, c_, w)
        hide = is_hidden("what_prints", fam_key)
        for lang in LANGS:
            rec = {
                "q": PRINT_Q[lang].format(code=code),
                "a": answer,
                "kind": "what_prints",
                "lang": lang,
                "func_name": "",
                "tests": [],
                "hidden_tests": [],
            }
            (test if hide else train).append(rec)


# --------------------------------------------------------------------------------------
# "Исправь ошибку" — даём сломанный код + описание, ответ — исправленный чистый код
# --------------------------------------------------------------------------------------
BUG_FAMILIES = [
    {
        "fn": "double_all",
        "bug": "def double_all(numbers):\n    result = []\n    for number in numbers:\n        result = number * 2\n    return result\n",
        "desc": {"uz": "funksiya faqat oxirgi elementni ikki martalab qaytaradi, hammasini qaytarishi kerak",
                  "ru": "функция возвращает удвоенным только последний элемент, а должна все",
                  "en": "the function only doubles the last element, it should double all of them"},
        "fix": "def double_all(numbers: list[float]) -> list[float]:\n"
               "    \"\"\"Возвращает список чисел, умноженных на 2.\"\"\"\n"
               "    return [number * 2 for number in numbers]\n",
    },
    {
        "fn": "count_positive",
        "bug": "def count_positive(numbers):\n    count = 0\n    for number in numbers:\n        if number > 0:\n            count = 1\n    return count\n",
        "desc": {"uz": "count har doim 0 yoki 1 bo'lib qoladi, u ortib borishi kerak",
                  "ru": "count всегда 0 или 1, а должен увеличиваться на каждом положительном",
                  "en": "count stays 0 or 1, it should increment on every positive number"},
        "fix": "def count_positive(numbers: list[float]) -> int:\n"
               "    \"\"\"Считает количество положительных чисел.\"\"\"\n"
               "    count = 0\n"
               "    for number in numbers:\n"
               "        if number > 0:\n"
               "            count += 1\n"
               "    return count\n",
    },
    {
        "fn": "safe_divide",
        "bug": "def safe_divide(a, b):\n    return a / b\n",
        "desc": {"uz": "b nolga teng bo'lsa dastur xato beradi, 0 qaytarishi kerak",
                  "ru": "при b равном 0 программа падает с ошибкой, должна возвращать 0",
                  "en": "it crashes when b is 0, it should return 0 instead"},
        "fix": "def safe_divide(a: float, b: float) -> float:\n"
               "    \"\"\"Делит a на b, возвращает 0 при делении на ноль.\"\"\"\n"
               "    if b == 0:\n"
               "        return 0\n"
               "    return a / b\n",
    },
    {
        "fn": "first_word",
        "bug": "def first_word(text):\n    words = text.split()\n    return words[1]\n",
        "desc": {"uz": "funksiya ikkinchi so'zni qaytaradi, birinchisini qaytarishi kerak",
                  "ru": "функция возвращает второе слово, а должна первое",
                  "en": "the function returns the second word, it should return the first"},
        "fix": "def first_word(text: str) -> str:\n"
               "    \"\"\"Возвращает первое слово строки.\"\"\"\n"
               "    return text.split()[0]\n",
    },
]
BUG_Q = {
    "uz": "bu kodda xato bor: {desc}. Tuzat:\n{code}",
    "ru": "в этом коде ошибка: {desc}. исправь:\n{code}",
    "en": "this code has a bug: {desc}. fix it:\n{code}",
}


def build_fix_bug(rng, train, test):
    """Одна семья = одна связка (fn, bug, fix) — целиком уходит либо в train, либо в test."""
    for fam in BUG_FAMILIES:
        hide = is_hidden("fix_bug", fam["fn"])
        for lang in LANGS:
            rec = {
                "q": BUG_Q[lang].format(desc=fam["desc"][lang], code=fam["bug"]),
                "a": fam["fix"],
                "kind": "fix_bug",
                "lang": lang,
                "func_name": fam["fn"],
                "tests": [],
                "hidden_tests": [],
            }
            (test if hide else train).append(rec)


# --------------------------------------------------------------------------------------
# "Сделай код чище" — рабочий, но уродливый код → идиоматичный
# --------------------------------------------------------------------------------------
CLEANUP_FAMILIES = [
    {
        "fn": "check_even",
        "ugly": "def check_even(n):\n    if n % 2 == 0:\n        return True\n    else:\n        return False\n",
        "clean": "def check_even(n: int) -> bool:\n"
                 "    \"\"\"Проверяет, чётное ли число.\"\"\"\n"
                 "    return n % 2 == 0\n",
    },
    {
        "fn": "sum_items",
        "ugly": "def sum_items(lst):\n    s = 0\n    for i in range(len(lst)):\n        s = s + lst[i]\n    return s\n",
        "clean": "def sum_items(numbers: list[float]) -> float:\n"
                  "    \"\"\"Возвращает сумму списка чисел.\"\"\"\n"
                  "    total = 0\n"
                  "    for number in numbers:\n"
                  "        total += number\n"
                  "    return total\n",
    },
    {
        "fn": "apply_discount",
        "ugly": "def apply_discount(price):\n    return price - price * 0.15\n",
        "clean": "def apply_discount(price: float) -> float:\n"
                  "    \"\"\"Применяет скидку к цене.\"\"\"\n"
                  "    DISCOUNT_RATE = 0.15\n"
                  "    return price - price * DISCOUNT_RATE\n",
    },
    {
        "fn": "has_items",
        "ugly": "def has_items(lst):\n    if len(lst) > 0:\n        return True\n    return False\n",
        "clean": "def has_items(items: list) -> bool:\n"
                  "    \"\"\"Проверяет, не пуст ли список.\"\"\"\n"
                  "    return len(items) > 0\n",
    },
]
CLEANUP_Q = {
    "uz": "bu kod ishlaydi, lekin uni yozuvi chiroyli emas. Toza qilib qayta yoz:\n{code}",
    "ru": "этот код работает, но написан некрасиво. перепиши чище:\n{code}",
    "en": "this code works but is messy, rewrite it cleanly:\n{code}",
}


def build_cleanup(rng, train, test):
    """Одна семья = одна связка (fn, ugly, clean) — целиком уходит в train или test."""
    for fam in CLEANUP_FAMILIES:
        hide = is_hidden("cleanup", fam["fn"])
        for lang in LANGS:
            rec = {
                "q": CLEANUP_Q[lang].format(code=fam["ugly"]),
                "a": fam["clean"],
                "kind": "cleanup",
                "lang": lang,
                "func_name": fam["fn"],
                "tests": [],
                "hidden_tests": [],
                "ugly": fam["ugly"],       # для тестов codecheck: ugly должен проходить тесты, но quality < 80
            }
            (test if hide else train).append(rec)


def main():
    rng = random.Random(0)
    train, test = [], []

    basic_kinds = [k for k in PYTHON_KINDS if k in A]
    for kind in basic_kinds:
        build_basic(rng, kind, train, test)
    build_what_prints(rng, train, test)
    build_fix_bug(rng, train, test)
    build_cleanup(rng, train, test)

    out = ROOT / "data" / "python_tasks.json"
    out.write_text(json.dumps({"train": train, "test": test}, ensure_ascii=False), encoding="utf-8")
    kinds = sorted({x["kind"] for x in train + test})
    print(f"{out}: train {len(train)}, test {len(test)}")
    for k in kinds:
        print(f"  {k:20s} train {sum(x['kind'] == k for x in train):5d}  test {sum(x['kind'] == k for x in test):4d}")


if __name__ == "__main__":
    main()
