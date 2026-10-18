"""
Задачи по программированию на Python для Mini AI: модель пишет КОД (одну функцию),
а не просто ответ-число. Эталоны (`a`) всегда пишет скрипт/шаблон — никогда ИИ.

    .venv/bin/python tools/make_python_tasks.py   # → data/python_tasks.json

Формат как у data/school_math.json: {"train": [...], "test": [...]}, каждая запись —
    {q, a, kind, lang, func_name, tests: [{args, expected}], hidden_tests: [...]}
q — одна и та же задача на uz (латиница, обычный апостроф '), ru, en.
a — код функции (для kind="what_prints" — вывод кода; fix_bug — исправленный код).

Стиль эталонов описан в docs/PYTHON_STYLE.md, его же проверяет server/codecheck.py.

ЧЕСТНЫЙ ЭКЗАМЕН: каждая задача принадлежит "семье" — (kind, variant, func_name, параметры).
Семья целиком (во всех языках) прячется в test по crc32 от ключа семьи — модель никогда не
видит эти конкретные параметры на обучении. Если она решает их на экзамене — она поняла
СТРУКТУРУ задачи, а не запомнила код.

РЕАЛЬНАЯ РАЗНООБРАЗНОСТЬ: внутри каждого kind есть несколько variant — РАЗНЫХ задач
(sum / sum_squares / sum_positive / ... для kind=sum_list), не только разных имён функций.
Параметры (K, буква, слово) видны в тексте вопроса на всех 3 языках и зашиты в решении
как НАЗВАННАЯ константа/параметр (не магическое число).

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

import ast
import io
import json
import random
import zlib
from contextlib import redirect_stdout
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


def is_hidden(namespace: str, family_key) -> bool:
    """Детерминированно: эта семья параметров спрятана в экзамен? Одинаково для всех языков."""
    return zlib.crc32(f"{namespace}:{family_key}".encode()) % 100 < TEST_FRAC * 100


# --------------------------------------------------------------------------------------
# Пулы имён/параметров для параметризации
# --------------------------------------------------------------------------------------
FUNC_NAMES = {
    "sum_list": ["sum_list", "total_of", "add_all", "list_sum", "accumulate_values", "combine_numbers", "aggregate_numbers", "fold_sum"],
    "count_vowels": ["count_vowels", "vowel_count", "num_vowels", "letters_tally", "char_counter", "scan_chars", "tally_chars", "inspect_text"],
    "reverse_words": ["reverse_words", "flip_words", "words_reversed", "text_transform", "rearrange_text", "restyle_string", "reshape_text", "transform_string"],
    "is_palindrome": ["is_palindrome", "is_mirror", "reads_same_backward", "check_symmetry", "same_backward", "mirror_check", "palindrome_check", "symmetric_text"],
    "max_no_builtin": ["largest", "find_max", "biggest_value", "list_extreme", "scan_values", "pick_value", "top_value", "extreme_of"],
    "fizzbuzz": ["fizzbuzz_word", "label_number", "classify_number", "tag_number", "number_label", "mark_number"],
    "factorial": ["factorial", "fact_of", "number_fact", "compute_fact", "fact_value", "n_factorial", "fact_calc", "factorial_of"],
    "gcd": ["gcd_of", "greatest_common_divisor", "common_divisor", "divisor_of", "biggest_common_factor", "shared_divisor", "max_common_divisor"],
    "c2f": ["celsius_to_fahrenheit", "to_fahrenheit", "temp_convert", "convert_temp", "scale_temp", "temperature_shift", "degree_convert"],
    "average_guard": ["average", "mean_of", "safe_average", "central_value", "guarded_mean", "list_center", "mean_guard", "stat_center"],
    "filter_evens": ["filter_evens", "even_numbers", "only_evens", "pick_items", "sift_values", "select_items", "pick_matching", "filter_values"],
    "word_freq": ["word_frequency", "count_words", "tally_words", "words_tally", "word_counter", "count_each_word", "frequency_of_words", "word_stats"],
    "group_first_letter": ["group_by_first_letter", "group_by_letter", "bucket_words", "letter_buckets", "sort_into_groups", "classify_words", "arrange_by_letter"],
    "merge_dicts": ["merge_dicts", "combine_dicts", "join_maps", "unite_dicts", "blend_maps", "merge_maps", "combine_maps", "join_dicts"],
}

VOWEL_SETS = ["aeiou", "aeiouy", "aeiou01"[:5]]
LETTERS = list("abcdefgh")
K_POOL = [2, 3, 4, 5, 6, 7, 8, 10, 12, 15]
WORDS_POOL = ["cat", "dog", "bird", "lion", "fish", "wolf", "bear", "frog"]

FIZZ_WORDS = [("fizz", "buzz"), ("ping", "pong"), ("tap", "top"), ("left", "right")]
FIZZ_DIVS = [(3, 5), (2, 7), (3, 4), (2, 5)]


# --------------------------------------------------------------------------------------
# Вопросы (шаблоны) и решения (шаблоны) по variant внутри каждого kind
# --------------------------------------------------------------------------------------
def _q(uz, ru, en):
    return {"uz": uz, "ru": ru, "en": en}


VARIANTS = {
    # ---- SUM FAMILY -------------------------------------------------------------
    "sum_list": [
        {
            "id": "sum", "params": [{}],
            "q": _q("{fn}(lst) funksiyasini yoz: sonlar ro'yxatining yig'indisini qaytaradi",
                    "напиши функцию {fn}(lst): возвращает сумму списка чисел",
                    "write a function {fn}(lst) that returns the sum of a list of numbers"),
            "code": "def {fn}(numbers: list[float]) -> float:\n"
                    "    \"\"\"Возвращает сумму чисел списка.\"\"\"\n"
                    "    total = 0\n"
                    "    for number in numbers:\n"
                    "        total += number\n"
                    "    return total\n",
            "compute": lambda lst, **p: sum(lst),
        },
        {
            "id": "sum_squares", "params": [{}],
            "q": _q("{fn}(lst) funksiyasini yoz: sonlar kvadratlari yig'indisini qaytaradi",
                    "напиши функцию {fn}(lst): возвращает сумму квадратов чисел списка",
                    "write a function {fn}(lst) that returns the sum of the squares of a list of numbers"),
            "code": "def {fn}(numbers: list[float]) -> float:\n"
                    "    \"\"\"Возвращает сумму квадратов чисел списка.\"\"\"\n"
                    "    total = 0\n"
                    "    for number in numbers:\n"
                    "        total += number * number\n"
                    "    return total\n",
            "compute": lambda lst, **p: sum(x * x for x in lst),
        },
        {
            "id": "sum_positive", "params": [{}],
            "q": _q("{fn}(lst) funksiyasini yoz: faqat musbat sonlar yig'indisini qaytaradi",
                    "напиши функцию {fn}(lst): возвращает сумму только положительных чисел списка",
                    "write a function {fn}(lst) that returns the sum of only the positive numbers in a list"),
            "code": "def {fn}(numbers: list[float]) -> float:\n"
                    "    \"\"\"Возвращает сумму положительных чисел списка.\"\"\"\n"
                    "    total = 0\n"
                    "    for number in numbers:\n"
                    "        if number > 0:\n"
                    "            total += number\n"
                    "    return total\n",
            "compute": lambda lst, **p: sum(x for x in lst if x > 0),
        },
        {
            "id": "sum_odd", "params": [{}],
            "q": _q("{fn}(lst) funksiyasini yoz: toq sonlar yig'indisini qaytaradi",
                    "напиши функцию {fn}(lst): возвращает сумму нечётных чисел списка",
                    "write a function {fn}(lst) that returns the sum of the odd numbers in a list"),
            "code": "def {fn}(numbers: list[int]) -> int:\n"
                    "    \"\"\"Возвращает сумму нечётных чисел списка.\"\"\"\n"
                    "    total = 0\n"
                    "    for number in numbers:\n"
                    "        if number % 2 != 0:\n"
                    "            total += number\n"
                    "    return total\n",
            "compute": lambda lst, **p: sum(x for x in lst if x % 2 != 0),
        },
        {
            "id": "sum_gt_k", "params": [{"k": k} for k in K_POOL],
            "q": _q("{fn}(lst) funksiyasini yoz: {k} dan katta sonlar yig'indisini qaytaradi",
                    "напиши функцию {fn}(lst): возвращает сумму чисел списка, которые больше {k}",
                    "write a function {fn}(lst) that returns the sum of the numbers in a list greater than {k}"),
            "code": "def {fn}(numbers: list[float]) -> float:\n"
                    "    \"\"\"Возвращает сумму чисел списка больше THRESHOLD.\"\"\"\n"
                    "    THRESHOLD = {k}\n"
                    "    total = 0\n"
                    "    for number in numbers:\n"
                    "        if number > THRESHOLD:\n"
                    "            total += number\n"
                    "    return total\n",
            "compute": lambda lst, k, **p: sum(x for x in lst if x > k),
        },
        {
            "id": "product", "params": [{}],
            "q": _q("{fn}(lst) funksiyasini yoz: ro'yxatdagi sonlar ko'paytmasini qaytaradi",
                    "напиши функцию {fn}(lst): возвращает произведение чисел списка",
                    "write a function {fn}(lst) that returns the product of the numbers in a list"),
            "code": "def {fn}(numbers: list[float]) -> float:\n"
                    "    \"\"\"Возвращает произведение чисел списка.\"\"\"\n"
                    "    product = 1\n"
                    "    for number in numbers:\n"
                    "        product *= number\n"
                    "    return product\n",
            "compute": lambda lst, **p: _product(lst),
        },
    ],
    # ---- COUNT FAMILY -------------------------------------------------------------
    "count_vowels": [
        {
            "id": "count_vowels", "params": [{"vowels": v} for v in VOWEL_SETS],
            "q": _q("{fn}(s) funksiyasini yoz: satrdagi unlilar sonini qaytaradi ({vowels})",
                    "напиши функцию {fn}(s): возвращает количество гласных в строке ({vowels})",
                    "write a function {fn}(s) that returns the number of vowels in a string ({vowels})"),
            "code": "def {fn}(text: str) -> int:\n"
                    "    \"\"\"Считает гласные в строке.\"\"\"\n"
                    "    VOWELS = \"{vowels}\"\n"
                    "    return sum(1 for char in text.lower() if char in VOWELS)\n",
            "compute": lambda s, vowels, **p: sum(1 for c in s.lower() if c in vowels),
        },
        {
            "id": "count_letter", "params": [{"letter": letter} for letter in LETTERS],
            "q": _q("{fn}(s) funksiyasini yoz: satrda '{letter}' harfi necha marta uchraydi, shuni qaytaradi",
                    "напиши функцию {fn}(s): возвращает, сколько раз буква '{letter}' встречается в строке",
                    "write a function {fn}(s) that returns how many times the letter '{letter}' appears in a string"),
            "code": "def {fn}(text: str) -> int:\n"
                    "    \"\"\"Считает вхождения буквы LETTER в строке.\"\"\"\n"
                    "    LETTER = \"{letter}\"\n"
                    "    return sum(1 for char in text.lower() if char == LETTER)\n",
            "compute": lambda s, letter, **p: sum(1 for c in s.lower() if c == letter),
        },
        {
            "id": "count_digits", "params": [{}],
            "q": _q("{fn}(s) funksiyasini yoz: satrdagi raqamlar sonini qaytaradi",
                    "напиши функцию {fn}(s): возвращает количество цифр в строке",
                    "write a function {fn}(s) that returns the number of digits in a string"),
            "code": "def {fn}(text: str) -> int:\n"
                    "    \"\"\"Считает цифры в строке.\"\"\"\n"
                    "    return sum(1 for char in text if char.isdigit())\n",
            "compute": lambda s, **p: sum(1 for c in s if c.isdigit()),
        },
        {
            "id": "count_uppercase", "params": [{}],
            "q": _q("{fn}(s) funksiyasini yoz: satrdagi katta harflar sonini qaytaradi",
                    "напиши функцию {fn}(s): возвращает количество заглавных букв в строке",
                    "write a function {fn}(s) that returns the number of uppercase letters in a string"),
            "code": "def {fn}(text: str) -> int:\n"
                    "    \"\"\"Считает заглавные буквы в строке.\"\"\"\n"
                    "    return sum(1 for char in text if char.isupper())\n",
            "compute": lambda s, **p: sum(1 for c in s if c.isupper()),
        },
        {
            "id": "count_words_longer_than_k", "params": [{"k": k} for k in K_POOL[:6]],
            "q": _q("{fn}(s) funksiyasini yoz: uzunligi {k} dan katta so'zlar sonini qaytaradi",
                    "напиши функцию {fn}(s): возвращает количество слов строки длиннее {k} символов",
                    "write a function {fn}(s) that returns the number of words in a string longer than {k} characters"),
            "code": "def {fn}(text: str) -> int:\n"
                    "    \"\"\"Считает слова длиннее MIN_LEN символов.\"\"\"\n"
                    "    MIN_LEN = {k}\n"
                    "    return sum(1 for word in text.split() if len(word) > MIN_LEN)\n",
            "compute": lambda s, k, **p: sum(1 for w in s.split() if len(w) > k),
        },
    ],
    # ---- STRING FAMILY (reverse_words) ---------------------------------------------
    "reverse_words": [
        {
            "id": "reverse_words", "params": [{}],
            "q": _q("{fn}(s) funksiyasini yoz: so'zlar tartibini teskari qaytaradi",
                    "напиши функцию {fn}(s): возвращает слова строки в обратном порядке",
                    "write a function {fn}(s) that returns the words of a string in reverse order"),
            "code": "def {fn}(text: str) -> str:\n"
                    "    \"\"\"Возвращает слова строки в обратном порядке.\"\"\"\n"
                    "    return \" \".join(reversed(text.split()))\n",
            "compute": lambda s, **p: " ".join(reversed(s.split())),
        },
        {
            "id": "reverse_string", "params": [{}],
            "q": _q("{fn}(s) funksiyasini yoz: satrni teskari qaytaradi (harflar bo'yicha)",
                    "напиши функцию {fn}(s): возвращает строку, развёрнутую по символам",
                    "write a function {fn}(s) that returns the string reversed character by character"),
            "code": "def {fn}(text: str) -> str:\n"
                    "    \"\"\"Возвращает строку в обратном порядке символов.\"\"\"\n"
                    "    return text[::-1]\n",
            "compute": lambda s, **p: s[::-1],
        },
        {
            "id": "capitalize_words", "params": [{}],
            "q": _q("{fn}(s) funksiyasini yoz: har bir so'zni bosh harf bilan qaytaradi",
                    "напиши функцию {fn}(s): возвращает строку с каждым словом с большой буквы",
                    "write a function {fn}(s) that returns the string with every word capitalized"),
            "code": "def {fn}(text: str) -> str:\n"
                    "    \"\"\"Делает первую букву каждого слова заглавной.\"\"\"\n"
                    "    return \" \".join(word.capitalize() for word in text.split())\n",
            "compute": lambda s, **p: " ".join(w.capitalize() for w in s.split()),
        },
        {
            "id": "remove_spaces", "params": [{}],
            "q": _q("{fn}(s) funksiyasini yoz: satrdan barcha bo'sh joylarni olib tashlaydi",
                    "напиши функцию {fn}(s): возвращает строку без пробелов",
                    "write a function {fn}(s) that returns the string with all spaces removed"),
            "code": "def {fn}(text: str) -> str:\n"
                    "    \"\"\"Удаляет все пробелы из строки.\"\"\"\n"
                    "    return text.replace(\" \", \"\")\n",
            "compute": lambda s, **p: s.replace(" ", ""),
        },
        {
            "id": "repeat_k_times", "params": [{"k": k} for k in K_POOL[:6]],
            "q": _q("{fn}(s) funksiyasini yoz: satrni {k} marta takrorlab qaytaradi",
                    "напиши функцию {fn}(s): возвращает строку, повторённую {k} раз",
                    "write a function {fn}(s) that returns the string repeated {k} times"),
            "code": "def {fn}(text: str) -> str:\n"
                    "    \"\"\"Повторяет строку REPEAT_COUNT раз.\"\"\"\n"
                    "    REPEAT_COUNT = {k}\n"
                    "    return text * REPEAT_COUNT\n",
            "compute": lambda s, k, **p: s * k,
        },
        {
            "id": "first_k_chars", "params": [{"k": k} for k in K_POOL[:6]],
            "q": _q("{fn}(s) funksiyasini yoz: satrning birinchi {k} belgisini qaytaradi",
                    "напиши функцию {fn}(s): возвращает первые {k} символов строки",
                    "write a function {fn}(s) that returns the first {k} characters of the string"),
            "code": "def {fn}(text: str) -> str:\n"
                    "    \"\"\"Возвращает первые PREFIX_LEN символов строки.\"\"\"\n"
                    "    PREFIX_LEN = {k}\n"
                    "    return text[:PREFIX_LEN]\n",
            "compute": lambda s, k, **p: s[:k],
        },
    ],
    "is_palindrome": [
        {
            "id": "is_palindrome", "params": [{}],
            "q": _q("{fn}(s) funksiyasini yoz: satr (bo'shliqlarsiz) palindrom bo'lsa True qaytaradi",
                    "напиши функцию {fn}(s): возвращает True, если строка (без пробелов) палиндром",
                    "write a function {fn}(s) that returns True if the string, ignoring spaces, is a palindrome"),
            "code": "def {fn}(text: str) -> bool:\n"
                    "    \"\"\"Проверяет, является ли строка палиндромом без пробелов.\"\"\"\n"
                    "    cleaned = text.lower().replace(\" \", \"\")\n"
                    "    return cleaned == cleaned[::-1]\n",
            "compute": lambda s, **p: (s.lower().replace(" ", "")) == (s.lower().replace(" ", "")[::-1]),
        },
        {
            "id": "is_palindrome_case", "params": [{}],
            "q": _q("{fn}(s) funksiyasini yoz: satr katta-kichik harflarga qaramay palindrom bo'lsa True qaytaradi (bo'shliqlar hisobga olinadi)",
                    "напиши функцию {fn}(s): возвращает True, если строка палиндром без учёта регистра (пробелы учитываются)",
                    "write a function {fn}(s) that returns True if the string is a palindrome ignoring case (spaces are kept)"),
            "code": "def {fn}(text: str) -> bool:\n"
                    "    \"\"\"Проверяет палиндром без учёта регистра.\"\"\"\n"
                    "    cleaned = text.lower()\n"
                    "    return cleaned == cleaned[::-1]\n",
            "compute": lambda s, **p: s.lower() == s.lower()[::-1],
        },
        {
            "id": "is_palindrome_alnum_only", "params": [{}],
            "q": _q("{fn}(s) funksiyasini yoz: faqat harf va raqamlarni hisobga olib palindrom tekshiradi",
                    "напиши функцию {fn}(s): проверяет палиндром, учитывая только буквы и цифры",
                    "write a function {fn}(s) that checks for a palindrome considering only letters and digits"),
            "code": "def {fn}(text: str) -> bool:\n"
                    "    \"\"\"Проверяет палиндром, учитывая только буквы и цифры.\"\"\"\n"
                    "    cleaned = \"\".join(char.lower() for char in text if char.isalnum())\n"
                    "    return cleaned == cleaned[::-1]\n",
            "compute": lambda s, **p: _alnum_pal(s),
        },
        {
            "id": "starts_and_ends_same", "params": [{}],
            "q": _q("{fn}(s) funksiyasini yoz: satrning birinchi va oxirgi harfi bir xil bo'lsa True qaytaradi",
                    "напиши функцию {fn}(s): возвращает True, если первая и последняя буква строки совпадают",
                    "write a function {fn}(s) that returns True if the first and last letter of the string match"),
            "code": "def {fn}(text: str) -> bool:\n"
                    "    \"\"\"Проверяет, совпадают ли первая и последняя буквы строки.\"\"\"\n"
                    "    cleaned = text.lower()\n"
                    "    return cleaned[0] == cleaned[-1]\n",
            "compute": lambda s, **p: s.lower()[0] == s.lower()[-1],
        },
        {
            "id": "is_rotation_of_itself_reversed_k", "params": [{"k": k} for k in K_POOL[:5]],
            "q": _q("{fn}(s) funksiyasini yoz: satrning birinchi {k} harfi oxirgi {k} harfga teskari tartibda teng bo'lsa True qaytaradi",
                    "напиши функцию {fn}(s): возвращает True, если первые {k} букв строки равны последним {k} буквам в обратном порядке",
                    "write a function {fn}(s) that returns True if the first {k} letters of the string equal the last {k} letters reversed"),
            "code": "def {fn}(text: str) -> bool:\n"
                    "    \"\"\"Сравнивает первые LEN букв с последними LEN в обратном порядке.\"\"\"\n"
                    "    LEN = {k}\n"
                    "    return text[:LEN] == text[-LEN:][::-1]\n",
            "compute": lambda s, k, **p: s[:k] == s[-k:][::-1],
        },
    ],
    # ---- LIST FAMILY (max_no_builtin) ----------------------------------------------
    "max_no_builtin": [
        {
            "id": "max_no_builtin", "params": [{}],
            "q": _q("{fn}(lst) funksiyasini yoz: max() dan foydalanmasdan ro'yxatdagi eng katta sonni toping",
                    "напиши функцию {fn}(lst): найди наибольшее число в списке без max()",
                    "write a function {fn}(lst) that finds the largest number in a list without using max()"),
            "code": "def {fn}(numbers: list[float]) -> float:\n"
                    "    \"\"\"Находит наибольшее число без max().\"\"\"\n"
                    "    best = numbers[0]\n"
                    "    for number in numbers[1:]:\n"
                    "        if number > best:\n"
                    "            best = number\n"
                    "    return best\n",
            "compute": lambda lst, **p: max(lst),
        },
        {
            "id": "min_no_builtin", "params": [{}],
            "q": _q("{fn}(lst) funksiyasini yoz: min() dan foydalanmasdan ro'yxatdagi eng kichik sonni toping",
                    "напиши функцию {fn}(lst): найди наименьшее число в списке без min()",
                    "write a function {fn}(lst) that finds the smallest number in a list without using min()"),
            "code": "def {fn}(numbers: list[float]) -> float:\n"
                    "    \"\"\"Находит наименьшее число без min().\"\"\"\n"
                    "    best = numbers[0]\n"
                    "    for number in numbers[1:]:\n"
                    "        if number < best:\n"
                    "            best = number\n"
                    "    return best\n",
            "compute": lambda lst, **p: min(lst),
        },
        {
            "id": "second_largest", "params": [{}],
            "q": _q("{fn}(lst) funksiyasini yoz: ro'yxatdagi ikkinchi eng katta (noyob) sonni qaytaradi",
                    "напиши функцию {fn}(lst): возвращает второе по величине уникальное число списка",
                    "write a function {fn}(lst) that returns the second largest unique number in a list"),
            "code": "def {fn}(numbers: list[float]) -> float:\n"
                    "    \"\"\"Возвращает второе по величине уникальное число.\"\"\"\n"
                    "    unique_sorted = sorted(set(numbers), reverse=True)\n"
                    "    return unique_sorted[1]\n",
            "compute": lambda lst, **p: sorted(set(lst), reverse=True)[1],
        },
        {
            "id": "remove_duplicates_keep_order", "params": [{}],
            "q": _q("{fn}(lst) funksiyasini yoz: takrorlanmaydigan elementlar ro'yxatini tartibni saqlab qaytaradi",
                    "напиши функцию {fn}(lst): возвращает список без повторов, сохраняя порядок",
                    "write a function {fn}(lst) that returns the list without duplicates, preserving order"),
            "code": "def {fn}(items: list) -> list:\n"
                    "    \"\"\"Удаляет дубликаты, сохраняя порядок.\"\"\"\n"
                    "    seen: set = set()\n"
                    "    result = []\n"
                    "    for item in items:\n"
                    "        if item not in seen:\n"
                    "            seen.add(item)\n"
                    "            result.append(item)\n"
                    "    return result\n",
            "compute": lambda lst, **p: _dedupe_keep_order(lst),
        },
        {
            "id": "running_total", "params": [{}],
            "q": _q("{fn}(lst) funksiyasini yoz: ro'yxatning o'sib boruvchi (kumulyativ) yig'indisini qaytaradi",
                    "напиши функцию {fn}(lst): возвращает список накопительных сумм",
                    "write a function {fn}(lst) that returns the list of running (cumulative) totals"),
            "code": "def {fn}(numbers: list[float]) -> list[float]:\n"
                    "    \"\"\"Возвращает накопительные суммы списка.\"\"\"\n"
                    "    totals = []\n"
                    "    running = 0\n"
                    "    for number in numbers:\n"
                    "        running += number\n"
                    "        totals.append(running)\n"
                    "    return totals\n",
            "compute": lambda lst, **p: _running_total(lst),
        },
        {
            "id": "pairwise_diff", "params": [{}],
            "q": _q("{fn}(lst) funksiyasini yoz: qo'shni elementlar orasidagi farqlar ro'yxatini qaytaradi",
                    "напиши функцию {fn}(lst): возвращает список разностей соседних элементов",
                    "write a function {fn}(lst) that returns the list of differences between consecutive elements"),
            "code": "def {fn}(numbers: list[float]) -> list[float]:\n"
                    "    \"\"\"Возвращает разности соседних элементов списка.\"\"\"\n"
                    "    diffs = []\n"
                    "    for index in range(1, len(numbers)):\n"
                    "        diffs.append(numbers[index] - numbers[index - 1])\n"
                    "    return diffs\n",
            "compute": lambda lst, **p: [lst[i] - lst[i - 1] for i in range(1, len(lst))],
        },
        {
            "id": "count_gt_average", "params": [{}],
            "q": _q("{fn}(lst) funksiyasini yoz: o'rtachadan katta sonlar sonini qaytaradi",
                    "напиши функцию {fn}(lst): возвращает количество чисел больше среднего значения списка",
                    "write a function {fn}(lst) that returns the count of numbers greater than the list's average"),
            "code": "def {fn}(numbers: list[float]) -> int:\n"
                    "    \"\"\"Считает числа больше среднего значения списка.\"\"\"\n"
                    "    mean = sum(numbers) / len(numbers)\n"
                    "    return sum(1 for number in numbers if number > mean)\n",
            "compute": lambda lst, **p: sum(1 for x in lst if x > sum(lst) / len(lst)),
        },
        {
            "id": "index_of_max", "params": [{}],
            "q": _q("{fn}(lst) funksiyasini yoz: eng katta elementning indeksini qaytaradi",
                    "напиши функцию {fn}(lst): возвращает индекс наибольшего элемента списка",
                    "write a function {fn}(lst) that returns the index of the largest element"),
            "code": "def {fn}(numbers: list[float]) -> int:\n"
                    "    \"\"\"Возвращает индекс наибольшего элемента списка.\"\"\"\n"
                    "    best_index = 0\n"
                    "    for index, number in enumerate(numbers):\n"
                    "        if number > numbers[best_index]:\n"
                    "            best_index = index\n"
                    "    return best_index\n",
            "compute": lambda lst, **p: lst.index(max(lst)),
        },
    ],
    # ---- FIZZBUZZ (keep as a parametrized family) ----------------------------------
    "fizzbuzz": [
        {
            "id": "fizzbuzz", "params": [{"d1": d1, "d2": d2, "w1": w1, "w2": w2}
                                          for (d1, d2) in FIZZ_DIVS for (w1, w2) in FIZZ_WORDS],
            "q": _q("{fn}(n) funksiyasini yoz: {d1} ga bo'linsa '{w1}', {d2} ga bo'linsa '{w2}', ikkisiga ham bo'linsa '{w1}{w2}', bo'lmasa son qaytaradi",
                    "напиши функцию {fn}(n): если делится на {d1} — '{w1}', на {d2} — '{w2}', на оба — '{w1}{w2}', иначе само число",
                    "write a function {fn}(n) that returns '{w1}' if divisible by {d1}, '{w2}' if by {d2}, '{w1}{w2}' if both, else the number"),
            "code": "def {fn}(n: int) -> str:\n"
                    "    \"\"\"Возвращает метку по делимости на DIV_A и DIV_B.\"\"\"\n"
                    "    DIV_A, DIV_B = {d1}, {d2}\n"
                    "    label = \"\"\n"
                    "    if n % DIV_A == 0:\n"
                    "        label += \"{w1}\"\n"
                    "    if n % DIV_B == 0:\n"
                    "        label += \"{w2}\"\n"
                    "    return label or str(n)\n",
            "compute": lambda n, d1, d2, w1, w2, **p: (
                (w1 if n % d1 == 0 else "") + (w2 if n % d2 == 0 else "")
            ) or str(n),
        },
    ],
    # ---- NUMBER FAMILY: factorial / digit_sum --------------------------------------
    "factorial": [
        {
            "id": "factorial", "params": [{}],
            "q": _q("{fn}(n) funksiyasini yoz: n ning faktorialini qaytaradi",
                    "напиши функцию {fn}(n): возвращает факториал числа n",
                    "write a function {fn}(n) that returns the factorial of n"),
            "code": "def {fn}(n: int) -> int:\n"
                    "    \"\"\"Возвращает факториал n.\"\"\"\n"
                    "    result = 1\n"
                    "    for value in range(2, n + 1):\n"
                    "        result *= value\n"
                    "    return result\n",
            "compute": lambda n, **p: _factorial(n),
        },
        {
            "id": "digit_sum", "params": [{}],
            "q": _q("{fn}(n) funksiyasini yoz: n ning raqamlari yig'indisini qaytaradi",
                    "напиши функцию {fn}(n): возвращает сумму цифр числа n",
                    "write a function {fn}(n) that returns the sum of the digits of n"),
            "code": "def {fn}(n: int) -> int:\n"
                    "    \"\"\"Возвращает сумму цифр числа.\"\"\"\n"
                    "    total = 0\n"
                    "    for digit in str(abs(n)):\n"
                    "        total += int(digit)\n"
                    "    return total\n",
            "compute": lambda n, **p: sum(int(d) for d in str(abs(n))),
        },
        {
            "id": "is_prime", "params": [{}],
            "q": _q("{fn}(n) funksiyasini yoz: n tub son bo'lsa True qaytaradi",
                    "напиши функцию {fn}(n): возвращает True, если n простое число",
                    "write a function {fn}(n) that returns True if n is a prime number"),
            "code": "def {fn}(n: int) -> bool:\n"
                    "    \"\"\"Проверяет, является ли n простым числом.\"\"\"\n"
                    "    if n < 2:\n"
                    "        return False\n"
                    "    for divisor in range(2, int(n ** 0.5) + 1):\n"
                    "        if n % divisor == 0:\n"
                    "            return False\n"
                    "    return True\n",
            "compute": lambda n, **p: _is_prime(n),
        },
        {
            "id": "sign", "params": [{}],
            "q": _q("{fn}(n) funksiyasini yoz: musbat bo'lsa 1, manfiy bo'lsa -1, nolga teng bo'lsa 0 qaytaradi",
                    "напиши функцию {fn}(n): возвращает 1 если положительное, -1 если отрицательное, 0 если ноль",
                    "write a function {fn}(n) that returns 1 if positive, -1 if negative, 0 if zero"),
            "code": "def {fn}(n: float) -> int:\n"
                    "    \"\"\"Возвращает знак числа: 1, -1 или 0.\"\"\"\n"
                    "    if n > 0:\n"
                    "        return 1\n"
                    "    if n < 0:\n"
                    "        return -1\n"
                    "    return 0\n",
            "compute": lambda n, **p: (1 if n > 0 else (-1 if n < 0 else 0)),
        },
        {
            "id": "power_of_k", "params": [{"k": k} for k in K_POOL[:6]],
            "q": _q("{fn}(n) funksiyasini yoz: n ning {k}-darajasini qaytaradi",
                    "напиши функцию {fn}(n): возвращает n в степени {k}",
                    "write a function {fn}(n) that returns n raised to the power of {k}"),
            "code": "def {fn}(n: float) -> float:\n"
                    "    \"\"\"Возвращает n в степени POWER.\"\"\"\n"
                    "    POWER = {k}\n"
                    "    return n ** POWER\n",
            "compute": lambda n, k, **p: n ** k,
        },
        {
            "id": "is_divisible_by_k", "params": [{"k": k} for k in K_POOL],
            "q": _q("{fn}(n) funksiyasini yoz: n {k} ga bo'linsa True qaytaradi",
                    "напиши функцию {fn}(n): возвращает True, если n делится на {k}",
                    "write a function {fn}(n) that returns True if n is divisible by {k}"),
            "code": "def {fn}(n: int) -> bool:\n"
                    "    \"\"\"Проверяет делимость n на DIVISOR.\"\"\"\n"
                    "    DIVISOR = {k}\n"
                    "    return n % DIVISOR == 0\n",
            "compute": lambda n, k, **p: n % k == 0,
        },
    ],
    "gcd": [
        {
            "id": "gcd", "params": [{}],
            "q": _q("{fn}(a, b) funksiyasini yoz: a va b ning eng katta umumiy bo'luvchisini qaytaradi",
                    "напиши функцию {fn}(a, b): возвращает наибольший общий делитель a и b",
                    "write a function {fn}(a, b) that returns the greatest common divisor of a and b"),
            "code": "def {fn}(a: int, b: int) -> int:\n"
                    "    \"\"\"Возвращает наибольший общий делитель a и b.\"\"\"\n"
                    "    while b:\n"
                    "        a, b = b, a % b\n"
                    "    return a\n",
            "compute": lambda a, b, **p: _gcd(a, b),
        },
        {
            "id": "lcm", "params": [{}],
            "q": _q("{fn}(a, b) funksiyasini yoz: a va b ning eng kichik umumiy bo'linuvchisini qaytaradi",
                    "напиши функцию {fn}(a, b): возвращает наименьшее общее кратное a и b",
                    "write a function {fn}(a, b) that returns the least common multiple of a and b"),
            "code": "def {fn}(a: int, b: int) -> int:\n"
                    "    \"\"\"Возвращает наименьшее общее кратное a и b.\"\"\"\n"
                    "    x, y = a, b\n"
                    "    while y:\n"
                    "        x, y = y, x % y\n"
                    "    return a * b // x\n",
            "compute": lambda a, b, **p: a * b // _gcd(a, b),
        },
        {
            "id": "clamp", "params": [{"lo": lo, "hi": lo + rng_offset} for lo, rng_offset in [(0, 10), (-10, 20), (5, 15), (-50, 50), (1, 9)]],
            "q": _q("{fn}(x, lo, hi) funksiyasini yoz: x qiymatini [lo, hi] oraliqqa cheklaydi (clamp)",
                    "напиши функцию {fn}(x, lo, hi): ограничивает x диапазоном [lo, hi] (clamp)",
                    "write a function {fn}(x, lo, hi) that clamps x into the range [lo, hi]"),
            "code": "def {fn}(value: float, low: float, high: float) -> float:\n"
                    "    \"\"\"Ограничивает значение диапазоном [low, high].\"\"\"\n"
                    "    if value < low:\n"
                    "        return low\n"
                    "    if value > high:\n"
                    "        return high\n"
                    "    return value\n",
            "compute": lambda x, lo, hi, **p: min(max(x, lo), hi),
        },
        {
            "id": "is_coprime", "params": [{}],
            "q": _q("{fn}(a, b) funksiyasini yoz: a va b o'zaro tub (gcd=1) bo'lsa True qaytaradi",
                    "напиши функцию {fn}(a, b): возвращает True, если a и b взаимно простые (gcd=1)",
                    "write a function {fn}(a, b) that returns True if a and b are coprime (gcd is 1)"),
            "code": "def {fn}(a: int, b: int) -> bool:\n"
                    "    \"\"\"Проверяет, взаимно просты ли a и b.\"\"\"\n"
                    "    x, y = a, b\n"
                    "    while y:\n"
                    "        x, y = y, x % y\n"
                    "    return x == 1\n",
            "compute": lambda a, b, **p: _gcd(a, b) == 1,
        },
        {
            "id": "gcd_plus_k", "params": [{"k": k} for k in K_POOL[:6]],
            "q": _q("{fn}(a, b) funksiyasini yoz: a va b ning EKUBiga {k} qo'shib qaytaradi",
                    "напиши функцию {fn}(a, b): возвращает НОД a и b, увеличенный на {k}",
                    "write a function {fn}(a, b) that returns the gcd of a and b, plus {k}"),
            "code": "def {fn}(a: int, b: int) -> int:\n"
                    "    \"\"\"Возвращает НОД a и b, увеличенный на EXTRA.\"\"\"\n"
                    "    EXTRA = {k}\n"
                    "    x, y = a, b\n"
                    "    while y:\n"
                    "        x, y = y, x % y\n"
                    "    return x + EXTRA\n",
            "compute": lambda a, b, k, **p: _gcd(a, b) + k,
        },
    ],
    "c2f": [
        {
            "id": "c2f", "params": [{}],
            "q": _q("{fn}(c) funksiyasini yoz: Selsiyni Farengeytga aylantiradi",
                    "напиши функцию {fn}(c): переводит градусы Цельсия в Фаренгейт",
                    "write a function {fn}(c) that converts Celsius to Fahrenheit"),
            "code": "def {fn}(celsius: float) -> float:\n"
                    "    \"\"\"Переводит градусы Цельсия в Фаренгейт.\"\"\"\n"
                    "    SCALE, OFFSET = 9 / 5, 32\n"
                    "    return celsius * SCALE + OFFSET\n",
            "compute": lambda c, **p: c * (9 / 5) + 32,
        },
        {
            "id": "f2c", "params": [{}],
            "q": _q("{fn}(f) funksiyasini yoz: Farengeytni Selsiyga aylantiradi",
                    "напиши функцию {fn}(f): переводит градусы Фаренгейта в Цельсий",
                    "write a function {fn}(f) that converts Fahrenheit to Celsius"),
            "code": "def {fn}(fahrenheit: float) -> float:\n"
                    "    \"\"\"Переводит градусы Фаренгейта в Цельсий.\"\"\"\n"
                    "    SCALE, OFFSET = 5 / 9, 32\n"
                    "    return (fahrenheit - OFFSET) * SCALE\n",
            "compute": lambda f, **p: (f - 32) * (5 / 9),
        },
        {
            "id": "km_to_miles", "params": [{}],
            "q": _q("{fn}(km) funksiyasini yoz: kilometrni milyaga aylantiradi",
                    "напиши функцию {fn}(km): переводит километры в мили",
                    "write a function {fn}(km) that converts kilometers to miles"),
            "code": "def {fn}(km: float) -> float:\n"
                    "    \"\"\"Переводит километры в мили.\"\"\"\n"
                    "    KM_TO_MILE = 0.621371\n"
                    "    return km * KM_TO_MILE\n",
            "compute": lambda km, **p: km * 0.621371,
        },
        {
            "id": "add_k_degrees", "params": [{"k": k} for k in K_POOL[:6]],
            "q": _q("{fn}(c) funksiyasini yoz: haroratga {k} daraja qo'shib qaytaradi",
                    "напиши функцию {fn}(c): возвращает температуру, увеличенную на {k} градусов",
                    "write a function {fn}(c) that returns the temperature increased by {k} degrees"),
            "code": "def {fn}(temperature: float) -> float:\n"
                    "    \"\"\"Увеличивает температуру на DELTA градусов.\"\"\"\n"
                    "    DELTA = {k}\n"
                    "    return temperature + DELTA\n",
            "compute": lambda c, k, **p: c + k,
        },
    ],
    # ---- average_guard: average / median with empty-list guard --------------------
    "average_guard": [
        {
            "id": "average_guard", "params": [{}],
            "q": _q("{fn}(lst) funksiyasini yoz: ro'yxat o'rtachasini qaytaradi, bo'sh bo'lsa 0 qaytaradi",
                    "напиши функцию {fn}(lst): возвращает среднее значение списка, а для пустого списка 0",
                    "write a function {fn}(lst) that returns the average of a list, or 0 for an empty list"),
            "code": "def {fn}(numbers: list[float]) -> float:\n"
                    "    \"\"\"Возвращает среднее списка, 0 для пустого списка.\"\"\"\n"
                    "    if not numbers:\n"
                    "        return 0\n"
                    "    return sum(numbers) / len(numbers)\n",
            "compute": lambda lst, **p: (sum(lst) / len(lst)) if lst else 0,
        },
        {
            "id": "median_guard", "params": [{}],
            "q": _q("{fn}(lst) funksiyasini yoz: ro'yxat medianasini qaytaradi, bo'sh bo'lsa 0 qaytaradi",
                    "напиши функцию {fn}(lst): возвращает медиану списка, а для пустого списка 0",
                    "write a function {fn}(lst) that returns the median of a list, or 0 for an empty list"),
            "code": "def {fn}(numbers: list[float]) -> float:\n"
                    "    \"\"\"Возвращает медиану списка, 0 для пустого списка.\"\"\"\n"
                    "    if not numbers:\n"
                    "        return 0\n"
                    "    ordered = sorted(numbers)\n"
                    "    middle = len(ordered) // 2\n"
                    "    if len(ordered) % 2 == 0:\n"
                    "        return (ordered[middle - 1] + ordered[middle]) / 2\n"
                    "    return ordered[middle]\n",
            "compute": lambda lst, **p: _median(lst),
        },
        {
            "id": "range_guard", "params": [{}],
            "q": _q("{fn}(lst) funksiyasini yoz: ro'yxat diapazonini (max-min) qaytaradi, bo'sh bo'lsa 0 qaytaradi",
                    "напиши функцию {fn}(lst): возвращает разброс списка (max-min), а для пустого списка 0",
                    "write a function {fn}(lst) that returns the range (max-min) of a list, or 0 for an empty list"),
            "code": "def {fn}(numbers: list[float]) -> float:\n"
                    "    \"\"\"Возвращает разброс (max - min) списка, 0 для пустого списка.\"\"\"\n"
                    "    if not numbers:\n"
                    "        return 0\n"
                    "    return max(numbers) - min(numbers)\n",
            "compute": lambda lst, **p: (max(lst) - min(lst)) if lst else 0,
        },
        {
            "id": "count_above_k_guard", "params": [{"k": k} for k in K_POOL[:6]],
            "q": _q("{fn}(lst) funksiyasini yoz: {k} dan katta sonlar sonini qaytaradi, bo'sh bo'lsa 0 qaytaradi",
                    "напиши функцию {fn}(lst): возвращает количество чисел больше {k}, а для пустого списка 0",
                    "write a function {fn}(lst) that returns the count of numbers greater than {k}, or 0 for an empty list"),
            "code": "def {fn}(numbers: list[float]) -> int:\n"
                    "    \"\"\"Считает числа больше THRESHOLD, 0 для пустого списка.\"\"\"\n"
                    "    THRESHOLD = {k}\n"
                    "    if not numbers:\n"
                    "        return 0\n"
                    "    return sum(1 for number in numbers if number > THRESHOLD)\n",
            "compute": lambda lst, k, **p: (sum(1 for x in lst if x > k) if lst else 0),
        },
    ],
    # ---- FILTER FAMILY ---------------------------------------------------------------
    "filter_evens": [
        {
            "id": "filter_evens", "params": [{}],
            "q": _q("{fn}(lst) funksiyasini yoz: ro'yxatdagi juft sonlarni qaytaradi",
                    "напиши функцию {fn}(lst): возвращает только чётные числа из списка",
                    "write a function {fn}(lst) that returns only the even numbers from a list"),
            "code": "def {fn}(numbers: list[int]) -> list[int]:\n"
                    "    \"\"\"Возвращает только чётные числа списка.\"\"\"\n"
                    "    return [number for number in numbers if number % 2 == 0]\n",
            "compute": lambda lst, **p: [x for x in lst if x % 2 == 0],
        },
        {
            "id": "filter_odds", "params": [{}],
            "q": _q("{fn}(lst) funksiyasini yoz: ro'yxatdagi toq sonlarni qaytaradi",
                    "напиши функцию {fn}(lst): возвращает только нечётные числа из списка",
                    "write a function {fn}(lst) that returns only the odd numbers from a list"),
            "code": "def {fn}(numbers: list[int]) -> list[int]:\n"
                    "    \"\"\"Возвращает только нечётные числа списка.\"\"\"\n"
                    "    return [number for number in numbers if number % 2 != 0]\n",
            "compute": lambda lst, **p: [x for x in lst if x % 2 != 0],
        },
        {
            "id": "filter_gt_k", "params": [{"k": k} for k in K_POOL],
            "q": _q("{fn}(lst) funksiyasini yoz: {k} dan katta sonlarni qaytaradi",
                    "напиши функцию {fn}(lst): возвращает числа списка, которые больше {k}",
                    "write a function {fn}(lst) that returns the numbers in a list greater than {k}"),
            "code": "def {fn}(numbers: list[float]) -> list[float]:\n"
                    "    \"\"\"Возвращает числа списка больше THRESHOLD.\"\"\"\n"
                    "    THRESHOLD = {k}\n"
                    "    return [number for number in numbers if number > THRESHOLD]\n",
            "compute": lambda lst, k, **p: [x for x in lst if x > k],
        },
        {
            "id": "filter_divisible_by_k", "params": [{"k": k} for k in K_POOL[1:]],
            "q": _q("{fn}(lst) funksiyasini yoz: {k} ga bo'linadigan sonlarni qaytaradi",
                    "напиши функцию {fn}(lst): возвращает числа списка, делящиеся на {k}",
                    "write a function {fn}(lst) that returns the numbers in a list divisible by {k}"),
            "code": "def {fn}(numbers: list[int]) -> list[int]:\n"
                    "    \"\"\"Возвращает числа списка, делящиеся на DIVISOR.\"\"\"\n"
                    "    DIVISOR = {k}\n"
                    "    return [number for number in numbers if number % DIVISOR == 0]\n",
            "compute": lambda lst, k, **p: [x for x in lst if x % k == 0],
        },
        {
            "id": "filter_nonempty_strings", "params": [{}],
            "q": _q("{fn}(lst) funksiyasini yoz: ro'yxatdagi bo'sh bo'lmagan satrlarni qaytaradi",
                    "напиши функцию {fn}(lst): возвращает непустые строки из списка",
                    "write a function {fn}(lst) that returns only the non-empty strings from a list"),
            "code": "def {fn}(items: list[str]) -> list[str]:\n"
                    "    \"\"\"Возвращает непустые строки списка.\"\"\"\n"
                    "    return [item for item in items if item]\n",
            "compute": lambda lst, **p: [x for x in lst if x],
        },
        {
            "id": "filter_starts_with_letter", "params": [{"letter": letter} for letter in LETTERS],
            "q": _q("{fn}(lst) funksiyasini yoz: '{letter}' harfidan boshlanadigan so'zlarni qaytaradi",
                    "напиши функцию {fn}(lst): возвращает слова, начинающиеся на букву '{letter}'",
                    "write a function {fn}(lst) that returns the words starting with the letter '{letter}'"),
            "code": "def {fn}(words: list[str]) -> list[str]:\n"
                    "    \"\"\"Возвращает слова, начинающиеся на LETTER.\"\"\"\n"
                    "    LETTER = \"{letter}\"\n"
                    "    return [word for word in words if word.lower().startswith(LETTER)]\n",
            "compute": lambda lst, letter, **p: [x for x in lst if x.lower().startswith(letter)],
        },
    ],
    # ---- DICT FAMILY: word_freq / group_first_letter / merge_dicts -----------------
    "word_freq": [
        {
            "id": "word_freq", "params": [{}],
            "q": _q("{fn}(s) funksiyasini yoz: satrdagi har bir so'z necha marta takrorlanganini lug'at qilib qaytaradi",
                    "напиши функцию {fn}(s): возвращает словарь — сколько раз встречается каждое слово в строке",
                    "write a function {fn}(s) that returns a dict with the count of each word in a string"),
            "code": "def {fn}(text: str) -> dict[str, int]:\n"
                    "    \"\"\"Считает, сколько раз встречается каждое слово.\"\"\"\n"
                    "    counts: dict[str, int] = {{}}\n"
                    "    for word in text.lower().split():\n"
                    "        counts[word] = counts.get(word, 0) + 1\n"
                    "    return counts\n",
            "compute": lambda s, **p: _word_freq(s),
        },
        {
            "id": "count_given_word", "params": [{"word": w} for w in WORDS_POOL],
            "q": _q("{fn}(s) funksiyasini yoz: satrda '{word}' so'zi necha marta uchraydi, shuni qaytaradi",
                    "напиши функцию {fn}(s): возвращает, сколько раз слово '{word}' встречается в строке",
                    "write a function {fn}(s) that returns how many times the word '{word}' appears in a string"),
            "code": "def {fn}(text: str) -> int:\n"
                    "    \"\"\"Считает вхождения слова WORD в строке.\"\"\"\n"
                    "    WORD = \"{word}\"\n"
                    "    return sum(1 for word in text.lower().split() if word == WORD)\n",
            "compute": lambda s, word, **p: sum(1 for w in s.lower().split() if w == word),
        },
        {
            "id": "unique_word_count", "params": [{}],
            "q": _q("{fn}(s) funksiyasini yoz: satrdagi noyob (takrorlanmaydigan) so'zlar sonini qaytaradi",
                    "напиши функцию {fn}(s): возвращает количество уникальных слов в строке",
                    "write a function {fn}(s) that returns the number of unique words in a string"),
            "code": "def {fn}(text: str) -> int:\n"
                    "    \"\"\"Считает количество уникальных слов в строке.\"\"\"\n"
                    "    return len(set(text.lower().split()))\n",
            "compute": lambda s, **p: len(set(s.lower().split())),
        },
        {
            "id": "most_common_word", "params": [{}],
            "q": _q("{fn}(s) funksiyasini yoz: satrda eng ko'p takrorlangan so'zni qaytaradi",
                    "напиши функцию {fn}(s): возвращает слово, которое встречается в строке чаще всех",
                    "write a function {fn}(s) that returns the word that appears most often in a string"),
            "code": "def {fn}(text: str) -> str:\n"
                    "    \"\"\"Возвращает самое часто встречающееся слово строки.\"\"\"\n"
                    "    counts: dict[str, int] = {{}}\n"
                    "    for word in text.lower().split():\n"
                    "        counts[word] = counts.get(word, 0) + 1\n"
                    "    return max(counts, key=counts.get)\n",
            "compute": lambda s, **p: _most_common_word(s),
        },
    ],
    "group_first_letter": [
        {
            "id": "group_first_letter", "params": [{}],
            "q": _q("{fn}(words) funksiyasini yoz: so'zlarni birinchi harfi bo'yicha lug'atga guruhlaydi",
                    "напиши функцию {fn}(words): группирует слова в словарь по первой букве",
                    "write a function {fn}(words) that groups words into a dict keyed by first letter"),
            "code": "def {fn}(words: list[str]) -> dict[str, list[str]]:\n"
                    "    \"\"\"Группирует слова в словарь по первой букве.\"\"\"\n"
                    "    groups: dict[str, list[str]] = {{}}\n"
                    "    for word in words:\n"
                    "        groups.setdefault(word[0], []).append(word)\n"
                    "    return groups\n",
            "compute": lambda words, **p: _group_by(words, lambda w: w[0]),
        },
        {
            "id": "group_by_length", "params": [{}],
            "q": _q("{fn}(words) funksiyasini yoz: so'zlarni uzunligi bo'yicha lug'atga guruhlaydi",
                    "напиши функцию {fn}(words): группирует слова в словарь по длине",
                    "write a function {fn}(words) that groups words into a dict keyed by length"),
            "code": "def {fn}(words: list[str]) -> dict[int, list[str]]:\n"
                    "    \"\"\"Группирует слова в словарь по длине.\"\"\"\n"
                    "    groups: dict[int, list[str]] = {{}}\n"
                    "    for word in words:\n"
                    "        groups.setdefault(len(word), []).append(word)\n"
                    "    return groups\n",
            "compute": lambda words, **p: _group_by(words, len),
        },
        {
            "id": "group_by_contains_letter", "params": [{"letter": letter} for letter in LETTERS[:6]],
            "q": _q("{fn}(words) funksiyasini yoz: so'zlarni tarkibida '{letter}' harfi bor-yo'qligi bo'yicha guruhlaydi",
                    "напиши функцию {fn}(words): группирует слова по наличию буквы '{letter}' в словаре с ключами True/False",
                    "write a function {fn}(words) that groups words by whether they contain the letter '{letter}' into a dict keyed True/False"),
            "code": "def {fn}(words: list[str]) -> dict[bool, list[str]]:\n"
                    "    \"\"\"Группирует слова по наличию буквы LETTER.\"\"\"\n"
                    "    LETTER = \"{letter}\"\n"
                    "    groups: dict[bool, list[str]] = {{}}\n"
                    "    for word in words:\n"
                    "        groups.setdefault(LETTER in word.lower(), []).append(word)\n"
                    "    return groups\n",
            "compute": lambda words, letter, **p: _group_by(words, lambda w: letter in w.lower()),
        },
    ],
    "merge_dicts": [
        {
            "id": "merge_dicts", "params": [{}],
            "q": _q("{fn}(d1, d2) funksiyasini yoz: ikki lug'atni birlashtiradi (d2 ustunlik qiladi)",
                    "напиши функцию {fn}(d1, d2): объединяет два словаря (d2 имеет приоритет)",
                    "write a function {fn}(d1, d2) that merges two dicts (d2 values take priority)"),
            "code": "def {fn}(first: dict, second: dict) -> dict:\n"
                    "    \"\"\"Объединяет два словаря, второй имеет приоритет.\"\"\"\n"
                    "    merged = dict(first)\n"
                    "    merged.update(second)\n"
                    "    return merged\n",
            "compute": lambda d1, d2, **p: dict(d1, **d2),
        },
        {
            "id": "merge_sum_values", "params": [{}],
            "q": _q("{fn}(d1, d2) funksiyasini yoz: ikki lug'atni birlashtiradi, bir xil kalitlar qiymatlari qo'shiladi",
                    "напиши функцию {fn}(d1, d2): объединяет два словаря, складывая значения общих ключей",
                    "write a function {fn}(d1, d2) that merges two dicts, summing values for shared keys"),
            "code": "def {fn}(first: dict, second: dict) -> dict:\n"
                    "    \"\"\"Объединяет словари, суммируя значения общих ключей.\"\"\"\n"
                    "    merged = dict(first)\n"
                    "    for key, value in second.items():\n"
                    "        merged[key] = merged.get(key, 0) + value\n"
                    "    return merged\n",
            "compute": lambda d1, d2, **p: _merge_sum(d1, d2),
        },
        {
            "id": "invert_dict", "params": [{}],
            "q": _q("{fn}(d) funksiyasini yoz: lug'atdagi kalit va qiymatlarni joylarini almashtiradi",
                    "напиши функцию {fn}(d): возвращает словарь с обменянными ключами и значениями",
                    "write a function {fn}(d) that returns the dict with keys and values swapped"),
            "code": "def {fn}(mapping: dict) -> dict:\n"
                    "    \"\"\"Меняет местами ключи и значения словаря.\"\"\"\n"
                    "    return {{value: key for key, value in mapping.items()}}\n",
            "compute": lambda d, **p: {v: k for k, v in d.items()},
        },
        {
            "id": "keys_with_max_value", "params": [{}],
            "q": _q("{fn}(d) funksiyasini yoz: eng katta qiymatga ega bo'lgan kalitlar ro'yxatini qaytaradi",
                    "напиши функцию {fn}(d): возвращает список ключей с наибольшим значением",
                    "write a function {fn}(d) that returns the list of keys that have the maximum value"),
            "code": "def {fn}(mapping: dict) -> list:\n"
                    "    \"\"\"Возвращает ключи с наибольшим значением.\"\"\"\n"
                    "    best = max(mapping.values())\n"
                    "    return [key for key, value in mapping.items() if value == best]\n",
            "compute": lambda d, **p: [k for k, v in d.items() if v == max(d.values())],
        },
        {
            "id": "keys_in_both", "params": [{}],
            "q": _q("{fn}(d1, d2) funksiyasini yoz: ikkala lug'atda ham mavjud bo'lgan kalitlar ro'yxatini qaytaradi",
                    "напиши функцию {fn}(d1, d2): возвращает список ключей, присутствующих в обоих словарях",
                    "write a function {fn}(d1, d2) that returns the list of keys present in both dicts"),
            "code": "def {fn}(first: dict, second: dict) -> list:\n"
                    "    \"\"\"Возвращает ключи, присутствующие в обоих словарях.\"\"\"\n"
                    "    return [key for key in first if key in second]\n",
            "compute": lambda d1, d2, **p: [k for k in d1 if k in d2],
        },
        {
            "id": "add_k_to_values", "params": [{"k": k} for k in K_POOL[:6]],
            "q": _q("{fn}(d) funksiyasini yoz: lug'atdagi har bir qiymatga {k} qo'shib yangi lug'at qaytaradi",
                    "напиши функцию {fn}(d): возвращает новый словарь, где к каждому значению добавлено {k}",
                    "write a function {fn}(d) that returns a new dict where {k} is added to every value"),
            "code": "def {fn}(mapping: dict) -> dict:\n"
                    "    \"\"\"Добавляет DELTA к каждому значению словаря.\"\"\"\n"
                    "    DELTA = {k}\n"
                    "    return {{key: value + DELTA for key, value in mapping.items()}}\n",
            "compute": lambda d, k, **p: {kk: vv + k for kk, vv in d.items()},
        },
    ],
}


# --------------------------------------------------------------------------------------
# Вспомогательные чистые функции (для подсчёта ожидаемых значений)
# --------------------------------------------------------------------------------------
def _product(lst):
    result = 1
    for x in lst:
        result *= x
    return result


def _dedupe_keep_order(lst):
    seen, out = set(), []
    for x in lst:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _running_total(lst):
    out, running = [], 0
    for x in lst:
        running += x
        out.append(running)
    return out


def _factorial(n):
    r = 1
    for v in range(2, n + 1):
        r *= v
    return r


def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _median(lst):
    if not lst:
        return 0
    ordered = sorted(lst)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[mid - 1] + ordered[mid]) / 2
    return ordered[mid]


def _word_freq(s):
    counts = {}
    for w in s.lower().split():
        counts[w] = counts.get(w, 0) + 1
    return counts


def _group_by(words, keyfn):
    groups = {}
    for w in words:
        groups.setdefault(keyfn(w), []).append(w)
    return groups


def _merge_sum(d1, d2):
    merged = dict(d1)
    for k, v in d2.items():
        merged[k] = merged.get(k, 0) + v
    return merged


def _is_prime(n):
    if n < 2:
        return False
    for d in range(2, int(n ** 0.5) + 1):
        if n % d == 0:
            return False
    return True


def _alnum_pal(s):
    cleaned = "".join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]


def _most_common_word(s):
    counts = {}
    for w in s.lower().split():
        counts[w] = counts.get(w, 0) + 1
    return max(counts, key=counts.get)


# --------------------------------------------------------------------------------------
# Генерация тестовых входов по kind (случайные значения — НЕ влияют на (q, a))
# --------------------------------------------------------------------------------------
def _gen_input(kind, variant_id, rng):
    if kind == "sum_list":
        lst = [rng.randint(-50, 50) for _ in range(rng.randint(1, 6))]
        return (lst,)
    if kind == "count_vowels":
        words = ["banana", "stereo", "umbrella", "yogurt", "picture", "orange"]
        s = rng.choice(words) + str(rng.randint(0, 999))
        return (s,)
    if kind in ("reverse_words",):
        s = f"word{rng.randint(0,99)} another{rng.randint(0,99)} last{rng.randint(0,99)}"
        return (s,)
    if kind == "is_palindrome":
        base = rng.choice(["level", "radar", "hello", "kayak", "python", "noon"])
        s = base + (base[::-1] if rng.random() < 0.5 else str(rng.randint(0, 99)))
        return (s,)
    if kind == "max_no_builtin":
        lst = [rng.randint(-100, 100) for _ in range(rng.randint(3, 7))]
        # гарантируем >=2 уникальных значения для second_largest
        if variant_id == "second_largest" and len(set(lst)) < 2:
            lst[0] = lst[1] + 1
        return (lst,)
    if kind == "fizzbuzz":
        return (rng.randint(1, 100),)
    if kind == "factorial":
        return (rng.randint(0, 10),)
    if kind == "gcd":
        if variant_id == "clamp":
            return (rng.randint(-60, 60),)  # второй/третий аргумент добавляются в make_tests
        return (rng.randint(1, 200), rng.randint(1, 200))
    if kind == "c2f":
        if variant_id == "km_to_miles":
            return (rng.randint(0, 500),)
        return (rng.randint(-40, 100),)
    if kind == "average_guard":
        lst = [] if rng.random() < 0.1 else [rng.randint(-50, 50) for _ in range(rng.randint(1, 6))]
        return (lst,)
    if kind == "filter_evens":
        if variant_id == "filter_nonempty_strings":
            words = ["cat", "", "dog", "", "bird"]
            return (rng.sample(words, k=rng.randint(2, len(words))),)
        if variant_id == "filter_starts_with_letter":
            pool = ["apple", "ant", "bear", "cat", "car", "dog", "deer"]
            return (rng.sample(pool, k=rng.randint(2, len(pool))),)
        lst = [rng.randint(-20, 20) for _ in range(rng.randint(1, 8))]
        return (lst,)
    if kind == "word_freq":
        s = " ".join(rng.choice(["cat", "dog", "cat", "bird", "dog", "dog"]) for _ in range(rng.randint(3, 7)))
        s += str(rng.randint(0, 999))
        return (s,)
    if kind == "group_first_letter":
        pool = ["apple", "ant", "bear", "cat", "car", "dog", "deer", "ape"]
        words = rng.sample(pool, k=rng.randint(2, len(pool)))
        return (words,)
    if kind == "merge_dicts":
        d1_v = {f"k{j}": rng.randint(0, 9) for j in range(rng.randint(1, 4))}
        if variant_id in ("invert_dict", "keys_with_max_value", "add_k_to_values"):
            # гарантируем уникальные значения, чтобы invert_dict был корректной биекцией
            d1_v = {f"k{j}": j * 10 + rng.randint(0, 9) for j in range(rng.randint(1, 4))}
            return (d1_v,)
        d2_v = {f"k{j}": rng.randint(0, 9) for j in range(rng.randint(0, 3), rng.randint(0, 3) + rng.randint(1, 3))}
        return (d1_v, d2_v)
    raise ValueError(kind)


def make_tests(kind, variant, param, rng, n=3):
    tests = []
    for _ in range(n):
        args = _gen_input(kind, variant["id"], rng)
        if variant["id"] == "clamp":
            # lo/hi — настоящие аргументы функции, а не константы в коде
            args = (args[0], param["lo"], param["hi"])
            expected = variant["compute"](*args)
        else:
            expected = variant["compute"](*args, **param)
        tests.append({"args": list(args), "expected": expected})
    return tests


def build_basic(rng, kind, train, test, seen=None):
    """Строит все variant x func_name x param семьи для одного kind."""
    if seen is None:
        seen = set()
    names = FUNC_NAMES[kind]
    for variant in VARIANTS[kind]:
        for fn in names:
            for param in variant["params"]:
                q_kw = {"fn": fn, **param}
                a = variant["code"].format(**q_kw)
                fam_key = (kind, variant["id"], fn, tuple(sorted(param.items())))
                hide = is_hidden(f"{kind}:{variant['id']}", fam_key)
                tests = make_tests(kind, variant, param, rng)
                for lang in LANGS:
                    q = variant["q"][lang].format(**q_kw)
                    sig = (q, a)
                    if sig in seen:
                        continue
                    seen.add(sig)
                    rec = {
                        "q": q, "a": a, "kind": kind, "lang": lang, "func_name": fn,
                        "tests": tests, "hidden_tests": tests,
                    }
                    (test if hide else train).append(rec)


# --------------------------------------------------------------------------------------
# "Что выведет этот код" — ответ считает сам Python (запуск в sandbox при генерации)
# --------------------------------------------------------------------------------------
PRINT_SNIPPETS = [
    "x = {a}\ny = {b}\nprint(x + y)",
    "values = [{a}, {b}, {c}]\nprint(sum(values))",
    "text = '{w}'\nprint(text[::-1])",
    "n = {a}\nprint(n % {b})",
    "items = [{a}, {b}, {c}]\nprint(len(items))",
    "n = {a}\nprint(n * {b})",
    "n = {a}\nprint(n - {b})",
    "values = [{a}, {b}, {c}]\nprint(max(values))",
    "values = [{a}, {b}, {c}]\nprint(min(values))",
    "text = '{w}'\nprint(text.upper())",
    "text = '{w}'\nprint(len(text))",
    "n = {a}\nprint(n // {b})",
]
PRINT_Q = {
    "uz": "bu kod nima chiqaradi:\n{code}",
    "ru": "что выведет этот код:\n{code}",
    "en": "what does this code print:\n{code}",
}


def build_what_prints(rng, train, test, seen=None):
    if seen is None:
        seen = set()
    words = ["hello", "world", "python", "mini", "code", "exam", "data", "model"]
    for i in range(2200):
        a_, b_, c_ = rng.randint(1, 50), rng.randint(1, 50), rng.randint(0, 50)
        snippet_i = rng.randrange(len(PRINT_SNIPPETS))
        tpl = PRINT_SNIPPETS[snippet_i]
        w = rng.choice(words)
        code = tpl.format(a=a_, b=b_, c=c_, w=w)
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(compile(code, "<snippet>", "exec"), {})
        answer = buf.getvalue().strip()
        fam_key = (snippet_i, a_, b_, c_, w)
        hide = is_hidden("what_prints", fam_key)
        for lang in LANGS:
            q = PRINT_Q[lang].format(code=code)
            sig = (q, answer)
            if sig in seen:
                continue
            seen.add(sig)
            rec = {
                "q": q, "a": answer, "kind": "what_prints", "lang": lang,
                "func_name": "", "tests": [], "hidden_tests": [],
            }
            (test if hide else train).append(rec)


# --------------------------------------------------------------------------------------
# "Исправь ошибку" / "Сделай код чище" — строятся ПРОГРАММНО из эталонов VARIANTS
# --------------------------------------------------------------------------------------
BUG_DESC = {
    "flip_operator": _q(
        "kodda shart operatori noto'g'ri tanlangan, shu sabab natija ba'zi holatlarda xato",
        "в коде неверно выбран оператор сравнения, из-за этого результат неверен в некоторых случаях",
        "the comparison operator in the condition is wrong, so the result is incorrect in some cases"),
    "off_by_one": _q(
        "siklda bitta elementga xato bor (off-by-one), chegara noto'g'ri",
        "в цикле ошибка на единицу (off-by-one) — граница выбрана неверно",
        "there is an off-by-one error in the loop — the boundary is wrong"),
    "wrong_init": _q(
        "o'zgaruvchining boshlang'ich qiymati noto'g'ri tanlangan",
        "начальное значение переменной выбрано неверно",
        "the variable's initial value is chosen incorrectly"),
    "missing_return": _q(
        "funksiya ba'zi holatlarda natijani qaytarmaydi",
        "функция в некоторых случаях не возвращает результат",
        "the function fails to return a result in some cases"),
}
BUG_Q = {
    "uz": "bu kodda xato bor: {desc}. Tuzat:\n{code}",
    "ru": "в этом коде ошибка: {desc}. исправь:\n{code}",
    "en": "this code has a bug: {desc}. fix it:\n{code}",
}
CLEANUP_Q = {
    "uz": "bu kod ishlaydi, lekin uni yozuvi chiroyli emas. Toza qilib qayta yoz:\n{code}",
    "ru": "этот код работает, но написан некрасиво. перепиши чище:\n{code}",
    "en": "this code works but is messy, rewrite it cleanly:\n{code}",
}


class _FlipOperator(ast.NodeTransformer):
    OPP = {ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Lt: ast.LtE, ast.LtE: ast.Lt,
           ast.Eq: ast.NotEq, ast.NotEq: ast.Eq}

    def __init__(self):
        self.done = False

    def visit_Compare(self, node):
        if not self.done and len(node.ops) == 1 and type(node.ops[0]) in self.OPP:
            node.ops[0] = self.OPP[type(node.ops[0])]()
            self.done = True
        return self.generic_visit(node)


class _OffByOne(ast.NodeTransformer):
    def __init__(self):
        self.done = False

    def visit_Call(self, node):
        self.generic_visit(node)
        if not self.done and isinstance(node.func, ast.Name) and node.func.id == "range" and node.args:
            last = node.args[-1]
            if isinstance(last, ast.BinOp) and isinstance(last.op, ast.Add):
                node.args[-1] = last.left
                self.done = True
            elif isinstance(last, ast.Constant) and isinstance(last.value, int):
                node.args[-1] = ast.Constant(value=last.value - 1)
                self.done = True
        return node


class _WrongInit(ast.NodeTransformer):
    def __init__(self):
        self.done = False

    def visit_Assign(self, node):
        if not self.done and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and isinstance(node.value, ast.Constant) and node.value.value in (0, 1):
            node.value = ast.Constant(value=1 if node.value.value == 0 else 0)
            self.done = True
        return node


class _MissingReturn(ast.NodeTransformer):
    def __init__(self):
        self.done = False

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if not self.done and node.body and isinstance(node.body[-1], ast.Return):
            node.body[-1] = ast.Pass()
            self.done = True
        return node


BUG_TRANSFORMS = [
    ("flip_operator", _FlipOperator),
    ("off_by_one", _OffByOne),
    ("wrong_init", _WrongInit),
    ("missing_return", _MissingReturn),
]


def _apply_transform(code, cls):
    tree = ast.parse(code)
    transformer = cls()
    new_tree = transformer.visit(tree)
    if not getattr(transformer, "done", False):
        return None
    ast.fix_missing_locations(new_tree)
    return ast.unparse(new_tree) + "\n"


def _run_fn(code, func_name, args):
    ns = {}
    try:
        exec(compile(code, "<t>", "exec"), ns)
        return True, ns[func_name](*args)
    except Exception:
        return False, None


def _verify_breaks(buggy_code, clean_code, func_name, tests):
    """Возвращает True, если buggy_code не совпадает с ожиданиями хотя бы на одном тесте."""
    for t in tests:
        ok, got = _run_fn(buggy_code, func_name, t["args"])
        if (not ok) or got != t["expected"]:
            return True
    return False


class _StripDocstring(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) \
                and isinstance(node.body[0].value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
        return node


class _StripAnnotations(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        for a in node.args.args:
            a.annotation = None
        node.returns = None
        return node


class _RenameSingleLetter(ast.NodeTransformer):
    """Переименовывает параметры и локальные переменные в однобуквенные имена."""

    def visit_FunctionDef(self, node):
        letters = iter("abcdefghijklmnopqrstuvwxyz")
        mapping = {}
        for a in node.args.args:
            mapping[a.arg] = next(letters)
        for n in ast.walk(node):
            if isinstance(n, ast.Assign):
                for t in n.targets:
                    if isinstance(t, ast.Name) and t.id not in mapping:
                        mapping[t.id] = next(letters, t.id)
            if isinstance(n, ast.comprehension) and isinstance(n.target, ast.Name) and n.target.id not in mapping:
                mapping[n.target.id] = next(letters, n.target.id)
            if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id not in mapping:
                mapping[n.target.id] = next(letters, n.target.id)
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id in mapping:
                n.id = mapping[n.id]
            if isinstance(n, ast.arg) and n.arg in mapping:
                n.arg = mapping[n.arg]
        return node


def _uglify(code, rng):
    tree = ast.parse(code)
    tree = _StripDocstring().visit(tree)
    tree = _StripAnnotations().visit(tree)
    if rng.random() < 0.6:
        tree = _RenameSingleLetter().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree) + "\n"


def _collect_reference_families():
    """Все (kind, variant, fn, param) из VARIANTS, с эталонным кодом и одним набором тестов."""
    rng = random.Random(12345)
    out = []
    for kind, variants in VARIANTS.items():
        for variant in variants:
            for fn in FUNC_NAMES[kind]:
                for param in variant["params"]:
                    code = variant["code"].format(fn=fn, **param)
                    tests = make_tests(kind, variant, param, rng, n=4)
                    out.append((kind, variant["id"], fn, param, code, tests))
    return out


def build_fix_bug(rng, train, test, seen=None):
    if seen is None:
        seen = set()
    families = _collect_reference_families()
    rng.shuffle(families)
    n_built = 0
    for kind, vid, fn, param, code, tests in families:
        if n_built >= 320:
            break
        applied = False
        for t_name, t_cls in BUG_TRANSFORMS:
            buggy = _apply_transform(code, t_cls)
            if buggy is None:
                continue
            if not _verify_breaks(buggy, code, fn, tests):
                continue
            fam_key = (kind, vid, fn, tuple(sorted(param.items())), t_name)
            hide = is_hidden("fix_bug", fam_key)
            for lang in LANGS:
                q = BUG_Q[lang].format(desc=BUG_DESC[t_name][lang], code=buggy)
                sig = (q, code)
                if sig in seen:
                    continue
                seen.add(sig)
                rec = {
                    "q": q, "a": code, "kind": "fix_bug", "lang": lang, "func_name": fn,
                    "tests": tests, "hidden_tests": tests,
                }
                (test if hide else train).append(rec)
            applied = True
            n_built += 1
            break
        if not applied:
            continue


def build_cleanup(rng, train, test, seen=None):
    if seen is None:
        seen = set()
    families = _collect_reference_families()
    rng.shuffle(families)
    n_built = 0
    for kind, vid, fn, param, code, tests in families:
        if n_built >= 320:
            break
        ugly = _uglify(code, rng)
        broke = False
        for t in tests:
            ok1, got1 = _run_fn(ugly, fn, t["args"])
            _ok2, got2 = _run_fn(code, fn, t["args"])
            if (not ok1) or got1 != got2:
                broke = True
                break
        if broke:
            continue
        fam_key = (kind, vid, fn, tuple(sorted(param.items())), "ugly")
        hide = is_hidden("cleanup", fam_key)
        for lang in LANGS:
            q = CLEANUP_Q[lang].format(code=ugly)
            sig = (q, code)
            if sig in seen:
                continue
            seen.add(sig)
            rec = {
                "q": q, "a": code, "kind": "cleanup", "lang": lang, "func_name": fn,
                "tests": tests, "hidden_tests": tests, "ugly": ugly,
            }
            (test if hide else train).append(rec)
        n_built += 1


def main():
    rng = random.Random(0)
    train, test = [], []
    seen = set()

    for kind in VARIANTS:
        build_basic(rng, kind, train, test, seen)
    build_what_prints(random.Random(0), train, test, seen)
    build_fix_bug(random.Random(0), train, test, seen)
    build_cleanup(random.Random(0), train, test, seen)

    out = ROOT / "data" / "python_tasks.json"
    out.write_text(json.dumps({"train": train, "test": test}, ensure_ascii=False), encoding="utf-8")
    kinds = sorted({x["kind"] for x in train + test})
    print(f"{out}: train {len(train)}, test {len(test)}")
    for k in kinds:
        tr_u = len({(x["q"], x["a"]) for x in train if x["kind"] == k})
        te_u = len({(x["q"], x["a"]) for x in test if x["kind"] == k})
        print(f"  {k:24s} train {sum(x['kind'] == k for x in train):5d} (unique {tr_u:5d})"
              f"  test {sum(x['kind'] == k for x in test):4d} (unique {te_u:4d})")


if __name__ == "__main__":
    main()
