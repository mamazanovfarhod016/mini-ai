"""Эксперимент 43: СОЗДАНИЕ КОДА без нейросети — память образцов + сборка + проверка запуском + исправление.
Фаза 1: агент читает учебник (20 функций с docstring — как человек читает книгу по Python).
Фаза 2: 12 задач с тестами. Из них 6 — есть близкий образец (переименовать), 4 — композиция двух образцов,
        2 — нет образца (ожидаем провал — честно).
Фаза 3: те же задачи после обучения → из памяти, 0 попыток перебора.
Метрика: решено / 12; сколько через композицию; сколько исправлено по ошибке.
"""
import sys, os, json, time, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_ai.coder import CodeMemory, Coder

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"; HOME = f"{OUT}/exp43_coder"; shutil.rmtree(HOME, ignore_errors=True); os.makedirs(HOME)

TEXTBOOK = '''
def reverse_string(s):
    """Reverse a string."""
    return s[::-1]

def count_words(text):
    """Count the number of words in text."""
    return len(text.split())

def is_palindrome(s):
    """Check if a string reads the same backwards."""
    t = s.lower().replace(" ", "")
    return t == t[::-1]

def sum_list(nums):
    """Sum all numbers in a list."""
    return sum(nums)

def max_of_list(nums):
    """Return the largest number in a list."""
    return max(nums)

def sort_ascending(nums):
    """Sort numbers from smallest to largest."""
    return sorted(nums)

def unique_items(items):
    """Remove duplicates, keep order."""
    seen = set(); out = []
    for x in items:
        if x not in seen: seen.add(x); out.append(x)
    return out

def to_upper(s):
    """Convert string to uppercase."""
    return s.upper()

def count_vowels(s):
    """Count vowels in a string."""
    return sum(1 for ch in s.lower() if ch in "aeiou")

def factorial(n):
    """Factorial of n."""
    r = 1
    for i in range(2, n + 1): r *= i
    return r

def fibonacci(n):
    """n-th Fibonacci number, fib(0)=0, fib(1)=1."""
    a, b = 0, 1
    for _ in range(n): a, b = b, a + b
    return a

def is_even(n):
    """True if n is even."""
    return n % 2 == 0

def average(nums):
    """Average of numbers."""
    return sum(nums) / len(nums)

def char_frequency(s):
    """Dictionary of character counts."""
    d = {}
    for ch in s: d[ch] = d.get(ch, 0) + 1
    return d

def join_words(words):
    """Join a list of words with spaces."""
    return " ".join(words)

def split_words(text):
    """Split text into a list of words."""
    return text.split()

def squares(nums):
    """Square every number in a list."""
    return [x * x for x in nums]

def filter_even(nums):
    """Keep only even numbers."""
    return [x for x in nums if x % 2 == 0]

def first_item(items):
    """First element of a list."""
    return items[0]

def string_length(s):
    """Length of a string."""
    return len(s)
'''

TASKS = [  # (задача, имя функции, тесты [(args, expected)])
    ("reverse the given string", "rev", [(["abc"], "cba"), (["hello"], "olleh")]),
    ("count how many words are in the text", "nwords", [(["a b c"], 3), (["hello world"], 2)]),
    ("return the biggest number from the list", "biggest", [([[1, 5, 3]], 5), ([[-2, -1]], -1)]),
    ("sort the numbers from smallest to largest", "sort_nums", [([[3, 1, 2]], [1, 2, 3])]),
    ("compute factorial of n", "fact", [([5], 120), ([0], 1)]),
    ("check whether the number is even", "even", [([4], True), ([7], False)]),
    # композиция
    ("reverse the string and convert to uppercase", "rev_upper", [(["abc"], "CBA")]),
    ("sum of the squares of the numbers", "sum_squares", [([[1, 2, 3]], 14)]),
    ("number of unique items in the list", "n_unique", [([[1, 1, 2, 3, 3]], 3)]),
    ("largest of the even numbers in the list", "max_even", [([[1, 4, 7, 2]], 4)]),
    # нет образца
    ("convert a number to binary string", "to_bin", [([5], "101"), ([2], "10")]),
    ("return the second largest number", "second_max", [([[1, 5, 3]], 3)]),
]

mem = CodeMemory(f"{HOME}/code_memory.json"); n = mem.learn_from_code(TEXTBOOK, "textbook")
print(f"прочитал учебник: {n} функций-образцов")
coder = Coder(mem)

def phase(label):
    ok = comp = fixed = 0; t0 = time.time(); print(f"--- {label} ---")
    for task, fname, tests in TASKS:
        r = coder.solve(task, fname, tests)
        good = r["status"] == "ok"; parts = r.get("parts") or []; ok += good; comp += good and len(parts) > 1; fixed += good and (r.get("fixes") or 0) > 0
        how = f"из {'+'.join(parts)} (проб {r['tried']}, правок {r.get('fixes', 0)})" if good else f"не собрал за {r['tried']}"
        print(f"  {'OK ' if good else 'FAIL'} {task:50} ← {how}")
        if good and len(parts) > 1 and fname == "rev_upper": print("       " + str(r["code"]).strip().replace("\n", "\n       "))
    print(f"  {label}: {ok}/{len(TASKS)} (композиций {comp}, исправлено по ошибке {fixed}), {time.time()-t0:.1f}s\n"); return ok

r1 = phase("ФАЗА 2: решение (перебор образцов + композиция + проверка)")
mem.save(); mem2 = CodeMemory(f"{HOME}/code_memory.json"); coder = Coder(mem2)
print(f"перезапуск: {len(mem2.items)} образцов (учебник + решённые)")
r2 = phase("ФАЗА 3: после перезапуска")
json.dump(dict(first=r1, restart=r2, n=len(TASKS), samples=len(mem2.items)), open(f"{OUT}/exp43_coder.json", "w"))
