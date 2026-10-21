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
