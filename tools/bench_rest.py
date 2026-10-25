"""bench_rest — часть bench.py после bAbI и мира-350 (SciQ 300 + операторы 13), когда полный bench падает по памяти рядом с сервером."""
import os, sys, re
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench.py")).read()
a = src.index("# 1. bAbI"); b = src.index("# 3. SciQ")
src = src[:a] + "babi = {}\nfrom mini_ai.experimental import ExperimentalMind\n" + src[b:]
exec(compile(src, "bench_rest", "exec"))
