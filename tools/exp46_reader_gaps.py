import sys, re, collections, random; sys.path.insert(0, '.')
import pandas as pd
from mini_ai.reader import Reader
from mini_ai.memory_v2 import toks
train = pd.read_parquet("data/school/sciq/data/train-00000-of-00001.parquet")
sents = list(dict.fromkeys(s.strip() for sup in train["support"].dropna() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", str(sup)) if 20 < len(s.strip()) < 400))
r = Reader(); miss = []
for s in sents[:3000]:
    if not r.read(s, "")[0]: miss.append(s)
print(f"без фактов: {len(miss)}/3000")
print(collections.Counter(" ".join(toks(s)[:2]) for s in miss).most_common(25))
random.seed(1)
for s in random.sample(miss, 15): print("  -", s[:120])
