"""learn_ru.py — ВЫУЧИТЬ пары ru↔en из параллельных предложений (OPUS-100), без градиента.
Мешки слов: для каждой пары предложений (ru, en) — все (ru-основа, en-слово). Пара выучена, если
  PMI = log( c(ru,en)·N / (c(ru)·c(en)) ) ≥ 2.5  и  c(ru,en) ≥ 5  и  en — лучший партнёр для ru (argmax по PMI среди кандидатов с c≥5).
Это «кто с кем встречается чаще, чем случайно» — та же идея, что meaning.py, но между языками.
Результат: data/ru/learned_ru_en.txt (ru en weight) — подхватывается russian.py как второй словарь (после MUSE).
Запуск: python3 tools/learn_ru.py [limit=300000]   (память: ~1.2 ГБ при 300k пар; идём порциями по 100k, сохраняя счётчики)
"""
import os, sys, re, math, json, collections, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_ai.russian import rtoks, stem, STOP_RU, RU_DIR
import pandas as pd
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a); LIMIT = int(args.get("limit", 300000)); CHUNK = 100000
EN_STOP = set("the a an of in on at to for by with from and or is are was were be been do does did what which who it its their there as into than then so such not no he she they we you i his her my your our this that these those have has had will would can could should may might".split())
def etoks(s): return [w for w in re.findall(r"[a-z]+", s.lower()) if w not in EN_STOP and len(w) > 2]
P = os.path.join(RU_DIR, "opus_en_ru.parquet"); OUT = os.path.join(RU_DIR, "learned_ru_en.txt"); STATE = os.path.join(RU_DIR, "learn_ru_counts.json")
df = pd.read_parquet(P)
c_ru, c_en, c_pair = collections.Counter(), collections.Counter(), collections.Counter(); N = 0; start = 0
if os.path.exists(STATE):
    st = json.load(open(STATE)); c_ru.update(st["ru"]); c_en.update(st["en"]); c_pair.update({tuple(k.split("\t")): v for k, v in st["pair"].items()}); N = st["N"]; start = st["start"]
    print("resume from", start, "N", N, flush=True)
t0 = time.time()
for i in range(start, min(LIMIT, len(df))):
    tr = df.iloc[i]["translation"]; ru, en = tr.get("ru", ""), tr.get("en", "")
    rw = {stem(w) for w in rtoks(ru) if w not in STOP_RU and len(w) > 2 and re.fullmatch(r"[а-я]+", w)}
    ew = set(etoks(en))
    if not rw or not ew or len(rw) > 25 or len(ew) > 25: continue
    N += 1
    for r in rw: c_ru[r] += 1
    for e in ew: c_en[e] += 1
    for r in rw:
        for e in ew: c_pair[(r, e)] += 1
    if (i + 1) % CHUNK == 0 or i + 1 == min(LIMIT, len(df)):
        # чистка редких пар (память): пара, встретившаяся 1 раз за 100k — шум
        c_pair = collections.Counter({k: v for k, v in c_pair.items() if v >= 2})
        json.dump(dict(ru=c_ru, en=c_en, pair={"\t".join(k): v for k, v in c_pair.items()}, N=N, start=i + 1), open(STATE, "w"), ensure_ascii=False)
        print(f"{i+1} pairs read, N={N}, ru {len(c_ru)}, en {len(c_en)}, pairs kept {len(c_pair)}, {time.time()-t0:.0f}s", flush=True)
# извлечение
best = {}
for (r, e), c in c_pair.items():
    if c < 5 or c_ru[r] < 5 or c_en[e] < 5: continue
    pmi = math.log(c * N / (c_ru[r] * c_en[e]))
    if pmi < 2.5: continue
    if r not in best or pmi > best[r][1]: best[r] = (e, pmi, c)
with open(OUT, "w", encoding="utf-8") as f:
    for r, (e, pmi, c) in sorted(best.items(), key=lambda kv: -kv[1][2]): f.write(f"{r} {e} {round(pmi, 2)}\n")
print("learned", len(best), "ru-stem → en pairs ->", OUT)
for r, (e, pmi, c) in list(sorted(best.items(), key=lambda kv: -kv[1][2]))[:25]: print(f"  {r} → {e}  pmi {pmi:.2f}  n {c}")
