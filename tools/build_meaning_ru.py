"""build_meaning_ru.py — смысл РУССКИХ слов из контекстов (тот же meaning.py, D=4096, ±3), по очереди порциями.
Источники по порядку: data/ru/ruwiki_intro.jsonl (вступления статей), data/corpus/ru.txt (TinyStories ru + wiki), OPUS ru-сторона.
→ ~/Developer/mini-ai-runs/learning-core/meaning_ru.npz (слова с n≥3). Память: чистка каждые 30k предложений (как build_meaning.py).
Запуск: python3 tools/build_meaning_ru.py [limit=200000]
"""
import os, sys, re, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import mini_ai.meaning as MM
from mini_ai.meaning import Meaning
from mini_ai.russian import RU_DIR, RU_BASE_NPZ
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a); LIMIT = int(args.get("limit", 200000))
MM.toks = lambda s: re.findall(r"[а-яёa-z0-9]+", s.lower().replace("ё", "е"))        # meaning.read использует модульный toks → русские буквы
MM.STOP = set("и а но или же ли бы не ни это этот эта эти тот та то те он она оно они его ее их ему ей им в во на с со у к ко о об от до из за по под над при для про без через между есть был была было были будет быть "
              "мы вы я ты мне тебе нам вам мой моя мое мои наш ваш свой так также тоже уже еще очень как что кто где когда который которая которое которые чем".split())
M = Meaning(None); t0 = time.time(); n = 0
def feed(text):
    global n
    for s in re.split(r"(?<=[.!?])\s+", text):
        if len(s) < 20: continue
        M.read(s); n += 1
        if n % 30000 == 0:
            rare = [w for w in M.sem if M.n[w] <= 1]
            for w in rare: del M.sem[w]; del M.n[w]
            for w in M.sem: M.sem[w] = M.sem[w].astype(np.float16)
            M.base = {w: M.base[w] for w in M.sem if w in M.base}
            print(n, "sentences,", len(M.sem), "words kept,", f"{time.time()-t0:.0f}s", flush=True)
        if n >= LIMIT: return True
    return False
done = False
p = os.path.join(RU_DIR, "ruwiki_intro.jsonl")
if os.path.exists(p):
    for line in open(p):
        if feed(json.loads(line)["text"]): done = True; break
p = os.path.join(os.path.dirname(RU_DIR), "corpus/ru.txt")
if not done and os.path.exists(p):
    for line in open(p):
        if feed(line.strip()): done = True; break
p = os.path.join(RU_DIR, "opus_en_ru.parquet")
if not done and os.path.exists(p):
    import pandas as pd
    df = pd.read_parquet(p)
    for i in range(len(df)):
        if feed(df.iloc[i]["translation"].get("ru", "")): done = True; break
keep = [w for w in M.sem if M.n[w] >= 3]; M.sem = {w: M.sem[w] for w in keep}
M.path = RU_BASE_NPZ; M._dirty = 1; M.save(force=True)
print(f"{n} sentences, {len(keep)} words, {time.time()-t0:.0f}s -> {RU_BASE_NPZ}, {os.path.getsize(RU_BASE_NPZ)/1e6:.0f} MB")
M2 = Meaning(None); M2.load(RU_BASE_NPZ)
for a, b in (("кошка", "собака"), ("король", "королева"), ("кошка", "машина"), ("вода", "алгебра"), ("город", "деревня"), ("красный", "синий")):
    print(f"  {a}–{b}: {M2.similar(a, b)}")
for w in ("столица", "река", "учёный", "кошка"): print(" ", w, "→", M2.neighbors(w, k=6, min_n=5))
