"""fetch_ru.py — данные для русского языка (по очереди, не всё сразу):
  1. ru-en словарь MUSE (facebook, ~100k пар) → data/ru/dict_ru_en.txt
  2. OPUS-100 en-ru параллельные предложения (parquet, 124 МБ, 1M пар) → data/ru/opus_en_ru.parquet
  3. Русская Википедия: вступления случайных статей через API порциями (limit N) → data/ru/ruwiki_intro.jsonl (дозаписывается)
Запуск: python3 tools/fetch_ru.py dict | opus | wiki N
"""
import os, sys, json, time, re, urllib.request, urllib.parse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); OUT = os.path.join(ROOT, "data/ru"); os.makedirs(OUT, exist_ok=True)
what = sys.argv[1] if len(sys.argv) > 1 else "dict"

def get(url, binary=False, retries=4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "mini-ai/0.1 (research; contact via github mamazanovfarhod016)"})
            with urllib.request.urlopen(req, timeout=120) as r: return r.read() if binary else r.read().decode("utf-8", "ignore")
        except Exception as e:
            print("retry", i, e, flush=True); time.sleep(5 * (i + 1))
    raise RuntimeError(url)

if what == "dict":
    for name in ("ru-en.txt", "en-ru.txt"):
        p = os.path.join(OUT, "dict_" + name.replace("-", "_"))
        if os.path.exists(p): print("have", p); continue
        txt = get(f"https://dl.fbaipublicfiles.com/arrival/dictionaries/{name}")
        open(p, "w").write(txt); print(p, len(txt.splitlines()), "pairs")
elif what == "opus":
    p = os.path.join(OUT, "opus_en_ru.parquet")
    if os.path.exists(p): print("have", p)
    else:
        b = get("https://huggingface.co/datasets/Helsinki-NLP/opus-100/resolve/main/en-ru/train-00000-of-00001.parquet", binary=True)
        open(p, "wb").write(b); print(p, len(b) // 1e6, "MB")
elif what == "wiki":
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    p = os.path.join(OUT, "ruwiki_intro.jsonl"); seen = set()
    if os.path.exists(p):
        for line in open(p): seen.add(json.loads(line)["title"])
    print("have", len(seen), "articles"); n = 0
    while n < N:
        params = dict(action="query", format="json", generator="random", grnnamespace=0, grnlimit=20, prop="extracts", exintro=1, explaintext=1, exlimit=20)
        d = json.loads(get("https://ru.wikipedia.org/w/api.php?" + urllib.parse.urlencode(params)))
        with open(p, "a") as f:
            for pg in d.get("query", {}).get("pages", {}).values():
                t, ex = pg.get("title"), (pg.get("extract") or "").strip()
                if not ex or t in seen or len(ex) < 200: continue
                ex = re.sub(r"\s+", " ", ex)[:3000]
                f.write(json.dumps(dict(title=t, text=ex), ensure_ascii=False) + "\n"); seen.add(t); n += 1
        if n % 200 < 20: print(n, "new articles, total", len(seen), flush=True)
        time.sleep(0.3)
    print("done", n, "new; total", len(seen))
