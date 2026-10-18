"""
Этап 1 «чтение»: собираем корпус простых текстов на трёх языках (uz, ru, en), чтобы Mini AI
сначала научилась ЯЗЫКУ (угадывать следующее слово в обычном тексте), а уже потом — вопросам и ответам.
Так учатся и большие модели: сначала много текста, потом диалоги.

    .venv/bin/python tools/make_corpus.py            # → data/corpus/{uz,ru,en}.txt + data/corpus/REPORT.md

Источники (все с открытой лицензией, ссылки в REPORT.md):
  en  TinyStories V2 (roneneldan/TinyStories, CDLA-Sharing-1.0) — простые детские рассказы, написаны GPT-4
  ru  TinyStories-Multilingual (Dxniz, Apache-2.0) — те же рассказы по-русски + вступления статей Википедии (CC BY-SA)
  uz  TinyStories-Multilingual (uz) + узбекская Википедия (ffatty/plaintext-wikipedia-full-uzbek, CC BY-SA)

КАЧЕСТВО ВАЖНЕЕ ОБЪЁМА — каждый текст проходит фильтры (скрипт, не ИИ):
  1. язык: uz — латиница с узбекскими буквами (o‘ g‘ sh ch), ru — кириллица, en — латиница; чужие буквы → выкинуть;
  2. предложение 4–30 слов, без ссылок, вики-разметки, таблиц, длинных чисел;
  3. ПРОСТОТА: в предложении ≥ 90% слов из 8000 самых частых слов этого языка (в этом же корпусе);
  4. без повторов (одинаковые предложения — один раз);
  5. языки уравнены по объёму, чтобы ни один не «перевешивал».
20% предложений откладываем для проверки (perplexity на невиданном тексте).
"""
import gzip
import json
import random
import re
import time
import urllib.parse
import urllib.request
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache_corpus"
OUT = ROOT / "data/corpus"
UA = "MiniAI/0.4 (https://github.com/mamazanovfarhod016/mini-ai; study project)"
TARGET_CHARS = 12_000_000         # на язык (≈ 2 млн слов); узбекского и русского может быть меньше — берём сколько есть
HF = "https://huggingface.co/datasets"

SOURCES = {
    "tinystories_en": f"{HF}/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
    "tinystories_ml": f"{HF}/Dxniz/TinyStories-Multilingual/resolve/main/tinystories.jsonl",
    "uzwiki": f"{HF}/ffatty/plaintext-wikipedia-full-uzbek/resolve/main/data/uzwiki_01_of_03_part01_of_02.txt.gz",
    "uzwiki_02": f"{HF}/ffatty/plaintext-wikipedia-full-uzbek/resolve/main/data/uzwiki_02_of_03_part01_of_02.txt.gz",
    "uzwiki_03": f"{HF}/ffatty/plaintext-wikipedia-full-uzbek/resolve/main/data/uzwiki_03_of_03_part01_of_02.txt.gz",
    # первые 200 МБ большого файла TinyStories (скачиваются через Range — см. README корпуса)
    "tinystories_en_train": f"{HF}/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    # узбекские новости (tahrirchi/uz-crawl, Apache-2.0), один файл из 7
    "uz_news3": f"{HF}/tahrirchi/uz-crawl/resolve/main/data/news-00002-of-00007.parquet",
}
ALPHA = {
    "uz": re.compile(r"^[a-zA-Zʻʼ‘’'`ʻ\s.,!?:;\-—–\"«»()0-9]+$"),
    "ru": re.compile(r"^[а-яА-ЯёЁ\s.,!?:;\-—–\"«»()0-9]+$"),
    "en": re.compile(r"^[a-zA-Z\s.,!?:;\-—–\"'()0-9]+$"),
}
UZ_MARK = re.compile(r"[oOgG][ʻ‘’'`]|sh|ch|q|x", re.I)   # признаки узбекского (отличаем от английского)


def fetch(name, url):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / (name + Path(urllib.parse.urlparse(url).path).suffix)
    if not path.exists():
        print(f"  скачиваю {name} …", flush=True)
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        with urllib.request.urlopen(req, timeout=300) as r:
            path.write_bytes(r.read())
    return path


def ru_wiki_intros(n_calls=1000):
    """Вступления случайных статей русской Википедии (API, без ключей). Кэшируем."""
    path = CACHE / "ruwiki_intros.jsonl"
    if path.exists():
        return [json.loads(x)["t"] for x in path.read_text(encoding="utf-8").splitlines()]
    out = []
    for i in range(n_calls):
        q = urllib.parse.urlencode({"action": "query", "format": "json", "generator": "random", "grnnamespace": 0,
                                    "grnlimit": 20, "prop": "extracts", "exintro": 1, "explaintext": 1, "exlimit": 20})
        req = urllib.request.Request("https://ru.wikipedia.org/w/api.php?" + q, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                pages = json.loads(r.read()).get("query", {}).get("pages", {})
            out += [p.get("extract", "") for p in pages.values() if p.get("extract")]
        except Exception as e:  # сеть моргнула — пропускаем один запрос
            print("   ruwiki:", e)
        time.sleep(0.3)
        if i % 50 == 0:
            print(f"  ruwiki: {i}/{n_calls} запросов, {len(out)} статей", flush=True)
    path.write_text("\n".join(json.dumps({"t": t}, ensure_ascii=False) for t in out), encoding="utf-8")
    return out


def sentences(text):
    text = re.sub(r"\s+", " ", text).replace("ʻ", "'").replace("‘", "'").replace("’", "'").replace("ʼ", "'")
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ\"«])", text) if s.strip()]


def words(s):
    return re.findall(r"[^\W\d_]+(?:[ʻ‘’'`][^\W\d_]+)*", s.lower())


def clean_ok(s, lang):
    if not (6 <= len(s.split()) <= 30) or not ALPHA[lang].match(s):
        return False
    if len(re.findall(r"\d+", s)) > 1 or re.search(r"\b(km2|г\.|гг\.|см\.|т\. е|англ|лат)\b", s):
        return False                                   # справочные обрывки: «Телефонный код — 3238.»
    if sum(ch.isupper() for ch in s) > max(3, len(s.split()) // 2):
        return False                                   # списки имён и аббревиатур
    if re.search(r"https?:|www\.|\d{5,}|==|\|", s) or s[-1] not in ".!?\"»":
        return False
    if lang == "uz" and not UZ_MARK.search(s):
        return False
    if lang == "en" and re.search(r"[oOgG][ʻ‘]", s):
        return False
    return True


def simple(sents, lang, top=8000, share=0.9):
    """Оставляем предложения, где ≥ 90% слов — из 8000 самых частых слов языка."""
    freq = Counter(w for s in sents for w in words(s))
    common = {w for w, _ in freq.most_common(top)}
    out = []
    for s in sents:
        ws = words(s)
        if ws and sum(w in common for w in ws) / len(ws) >= share:
            out.append(s)
    return out


def main():
    raw = {"uz": [], "ru": [], "en": []}
    src_count = Counter()

    en = fetch("tinystories_en", SOURCES["tinystories_en"]).read_text(encoding="utf-8", errors="ignore")
    big = CACHE / "tinystories_en_train.txt"
    if big.exists():
        en += "\n" + big.read_text(encoding="utf-8", errors="ignore")
    for story in en.split("<|endoftext|>"):
        ss = sentences(story)
        raw["en"] += ss
        src_count["en: TinyStories"] += len(ss)

    for line in fetch("tinystories_ml", SOURCES["tinystories_ml"]).read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        if d["language_code"] in ("ru", "uz"):
            ss = sentences(d["output"])
            raw[d["language_code"]] += ss
            src_count[f"{d['language_code']}: TinyStories-Multilingual"] += len(ss)

    for key in ("uzwiki", "uzwiki_02", "uzwiki_03"):
        with gzip.open(fetch(key, SOURCES[key]), "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                ss = sentences(line)
                raw["uz"] += ss
                src_count["uz: Википедия"] += len(ss)
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(fetch("uz_news3", SOURCES["uz_news3"]), columns=["text"])
        for t in table.column("text").to_pylist()[:300_000]:
            ss = sentences(t or "")
            raw["uz"] += ss
            src_count["uz: новости (uz-crawl)"] += len(ss)
    except ImportError:
        print("  pyarrow нет — узбекские новости пропускаю")

    for t in ru_wiki_intros():
        ss = sentences(t)
        raw["ru"] += ss
        src_count["ru: Википедия"] += len(ss)

    OUT.mkdir(parents=True, exist_ok=True)
    report = ["# Корпус для этапа «чтение»\n", "| язык | предложений сырых | после фильтров | символов (обучение) | проверка |",
              "|---|---|---|---|---|"]
    stats = {}
    for lang in ("uz", "ru", "en"):
        seen, ok = set(), []
        for s in raw[lang]:
            key = s.lower()
            if key not in seen and clean_ok(s, lang):
                seen.add(key)
                ok.append(s)
        ok = simple(ok, lang)
        random.Random(0).shuffle(ok)
        picked, n = [], 0
        for s in ok:                                   # уравниваем языки по объёму
            if n >= TARGET_CHARS:
                break
            picked.append(s)
            n += len(s) + 1
        test = [s for s in picked if zlib.crc32(s.encode()) % 5 == 0]
        train = [s for s in picked if zlib.crc32(s.encode()) % 5 != 0]
        (OUT / f"{lang}.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
        (OUT / f"{lang}_test.txt").write_text("\n".join(test) + "\n", encoding="utf-8")
        stats[lang] = {"raw": len(raw[lang]), "clean": len(ok), "train": len(train), "test": len(test),
                       "chars": sum(len(s) + 1 for s in train)}
        report.append(f"| {lang} | {len(raw[lang]):,} | {len(ok):,} | {stats[lang]['chars']:,} | {len(test):,} предл. |")
        print(f"{lang}: сырых {len(raw[lang]):,} → чистых {len(ok):,} → обучение {len(train):,} предл., "
              f"{stats[lang]['chars']:,} символов, проверка {len(test):,}")
    report += ["", "## Откуда", *[f"- {k}: {v:,} предложений (до фильтров)" for k, v in sorted(src_count.items())],
               "", "Ссылки: " + ", ".join(SOURCES.values()) + ", https://ru.wikipedia.org (API, вступления статей).",
               "", "## Примеры (случайные, после фильтров)"]
    for lang in ("uz", "ru", "en"):
        lines = (OUT / f"{lang}.txt").read_text(encoding="utf-8").splitlines()
        report += [f"**{lang}**"] + [f"- {s}" for s in random.Random(1).sample(lines, min(5, len(lines)))]
    (OUT / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
