"""
Mini AI, этап «чтение», набор A: ЛИТЕРАТУРА на uz / ru / en (только общественное достояние и открытые лицензии).

    .venv/bin/python tools/make_literature.py fetch     # сырьё → ~/mini-ai-raw/ (долго; запускать фоном, есть лог)
    .venv/bin/python tools/make_literature.py build     # ~/mini-ai-raw/out/{en,ru,uz}.txt (+ en_tinystories_rest.txt.gz)
    .venv/bin/python tools/make_literature.py report    # data/literature/REPORT.md + heldout (<1 МБ на язык)

Источники (ссылки и лицензии — в data/literature/REPORT.md):
  en  TinyStories V2 train (CDLA-Sharing-1.0, полный: часть в en.txt, остальное в en_tinystories_rest.txt.gz)
      + детская классика Project Gutenberg (общественное достояние; отбор по каталогу pg_catalog.csv)
  ru  ru.wikisource.org через HF-набор kristaller486/wikisource-creative-ru (Apache-2.0; тексты — PD / свободные,
      компиляция Викитеки CC BY-SA) — только писатели из белого списка, умершие до 1956 года;
      + TinyStories-Multilingual (Dxniz, Apache-2.0)
  uz  wikisource.org, раздел «O‘zbek» (Викитека, CC BY-SA; сами тексты — PD) через MediaWiki API,
      кириллица → латиница (функция translit_uz) + TinyStories-Multilingual (uz)

ФОРМАТ: одна строка = один абзац; стих = строфа, переносы строк внутри строфы заменены на « / »;
        пустая строка = граница документа (книги/рассказа). Узбекский — латиница, ASCII-апостроф (o' g').
КАЧЕСТВО: фильтры скриптом — язык строки, длина, мусор (ссылки, сноски, номера страниц, оглавления, ЗАГЛАВНЫЕ
        заголовки, старая орфография), дубли строк.
ПАМЯТЬ: всё читается и пишется потоком (по документу / по группе строк parquet), процесс < 1 ГБ.
"""
import csv
import gzip
import hashlib
import io
import json
import os
import random
import re
import sys
import time
import urllib.parse
import urllib.request
import zlib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = Path(os.environ.get("MINIAI_RAW", str(Path.home() / "mini-ai-raw")))
DL = RAW / "dl"
OUT = RAW / "out"
REP = ROOT / "data/literature"
UA = {"User-Agent": "MiniAI/0.4 (https://github.com/mamazanovfarhod016/mini-ai; study project)"}
HF = "https://huggingface.co/datasets"
URL_TS = f"{HF}/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt"
URL_DXNIZ = f"{HF}/Dxniz/TinyStories-Multilingual/resolve/main/tinystories.jsonl"
URL_RUWS = [f"{HF}/kristaller486/wikisource-creative-ru/resolve/main/data/train-0000{i}-of-00005.parquet" for i in range(5)]
URL_PGCAT = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
WS_API = "https://wikisource.org/w/api.php?format=json&"

CAP_EN_GUTENBERG = 45_000_000       # символов детской классики Gutenberg
CAP_EN_TS_SAMPLE = 45_000_000       # ≈ 2% TinyStories в основной en.txt (остальное — en_tinystories_rest.txt.gz)
CAP_RU = 50_000_000
HELDOUT_CHARS = 400_000             # на язык; документы held-out НЕ попадают в обучение
PD_DEATH_MAX = 1955                 # автор умер не позже → общественное достояние (life+70 в РФ, life+50..70 везде)


# ───────────────────────────── транслитерация uz: кириллица → латиница ─────────────────────────────
_UZ_BASE = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "ж": "j", "з": "z", "и": "i", "й": "y", "к": "k", "л": "l",
    "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u", "ф": "f", "х": "x", "ц": "ts",
    "ч": "ch", "ш": "sh", "ъ": "'", "ь": "", "э": "e", "ю": "yu", "я": "ya", "ў": "o'", "ғ": "g'", "қ": "q",
    "ҳ": "h", "ё": "yo", "ы": "i", "ѐ": "e",
}
_UZ_VOWELS = set("аеёиоуэюяўАЕЁИОУЭЮЯЎ")


def translit_uz(text):
    """Узбекская кириллица → латиница (алфавит 1995 г., ASCII-апостроф: o' g'; ъ → ').
    е в начале слова и после гласной → ye, иначе e; ц → ts. Заглавные: Ў→O', Ғ→G', Ш→Sh (ШАХАР → SHAXAR)."""
    out = []
    n = len(text)
    for i, ch in enumerate(text):
        low = ch.lower()
        if low == "е":
            prev = text[i - 1] if i else " "
            lat = "ye" if (not prev.isalpha() or prev in _UZ_VOWELS or prev in "ъЪ") else "e"
        elif low in _UZ_BASE:
            lat = _UZ_BASE[low]
        else:
            out.append(ch)
            continue
        if ch != low and lat:                               # заглавная
            nxt = text[i + 1] if i + 1 < n else " "
            if len(lat) > 1 and lat[1] != "'" and nxt.isupper():
                lat = lat.upper()                           # SH в слове ЗАГЛАВНЫМИ
            elif len(lat) > 1 and lat[1] == "'" and nxt.isupper():
                lat = lat[0].upper() + "'"
            else:
                lat = lat[0].upper() + lat[1:]
        out.append(lat)
    return "".join(out)


def norm_uz_apostrophes(s):
    """Все варианты апострофа в латинском узбекском (ʻ ʼ ‘ ’ ` ´ ′) → ASCII '."""
    s = re.sub(r"([oOgG])[\u02bb\u02bc\u2018\u2019`\u00b4\u2032](?![\u02bb\u02bc\u2018\u2019`\u00b4\u2032])", r"\1'", s)   # o‘zbek, g’ayrat, tog‘ (в конце слова тоже)
    s = re.sub(r"([a-zA-Z])[\u02bb\u02bc\u2018\u2019`\u00b4\u2032](?=[a-zA-Z])", r"\1'", s)   # ma’no, sa’y
    return s


_HOMOGLYPH = str.maketrans("АаВЕеКкМНОоРрСсТуХхІі", "AaBEeKkMHOoPpCcTyXxIi")


# ───────────────────────────── общие фильтры строк ─────────────────────────────
_STOP_EN = set("the a an and of to in is was he she it that for with his her on as at by they i you said not be but "
               "had have were this from or one all there their we him them my me what so up out if no do would".split())
_URL = re.compile(r"https?:|www\.|\.com\b|\.org\b|@\w+\.|\|\||==|\{\{|\}\}|\[\[|<[a-z/]|&[a-z]+;", re.I)
_FOOT = re.compile(r"^\s*(\[\d+\]|\(\d+\)|\d+\)|\*+|\d+\.?|[ivxlc]+\.?|стр\.?\s*\d+|page\s+\d+|\[?illustration.*|\[footnote.*)\s*$", re.I)
_OLD_RU = re.compile(r"[ѣѳѵіІѢѲѴ]|\b\w+ъ\b")


def letters(s):
    return [c for c in s if c.isalpha()]


def line_ok(s, lang):
    n = len(s)
    if not (15 <= n <= 6000) or len(s.split()) < 3:
        return False
    if _URL.search(s) or _FOOT.match(s):
        return False
    if n < 140 and re.search(r"[^\d\s,.]\s+\d{1,4}$", s) and not re.search(r"[.!?\"»”'’]$", s):
        return False                                                      # строка оглавления: «Название ... 217»
    if re.search(r"\{\d+[a-z]?\}", s):
        return False                                                      # {115c} — номера страниц/сносок
    L = letters(s)
    if len(L) < 0.6 * n:                                                  # таблицы, цифры, значки
        return False
    up = sum(c.isupper() for c in L)
    if up > 0.7 * len(L) and n < 120:                                     # ЗАГОЛОВКИ ПРОПИСНЫМИ
        return False
    if sum(c.isdigit() for c in s) > 0.08 * n:
        return False
    if lang == "ru":
        cyr = sum("а" <= c.lower() <= "я" or c.lower() == "ё" for c in L)
        if cyr < 0.88 * len(L):
            return False
        words = s.split()
        if _OLD_RU.search(s) and sum(bool(re.search(r"ъ$", w.strip('.,;:!?—«»"()'))) for w in words) > 0:
            return False                                                  # дореформенная орфография
    elif lang == "uz":
        lat = sum(("a" <= c.lower() <= "z") for c in L)
        if lat < len(L) or re.search("[\u02bb\u02bc\u2018\u2019`\u00b4\u2032]", s):   # ни кириллицы, ни «умных» апострофов
            return False
        if not re.search(r"[aeiou]", s.lower()):
            return False
    elif lang == "en":
        asc = sum(c.isascii() for c in L)
        if asc < 0.97 * len(L):
            return False
        ws = re.findall(r"[a-z']+", s.lower())
        if len(ws) >= 6 and sum(w in _STOP_EN for w in ws) < 0.15 * len(ws):
            return False                                                  # не английский (латынь, французский, список имён)
    return True


def hash64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")


def split_blocks(text):
    """Текст → блоки (абзацы / строфы) по пустым строкам; \\r убираем."""
    return [b for b in re.split(r"\n\s*\n", text.replace("\r", "")) if b.strip()]


def block_to_line(block, hard_wrapped):
    """Абзац → одна строка. Для текстов с жёсткими переносами (Gutenberg) короткие строки подряд = стих → ' / ',
    иначе строки склеиваются пробелом. Для вики-текстов (без жёстких переносов) перенос внутри блока = стих."""
    lines = [re.sub(r"\s+", " ", l).strip() for l in block.split("\n") if l.strip()]
    if len(lines) == 1:
        return lines[0]
    if hard_wrapped:
        avg = sum(map(len, lines)) / len(lines)
        verse = len(lines) >= 3 and avg <= 50 and max(map(len, lines)) <= 75
        if verse and sum(bool(re.search(r"[,;:.!?—'\"]", l)) for l in lines) < 0.5 * len(lines):
            return ""                                                     # оглавление / список заголовков
        return " / ".join(lines) if verse else " ".join(lines)
    return " / ".join(lines)


_NOTES_HEAD = re.compile(r"^\s*(Примечания|Комментарии|Примечания редакции|Izohlar|Изоҳлар|Notes|Footnotes)"
                         r"(\s+и\s+комментарии|\s*\([^)]{0,60}\)|\s+(доктора|профессора|редакции|редактора|составителя|переводчика)\b.{0,80})?\s*[.:]?\s*$", re.I)
# служебные строки Викитеки: «Русская проза / Рассказы / Литература 1896 года / …», «Импорт/lib.ru/…», «Ссылки на комментарии…»
_WS_NAV = re.compile(r"Литература \d{4}(-х)? год|Импорт/|lib\.ru|Ссылки на комментарии|Прижизненные публикации"
                     r"|^(Русская (проза|поэзия)|Проза|Поэзия|Рассказы|Сказки|Пьесы|Переводы)\b[^.!?]* / "
                     r"|^(Впервые (напечатано|опубликовано)|Печатается по|Первоначально (напечатано|опубликовано)|Полностью публикуется)")
_FOOTNOTE_LINE = re.compile(r"^\s*(\d{1,3}|\*+)\s+\S")


def doc_lines(text, lang, hard_wrapped=False):
    """Абзацы документа → строки. Раздел «Примечания/Комментарии» и всё после него отбрасываем (это поздний
    редакторский текст, часто не в общественном достоянии), а также сноски вида «3 Речь идёт о…», «* Пред — …»."""
    for block in split_blocks(text):
        head = block.strip().split("\n")[0]
        second = block.strip().split("\n")[1] if "\n" in block.strip() else ""
        if (_NOTES_HEAD.match(head)
                or re.match(r"^\s*(Примечания|Комментарии)(\s*:|\s+[А-ЯЁ]{2,}|\s+[А-ЯЁ]\.\s?[А-ЯЁ]\.)", head)
                or (re.match(r"^\s*(Примечания|Комментарии)\s+\S", head)
                    and re.match(r"^\s*(Впервые|Полностью|Первоначально|Произведение написано|Печатается)", second))):
            return
        s = block_to_line(block, hard_wrapped)
        if not s or (lang in ("ru", "uz") and (_FOOTNOTE_LINE.match(s) or _WS_NAV.search(s))):
            continue
        if line_ok(s, lang):
            yield s


class Sink:
    """Пишет документы в файл (поток), следит за дублями, считает строки и символы по источникам."""

    def __init__(self, path, gz=False):
        self.f = gzip.open(path, "wt", encoding="utf-8", compresslevel=6) if gz else open(path, "w", encoding="utf-8")
        self.seen = set()
        self.stat = Counter()
        self.first = True

    def doc(self, lines, source, dedupe=True):
        keep = []
        for s in lines:
            if dedupe:
                h = hash64(s)
                if h in self.seen:
                    continue
                self.seen.add(h)
            keep.append(s)
        if not keep or sum(map(len, keep)) < 200:
            return 0
        if not self.first:
            self.f.write("\n")                                            # пустая строка = граница документов
        self.first = False
        self.f.write("\n".join(keep) + "\n")
        c = sum(len(s) + 1 for s in keep)
        self.stat[source + "|docs"] += 1
        self.stat[source + "|lines"] += len(keep)
        self.stat[source + "|chars"] += c
        return c

    def close(self):
        self.f.close()


class Heldout:
    """Маленький проверочный набор: документы с crc32(id) % 20 == 0. Эти документы не идут в обучение."""

    def __init__(self, path):
        self.path, self.chars, self.docs, self.parts = path, 0, 0, []

    @staticmethod
    def is_held(doc_id):
        return zlib.crc32(doc_id.encode()) % 20 == 0

    def add(self, lines):
        if self.chars < HELDOUT_CHARS and lines:
            keep, n = [], 0
            for s in lines:                                               # не больше 25 000 символов с одного документа
                if n + len(s) > 25_000:
                    break
                keep.append(s)
                n += len(s) + 1
            if keep:
                self.parts.append("\n".join(keep))
                self.chars += n
                self.docs += 1

    def save(self):
        Path(self.path).write_text("\n\n".join(self.parts) + "\n", encoding="utf-8")


# ───────────────────────────── скачивание ─────────────────────────────
def download(url, path, headers=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={**UA, **(headers or {})}, method="HEAD")
    total = int(urllib.request.urlopen(req, timeout=60).headers.get("Content-Length", 0))
    have = path.stat().st_size if path.exists() else 0
    if total and have == total:
        return path
    print(f"  скачиваю {path.name} ({total / 1e6:.0f} МБ), есть {have / 1e6:.0f} МБ", flush=True)
    req = urllib.request.Request(url, headers={**UA, **({"Range": f"bytes={have}-"} if have else {})})
    with urllib.request.urlopen(req, timeout=300) as r, open(path, "ab" if have else "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
    return path


def api(url):
    for i in range(6):
        try:
            return json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60))
        except Exception as e:
            print("   api retry:", e, flush=True)
            time.sleep(3 * (i + 1))
    raise RuntimeError(url)


def fetch_uz_wikisource(path):
    """Все страницы категории «O‘zbek» (кроме Author:/Category:) → jsonl {title, text}."""
    if Path(path).exists() and Path(path).stat().st_size > 1_000_000:
        return
    def members(cat):
        out, cont = [], ""
        while True:
            r = api(WS_API + "action=query&list=categorymembers&cmlimit=500&cmtype=page&cmtitle=" + urllib.parse.quote(cat) + cont)
            out += [m["title"] for m in r["query"]["categorymembers"]]
            if "continue" not in r:
                return out
            cont = "&cmcontinue=" + urllib.parse.quote(r["continue"]["cmcontinue"])
    titles = [t for t in dict.fromkeys(members("Category:O‘zbek")) if not t.startswith(("Author:", "Wikisource:", "Category:", "Index:"))]
    print("  uz wikisource страниц:", len(titles), flush=True)
    with open(path, "w", encoding="utf-8") as out:
        for i in range(0, len(titles), 50):
            r = api(WS_API + "action=query&prop=revisions&rvprop=content&rvslots=main&titles=" + urllib.parse.quote("|".join(titles[i:i + 50])))
            for p in r["query"]["pages"].values():
                try:
                    out.write(json.dumps({"title": p["title"], "text": p["revisions"][0]["slots"]["main"]["*"]}, ensure_ascii=False) + "\n")
                except KeyError:
                    pass
            time.sleep(0.5)


# Детская классика Gutenberg: сначала авторы по списку, потом «Juvenile / Fairy tales / Folklore / Children» по каталогу
PG_KIDS = ["Grimm", "Andersen", "Aesop", "Kipling", "Carroll", "Baum", "Barrie", "Potter, Beatrix", "Lang, Andrew",
           "Stevenson, Robert", "Twain", "Alcott", "Montgomery, L. M.", "Burnett, Frances", "Grahame", "Nesbit",
           "Lofting", "Lear, Edward", "MacDonald, George", "Collodi", "Defoe", "Swift, Jonathan", "Kingsley, Charles",
           "Jacobs, Joseph", "Perrault", "Wiggin", "Webster, Jean", "Porter, Eleanor", "Spyri", "Pyle, Howard",
           "Thompson, Ruth", "Seton", "Wilde, Oscar", "Dunsany"]
PG_CLASSIC = ["Dickens", "Austen", "Doyle", "Wells, H. G.", "Verne", "London, Jack", "Hawthorne", "Poe, Edgar",
              "Irving, Washington", "Dumas", "Chesterton", "Saki", "Jerome, Jerome", "Morris, William", "Hudson, W. H."]
PG_AUTHORS = PG_KIDS + PG_CLASSIC
PG_SUBJ = re.compile(r"juvenile|fairy tales|children'?s (stories|poetry|literature)|folklore|fables|legends|nursery", re.I)


def pg_death_year(authors):
    if re.search(r"BCE|B\.C\.", authors):
        return 0                                                          # античные авторы (Эзоп и т.п.)
    years = [int(m) for m in re.findall(r"-\s*(\d{4})\b", authors)]
    return max(years) if years else None


def pg_select():
    rows = list(csv.DictReader(io.StringIO(Path(DL / "pg_catalog.csv").read_text(encoding="utf-8"))))
    rows = [r for r in rows if r["Language"] == "en" and r["Type"] == "Text" and not re.search(r"periodical|catalog|dictionary|encyclop", r["Subjects"] + r["Title"], re.I)]
    def pd_ok(r):
        d = pg_death_year(r["Authors"])
        return d is not None and d <= PD_DEATH_MAX
    rows = [r for r in rows if pd_ok(r)]
    kids = [r for r in rows if any(a in r["Authors"] for a in PG_KIDS)]
    classic = [r for r in rows if r not in kids and any(a in r["Authors"] for a in PG_CLASSIC)]
    rest = [r for r in rows if r not in kids and r not in classic and PG_SUBJ.search(r["Subjects"] + r["Bookshelves"])]
    for i, lst in enumerate((kids, classic, rest)):
        random.Random(i).shuffle(lst)
    first = kids + rest[:300]               # сначала детские авторы и детские темы, потом взрослая классика
    rest = classic + rest[300:]
    return first[:450], rest[:900]


def fetch_gutenberg(budget_chars=CAP_EN_GUTENBERG * 1.25):
    d = DL / "gutenberg"
    d.mkdir(parents=True, exist_ok=True)
    first, rest = pg_select()
    total = sum(p.stat().st_size for p in d.glob("*.txt"))
    for r in first + rest:
        if total > budget_chars:
            break
        p = d / f"{r['Text#']}.txt"
        if p.exists():
            continue
        for url in (f"https://www.gutenberg.org/cache/epub/{r['Text#']}/pg{r['Text#']}.txt",
                    f"https://www.gutenberg.org/files/{r['Text#']}/{r['Text#']}-0.txt"):
            try:
                data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()
                p.write_bytes(data)
                (d / f"{r['Text#']}.meta.json").write_text(json.dumps({"id": r["Text#"], "title": r["Title"], "authors": r["Authors"]}, ensure_ascii=False))
                total += len(data)
                break
            except Exception as e:
                last = e
        else:
            print("  нет книги", r["Text#"], last, flush=True)
        time.sleep(1.0)
    print("  gutenberg скачано МБ:", round(total / 1e6), flush=True)


def fetch():
    DL.mkdir(parents=True, exist_ok=True)
    download(URL_PGCAT, DL / "pg_catalog.csv")
    download(URL_DXNIZ, DL / "dxniz.jsonl")
    for i, u in enumerate(URL_RUWS):
        download(u, DL / f"ruws_{i}.parquet")
    fetch_uz_wikisource(DL / "uz_wikisource.jsonl")
    fetch_gutenberg()
    download(URL_TS, DL / "TinyStoriesV2-GPT4-train.txt")
    print("fetch: готово", flush=True)


# ───────────────────────────── сборка ─────────────────────────────
def gutenberg_body(raw):
    m1 = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*", raw, re.S)
    m2 = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG", raw)
    body = raw[m1.end() if m1 else 0: m2.start() if m2 else len(raw)]
    body = re.sub(r"\[(Illustration|Footnote|Transcriber|Note)[^\]]*\]", "", body, flags=re.S | re.I)
    body = body.replace("_", "")                                          # _курсив_ в Gutenberg
    body = re.sub(r"\n\s*(CHAPTER|Chapter|BOOK|PART|CONTENTS|Contents|ILLUSTRATIONS)\b[^\n]*", "\n", body)
    return body


def build_en(sink_en, sink_rest, held):
    d = DL / "gutenberg"
    first, rest = pg_select()
    order = [r["Text#"] for r in first + rest]
    chars = 0
    for pid in order:
        p = d / f"{pid}.txt"
        if not p.exists():
            continue
        meta = json.loads((d / f"{pid}.meta.json").read_text())
        lines = list(doc_lines(gutenberg_body(p.read_text(encoding="utf-8", errors="ignore")), "en", hard_wrapped=True))
        if Heldout.is_held("pg" + pid):
            held.add(lines)
            continue
        if chars < CAP_EN_GUTENBERG:
            c = sink_en.doc(lines, "en: Project Gutenberg (детская/сказочная классика)")
            chars += c
            sink_en.stat["authors|" + meta["authors"].split(",")[0]] += c
    # TinyStories: поток по строкам, рассказ = до <|endoftext|>
    seen = set()
    story = []
    held_ts = Heldout(None)
    held_ts.parts = []
    n_ts_held = 0
    def flush():
        nonlocal story, n_ts_held
        text = "\n".join(story).strip()
        story = []
        if len(text) < 200:
            return
        h = hash64(text)
        if h in seen:
            return
        seen.add(h)
        k = zlib.crc32(text.encode()) % 50
        lines = [re.sub(r"\s+", " ", l).strip() for l in text.split("\n") if l.strip()]
        lines = [l for l in lines if line_ok(l, "en") or len(l) >= 3 and l[-1] in '.!?"']   # TinyStories и так чистые
        if k == 1:                                                         # проверочные истории — ни в обучение, ни в rest
            if n_ts_held < 150:
                held.add(lines)
                n_ts_held += 1
            return
        (sink_en if k == 0 else sink_rest).doc(lines, "en: TinyStories V2 train" + ("" if k == 0 else " (rest)"), dedupe=False)
    with open(DL / "TinyStoriesV2-GPT4-train.txt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("<|endoftext|>"):
                flush()
            else:
                story.append(line.rstrip("\n"))
    flush()


RU_TIER1 = ["Пушкин", "Лермонтов", "Толстой", "Чехов", "Крылов", "Андерсен", "Гримм", "Афанасьев", "Даль", "Одоевский",
            "Ушинский", "Аксаков", "С. Т. Аксаков", "Гоголь", "Жуковский", "Некрасов", "Бажов", "Гайдар", "Житков", "Куприн",
            "Тургенев", "Горький", "Мамин-Сибиряк", "Ершов", "Перро", "Гаршин", "Салтыков-Щедрин", "Фет", "Тютчев"]
RU_OTHER = ["Достоевский", "Аверченко", "Мережковский", "Гиппиус", "Лесков", "Карамзин", "Гончаров", "Тэффи", "Серафимович",
            "Ходасевич", "Андреев", "Короленко", "Булгаков", "Ильф и Петров", "Сологуб", "Беляев", "Островский", "Лондон",
            "Брюсов", "Замятин", "Белый", "Бунин", "Мопассан", "Блок", "Дойль", "О. Генри", "Бальмонт", "Флобер", "Майн Рид",
            "Грин", "Успенский", "Чарская", "Джекобс", "Уэллс", "Тынянов", "По", "Диккенс", "Твен", "Верн", "Киплинг", "Гофман",
            "Стивенсон", "Свифт", "Дефо", "Кэрролл", "Пришвин", "Гарин-Михайловский", "Станюкович", "Немирович-Данченко",
            "Вяземский", "Огарев", "Полонский", "Майков", "Плещеев", "Никитин", "Кольцов", "Надсон", "Хлебников", "Маяковский",
            "Есенин", "Цветаева", "Мандельштам", "Гумилёв", "Платонов", "Гауф", "Лагерлёф", "Уайльд", "Додэ", "Гюго", "Диккенс"]


def ru_author(title):
    m = re.search(r"\(([^()]*)\)", title.split("/")[0])
    return m.group(1).split(";")[0].strip() if m else None


def ru_docs(only_meta=False):
    import pyarrow.parquet as pq
    for fn in sorted(DL.glob("ruws_*.parquet")):
        f = pq.ParquetFile(fn)
        for g in range(f.metadata.num_row_groups):
            cols = ["title"] if only_meta else ["title", "original_text"]
            for r in f.read_row_group(g, columns=cols).to_pylist():
                yield r


def build_ru(sink, held):
    tier1, other = set(RU_TIER1), set(RU_OTHER)
    # проход 1: метаданные (название, автор, длина) → какие документы берём в рамках лимита
    cand = []
    for r in ru_docs():
        a = ru_author(r["title"])
        if a in tier1 or a in other:
            if len(r["original_text"]) > 300:
                cand.append((0 if a in tier1 else 1, zlib.crc32(r["title"].encode()), r["title"], len(r["original_text"])))
    cand.sort()
    chosen, total = set(), 0
    for _, _, title, n in cand:
        if total >= CAP_RU * 1.15:                       # запас на потери при фильтрах
            break
        chosen.add(title)
        total += n
    print(f"  ru: кандидатов {len(cand)}, выбрано {len(chosen)} документов, ≈{total / 1e6:.0f} млн сырых символов", flush=True)
    # проход 2: запись
    done = 0
    for r in ru_docs():
        t = r["title"]
        if t not in chosen:
            continue
        text = re.sub(r"\[\d+\]|\{\{[^}]*\}\}", "", r["original_text"])
        lines = list(doc_lines(text, "ru"))
        if Heldout.is_held(t):
            held.add(lines)
            continue
        if done < CAP_RU:
            src = "ru: Викитека (ru.wikisource.org), " + ("классика, основной список авторов" if ru_author(t) in tier1 else "прочие прозаики и поэты")
            c = sink.doc(lines, src)
            done += c
            sink.stat["authors|" + str(ru_author(t))] += c
    # TinyStories-Multilingual (ru)
    with open(DL / "dxniz.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["language_code"] == "ru":
                lines = [re.sub(r"\s+", " ", l).strip() for l in d["output"].split("\n") if l.strip()]
                lines = [l for l in lines if line_ok(l, "ru")]
                sink.doc(lines, "ru: TinyStories-Multilingual (ru)")


_UZ_SKIP_TITLE = re.compile(r"^(ЎТИЛ/|Uz pilsētu)|konstitutsi|конституц|qonun|қонун|кодекс|kodeks|farmon|қарор|qaror", re.I)


def clean_wikitext(t):
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<ref[^>]*?/>|<ref.*?</ref>", "", t, flags=re.S)
    t = re.sub(r"\{\|.*?\|\}", "", t, flags=re.S)
    for _ in range(6):
        t = re.sub(r"\{\{[^{}]*\}\}", "", t, flags=re.S)
    t = re.sub(r"\[\[(?:Category|Kategoriya|Fayl|File|Image|Rasm):[^\]]*\]\]", "", t, flags=re.I)
    t = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[\[([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\s+([^\]]*)\]|\[https?://\S+\]", r"\1", t)
    t = re.sub(r"</?(poem|div|span|p|center|small|big|font|u|b|i|sup|sub|nowiki|noinclude|includeonly)[^>]*>", "\n" if False else "", t, flags=re.I)
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
    t = re.sub(r"'{2,}", "", t)
    t = re.sub(r"^=+.*?=+\s*$", "", t, flags=re.M)             # == заголовки ==
    t = re.sub(r"^[*#:;]+\s*", "", t, flags=re.M)
    return t


def uz_docs():
    with open(DL / "uz_wikisource.jsonl", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def to_uz_latin(text):
    cyr = len(re.findall(r"[А-Яа-яЁёЎўҚқҒғҲҳ]", text))
    lat = len(re.findall(r"[A-Za-z]", text))
    if cyr > lat:
        text = translit_uz(text)
    else:
        text = text.translate(_HOMOGLYPH)                                 # кириллические «двойники» в латинском тексте (Аttor → Attor)
    return norm_uz_apostrophes(text.replace("\u00a0", " "))


def build_uz(sink, held):
    for d in uz_docs():
        t = d["title"]
        if _UZ_SKIP_TITLE.search(t) or d["text"].lstrip().lower().startswith("#redirect"):
            continue
        text = to_uz_latin(clean_wikitext(d["text"]))
        lines = list(doc_lines(text, "uz"))
        if Heldout.is_held(t):
            held.add(lines)
            continue
        cyr = bool(re.search(r"[А-Яа-яЎўҚқҒғҲҳ]", d["text"][:3000]))
        sink.doc(lines, "uz: Викитека (wikisource.org, O‘zbek), " + ("кириллица → латиница" if cyr else "латиница"))
    with open(DL / "dxniz.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["language_code"] == "uz":
                lines = [l for l in doc_lines(to_uz_latin(d["output"]), "uz")]
                sink.doc(lines, "uz: TinyStories-Multilingual (uz)")


def build(langs=("uz", "ru", "en")):
    OUT.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)
    stats = json.loads((OUT / "stats.json").read_text(encoding="utf-8")) if (OUT / "stats.json").exists() else {}
    for lang in langs:
        held = Heldout(REP / f"{lang}_heldout.txt")
        sink = Sink(OUT / f"{lang}.txt")
        rest = Sink(OUT / "en_tinystories_rest.txt.gz", gz=True) if lang == "en" else None
        {"uz": lambda: build_uz(sink, held), "ru": lambda: build_ru(sink, held),
         "en": lambda: build_en(sink, rest, held)}[lang]()
        sink.close()
        held.save()
        st = dict(sink.stat)
        if rest:
            rest.close()
            st.update(rest.stat)
        stats[lang] = {"sources": {}, "heldout": {"docs": held.docs, "chars": held.chars}}
        for k, v in st.items():
            s, m = k.split("|")
            if s == "authors":
                stats[lang].setdefault("authors", {})[m] = v
                continue
            stats[lang]["sources"].setdefault(s, {})[m] = v
        print(lang, json.dumps(stats[lang], ensure_ascii=False), flush=True)
    (OUT / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")


# ───────────────────────────── отчёт ─────────────────────────────
LICENSES = {
    "en: TinyStories V2 train": ("https://huggingface.co/datasets/roneneldan/TinyStories", "CDLA-Sharing-1.0 (синтетические рассказы GPT-4, Eldan & Li, Microsoft)"),
    "en: Project Gutenberg (детская/сказочная классика)": ("https://www.gutenberg.org (каталог https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv)", "общественное достояние (США); взяты только авторы, умершие до 1956 г."),
    "en: TinyStories V2 train (rest)": ("https://huggingface.co/datasets/roneneldan/TinyStories (файл en_tinystories_rest.txt.gz)", "CDLA-Sharing-1.0; остальные ≈98% рассказов, не вошедшие в en.txt"),
    "ru: Викитека (ru.wikisource.org), классика, основной список авторов": ("https://huggingface.co/datasets/kristaller486/wikisource-creative-ru ← https://ru.wikisource.org", "Apache-2.0 (набор); тексты — общественное достояние / свободные, компиляция Викитеки CC BY-SA 4.0; авторы умерли до 1956 г."),
    "ru: Викитека (ru.wikisource.org), прочие прозаики и поэты": ("https://huggingface.co/datasets/kristaller486/wikisource-creative-ru ← https://ru.wikisource.org", "то же"),
    "ru: TinyStories-Multilingual (ru)": ("https://huggingface.co/datasets/Dxniz/TinyStories-Multilingual", "Apache-2.0 (машинный перевод TinyStories, исходник CDLA-Sharing-1.0)"),
    "uz: Викитека (wikisource.org, O‘zbek), латиница": ("https://wikisource.org/wiki/Category:O‘zbek (MediaWiki API)", "тексты — общественное достояние (шаблоны PD-old и др.), компиляция CC BY-SA 4.0"),
    "uz: Викитека (wikisource.org, O‘zbek), кириллица → латиница": ("https://wikisource.org/wiki/Category:O‘zbek (MediaWiki API)", "то же; транслитерация нашей функцией translit_uz"),
    "uz: TinyStories-Multilingual (uz)": ("https://huggingface.co/datasets/Dxniz/TinyStories-Multilingual", "Apache-2.0 (машинный перевод, исходник CDLA-Sharing-1.0)"),
}


def report():
    stats = json.loads((OUT / "stats.json").read_text(encoding="utf-8"))
    rows, src_rows = [], []
    for lang in ("uz", "ru", "en"):
        s = stats[lang]["sources"]
        for name in sorted(s):
            m = s[name]
            url, lic = LICENSES.get(name, ("", ""))
            src_rows.append(f"| {lang} | {name} | {m['docs']:,} | {m['lines']:,} | {m['chars']:,} | {url} | {lic} |")
        main = {k: v for k, v in s.items() if not k.endswith("(rest)")}
        rows.append(f"| {lang} | `{lang}.txt` | {sum(v['docs'] for v in main.values()):,} | {sum(v['lines'] for v in main.values()):,} | {sum(v['chars'] for v in main.values()):,} |")
        for k, v in s.items():
            if k.endswith("(rest)"):
                rows.append(f"| {lang} | `en_tinystories_rest.txt.gz` (остальной TinyStories) | {v['docs']:,} | {v['lines']:,} | {v['chars']:,} |")
    out = ["# Литература для этапа «чтение» (набор A)", "",
           "Сделано `tools/make_literature.py` (fetch → build → report). Формат: одна строка = абзац; стих = строфа, переносы заменены на « / »; "
           "пустая строка = граница документа. uz — латиница с ASCII-апострофом. Сами файлы в git НЕ лежат (см. «Где лежит»).", "",
           "## Итог", "", "| язык | файл | документов | строк | символов |", "|---|---|---|---|---|", *rows, "",
           "## Источники и лицензии", "", "| язык | источник | документов | строк | символов | ссылка | лицензия |", "|---|---|---|---|---|---|---|", *src_rows, "",
           "## Главные авторы (символов в обучении)", "",
           *[f"- **{l}**: " + ", ".join(f"{a} {c / 1e6:.1f}M" for a, c in sorted(stats[l].get("authors", {}).items(), key=lambda x: -x[1])[:30])
             for l in ("ru", "en") if stats[l].get("authors")], "",
           "## Проверочные наборы (в git, < 1 МБ)", "",
           "Документы с `crc32(id) % 20 == 0` целиком вынесены из обучения: " + ", ".join(
               f"`data/literature/{l}_heldout.txt` ({stats[l]['heldout']['docs']} док., {stats[l]['heldout']['chars']:,} симв.)" for l in ("uz", "ru", "en")) + ".", "",
           "## Случайные примеры (по 5 строк на язык)", ""]
    rnd = random.Random(7)
    for lang in ("uz", "ru", "en"):
        out.append(f"**{lang}**")
        picked = []
        with open(OUT / f"{lang}.txt", encoding="utf-8") as f:                 # резервуарная выборка (поток)
            n = 0
            for line in f:
                line = line.strip()
                if len(line) < 40:
                    continue
                n += 1
                if len(picked) < 5:
                    picked.append(line)
                elif rnd.random() < 5 / n:
                    picked[rnd.randrange(5)] = line
        out += [f"- {p[:300]}" for p in picked] + [""]
    (REP / "REPORT.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    (REP / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print("report ok")


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(tuple(sys.argv[2:]) or ("uz", "ru", "en"))          # build [uz] [ru] [en] — пересобрать только нужные языки
    else:
        {"fetch": fetch, "report": report}[sys.argv[1]]()
