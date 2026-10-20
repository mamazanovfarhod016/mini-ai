#!/usr/bin/env python3
"""
Этап «слова»: значения слов для Mini AI на uz/ru/en из Викисловаря (CC BY-SA).

    .venv/bin/python tools/make_word_meanings.py --raw /home/hermes/mini-ai-raw/words --out /home/hermes/mini-ai-raw/words/out

Источники (все CC BY-SA, как Викисловарь; извлечение — wiktextract / kaikki.org):
  en  kaikki.org, English Wiktionary → English          (определения, синонимы, антонимы, примеры)
  ru  kaikki.org, Russian Wiktionary (ru.wiktionary) → Русский
  uz  uz.wiktionary.org (дамп pages-articles, определения на узбекском, примеры {{misol}})
      + kaikki.org, English Wiktionary → Uzbek (перевод на английский, если узбекского определения нет; формы слов)

Что делаем (всё скриптом, без ИИ):
  1. Частоты слов — по корпусу «чтение» (data/literature/*.txt, если есть, + data/corpus/*.txt), формы → лемма
     по таблицам форм Викисловаря (ru/en/uz) и отсечению аффиксов (uz).
  2. Для топ-N частых лемм (сущ., глагол, прил., нареч.): определение, синонимы, антонимы, пример.
  3. Задания q→a на языке слова, по 3 формулировки: meaning / synonym / antonym / odd (лишнее слово —
     другая часть речи, части речи берутся из Викисловаря; берём только слова с ОДНОЙ частью речи) / fill (вставь слово).
  4. Экзамен: слово целиком спрятано, если crc32(ключ слова) % 10 == 0 (ключ = нижний регистр, uz-апостроф ', ru ё→е).
     В учёбу не попадает ни одно задание, где экзаменационное слово стоит хоть где-то: в вопросе (в т.ч. в примере
     и вариантах) и в ответе (определение тоже) — проверка по леммам всех слов строки.
Читаем всё потоком: в памяти только компактные записи нужных слов.
"""
import argparse
import bz2
import html
import json
import random
import re
import sys
import time
import zlib
from collections import Counter, defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent.parent
LANGS = ("uz", "ru", "en")
POS_OK = ("noun", "verb", "adj", "adv")
APOS = str.maketrans({c: "'" for c in "ʻʼ‘’`′´ʹ"})
FILES = {
    "en": "en_en.jsonl",             # https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl
    "ru": "ru_ru.jsonl",             # https://kaikki.org/ruwiktionary/Русский/kaikki.org-dictionary-Русский.jsonl
    "uz_k": "en_uz.jsonl",           # https://kaikki.org/dictionary/Uzbek/kaikki.org-dictionary-Uzbek.jsonl
    "uz_w": "uzwikt.xml.bz2",        # https://dumps.wikimedia.org/uzwiktionary/latest/uzwiktionary-latest-pages-articles.xml.bz2
}
SOURCES = {
    "en": [("kaikki.org — English Wiktionary, English (wiktextract)",
            "https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl", "CC BY-SA 4.0 / GFDL")],
    "ru": [("kaikki.org — Russian Wiktionary (ru.wiktionary), Русский (wiktextract)",
            "https://kaikki.org/ruwiktionary/Русский/kaikki.org-dictionary-Русский.jsonl", "CC BY-SA 4.0 / GFDL")],
    "uz": [("uz.wiktionary.org — дамп pages-articles (определения и примеры на узбекском)",
            "https://dumps.wikimedia.org/uzwiktionary/latest/uzwiktionary-latest-pages-articles.xml.bz2", "CC BY-SA 4.0 / GFDL"),
           ("kaikki.org — English Wiktionary, Uzbek (переводы, формы, синонимы)",
            "https://kaikki.org/dictionary/Uzbek/kaikki.org-dictionary-Uzbek.jsonl", "CC BY-SA 4.0 / GFDL")],
}
TOK = {
    "uz": re.compile(r"[a-z](?:[a-z]|'(?=[a-z])|(?<=[og])')*"),
    "ru": re.compile(r"[а-яё]+(?:-[а-яё]+)*"),
    "en": re.compile(r"[a-z]+"),
}
LETTERS = {"uz": re.compile(r"^[A-Za-z' ,.;:!?\-—–()«»\"0-9]+$"),
           "ru": re.compile(r"^[А-Яа-яЁё ,.;:!?\-—–()«»\"0-9]+$"),
           "en": re.compile(r"^[A-Za-z' ,.;:!?\-—–()\"0-9]+$")}
POS_NAME = {
    "uz": {"noun": ("ot", "ot"), "verb": ("fe'l", "fe'l"), "adj": ("sifat", "sifat"), "adv": ("ravish", "ravish")},
    "ru": {"noun": ("существительное", "существительные"), "verb": ("глагол", "глаголы"),
           "adj": ("прилагательное", "прилагательные"), "adv": ("наречие", "наречия")},
    "en": {"noun": ("noun", "nouns"), "verb": ("verb", "verbs"), "adj": ("adjective", "adjectives"),
           "adv": ("adverb", "adverbs")},
}
Q = {
    "meaning": {"uz": ["{w} so'zi nimani anglatadi?", "{w} so'zining ma'nosi nima?", "{w} degani nima?"],
                "ru": ["что значит слово «{w}»?", "что означает «{w}»?", "объясни слово «{w}»"],
                "en": ['what does the word "{w}" mean?', 'what is the meaning of "{w}"?', 'explain the word "{w}"']},
    "synonym": {"uz": ["{w} so'ziga sinonim ayt", "{w} so'zining sinonimi qanday?", "{w} so'ziga ma'nodosh so'z ayt"],
                "ru": ["синоним к слову «{w}»", "подбери синоним к «{w}»", "какое слово близко по смыслу к «{w}»?"],
                "en": ['a synonym for "{w}"', 'give a synonym of "{w}"', 'what word means the same as "{w}"?']},
    "antonym": {"uz": ["{w} so'zining antonimi qanday?", "{w} so'ziga qarama-qarshi ma'noli so'z ayt",
                       "{w} so'zining teskarisi nima?"],
                "ru": ["антоним к слову «{w}»", "противоположное по смыслу к «{w}»", "какое слово обратно по смыслу «{w}»?"],
                "en": ['the opposite of "{w}"', 'give an antonym of "{w}"', 'what word means the opposite of "{w}"?']},
    "odd": {"uz": ["ortiqcha so'zni top: {l}", "qaysi so'z ortiqcha: {l}?", "{l} — bu so'zlardan qaysi biri ortiqcha?"],
            "ru": ["найди лишнее слово: {l}", "какое слово лишнее: {l}?", "лишнее слово в ряду: {l}"],
            "en": ["find the odd word: {l}", "which word does not belong: {l}?", "odd one out: {l}"]},
    "fill": {"uz": ["gapga mos so'zni qo'y ({o}): {s}", "bo'sh joyga qaysi so'z mos ({o})? {s}", "so'zni qo'yib gapni yoz ({o}): {s}"],
             "ru": ["вставь слово в предложение ({o}): {s}", "какое слово подходит ({o})? {s}", "заполни пропуск ({o}): {s}"],
             "en": ["fill in the blank ({o}): {s}", "which word fits ({o})? {s}", "complete the sentence ({o}): {s}"]},
}
A = {
    "meaning": {"uz": "{w} — {d}", "ru": "«{w}» — {d}", "en": 'The word "{w}" means: {d}'},
    "meaning_tr": {"uz": "{w} — inglizcha «{d}» degani."},
    "synonym": {"uz": "{w} so'zining sinonimi — {s}.", "ru": "Синоним к слову «{w}» — {s}.", "en": 'A synonym for "{w}" is {s}.'},
    "antonym": {"uz": "{w} so'zining antonimi — {s}.", "ru": "Антоним к слову «{w}» — {s}.", "en": 'The opposite of "{w}" is {s}.'},
    "odd": {"uz": "Ortiqcha so'z — {x}: u {p}, qolganlari — {pp}.", "ru": "Лишнее слово — «{x}»: это {p}, а остальные — {pp}.",
            "en": 'The odd word is "{x}": it is {art} {p}, the others are {pp}.'},
}
SYN_JOIN = {"uz": lambda xs: ", ".join(xs), "ru": lambda xs: ", ".join(f"«{x}»" for x in xs),
            "en": lambda xs: " or ".join(f'"{x}"' for x in xs)}
BAD_EN_TAGS = {"archaic", "obsolete", "dialectal", "rare", "slang", "vulgar", "offensive", "derogatory", "nonstandard",
               "historical", "dated", "Internet", "euphemistic", "misspelling", "abbreviation", "initialism", "acronym",
               "form-of", "alt-of", "obsolete", "ethnic", "slur", "uncommon", "Scotland", "Northern-England", "nonce-word"}
RU_BAD_LABEL = {"устар", "жарг", "груб", "вульг", "бранн", "диал", "прост", "обсц", "сленг", "арх", "истор", "эвф",
                "мол", "крим", "презр", "пренебр", "рег", "старин", "церк", "неодобр", "сниж", "редк", "поэт", "трад.-поэт"}
# местоимения/служебные — у Викисловаря часто помечены как «ot»/noun; в «лишнее слово» не берём
FUNCTION_WORDS = {
    "uz": set("""nima kim qaysi qanday qancha nega qayer qachon bu shu u o'sha men sen biz siz ular hamma har o'z ana mana
 hech barcha bari ba'zi allaqanday nimadir kimdir""".split()),
    "ru": set("""кто что какой который чей сколько где когда это то этот тот себя сам весь всё все каждый любой другой
 никто ничто некто нечто такой таков можно нельзя надо нужно ним него нее""".split()),
    "en": set("""who what which whose this that these those all each every other another any some none something
 nothing anything everything someone anyone everyone nobody one it used than up can may must will shall""".split()),
}
# части речи служебных слов: если слово бывает и таким — не берём (это грамматика, не «значение»)
FUNC_POS = {"pron", "det", "prep", "conj", "particle", "article", "num", "intj", "postp", "contraction", "aux",
            "character", "symbol", "phrase", "prep_phrase"}
UZ_SUFFIXES = sorted(set("""lar larni larning larga larda lardan larim laring lari larimiz laringiz ning ni ga ka qa da ta dan tan
 dagi dagini dek day cha lik ligi ligini likni likka likda likdan li siz chi chilar im ing i si imiz ingiz miz ngiz m ng
 man san di dim ding dik dingiz dilar ydi adi aman asan amiz asiz adilar gan gani ganda ganini gach ib ish ishi ishni ishga
 yapti yapman moqda moqchi sa sam sang sak may ma mas mi roq ib ar ay aylik ngan kan qan uvchi ovchi vchi la lan lash
 ingni imni ini ida idan iga mizni gina mi chi""".split()), key=len, reverse=True)


def norm(lang, w):
    w = w.replace("\u0301", "").replace("\u0300", "").strip()
    return w.translate(APOS) if lang == "uz" else w


def key(lang, w):
    k = norm(lang, w).lower()
    return k.replace("ё", "е") if lang == "ru" else k


def is_exam(k):
    return zlib.crc32(k.encode("utf-8")) % 10 == 0


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- частоты по корпусу «чтение»
def corpus_files(lang, freq_dir):
    fs = []
    for d in [freq_dir, ROOT / "data/literature"]:
        if d and Path(d).exists():
            fs += sorted(Path(d).glob(f"{lang}*.txt"))
    fs += [p for p in [ROOT / f"data/corpus/{lang}.txt", ROOT / f"data/corpus/{lang}_test.txt"] if p.exists()]
    return list(dict.fromkeys(fs))


def iter_extra(lang, path, cap):
    """Сырые тексты стадии «чтение» (пока data/literature не собран): .txt, .jsonl, .parquet — потоком, до cap символов."""
    p, got = Path(path), 0
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq
        f = pq.ParquetFile(p)
        col = next(c for c in ("original_text", "text", "output") if c in f.schema_arrow.names)
        for b in f.iter_batches(batch_size=64, columns=[col]):
            for t in b.column(0).to_pylist():
                if t:
                    got += len(t); yield t
                if got > cap:
                    return
        return
    with open(p, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if p.suffix == ".jsonl":
                d = json.loads(line)
                lc = d.get("language_code") or d.get("lang")
                if lc and lc != lang:
                    continue
                line = d.get("text") or d.get("output") or d.get("original_text") or ""
            got += len(line)
            yield line
            if got > cap:
                return


def surface_freq(lang, files, extra=(), cap=300_000_000):
    c = Counter()
    def lines():
        for f in files:
            with open(f, encoding="utf-8") as fh:
                yield from fh
        for f in extra:
            yield from iter_extra(lang, f, cap)
    for line in lines():
        line = norm(lang, line).lower()
        c.update(TOK[lang].findall(line.replace("ё", "е") if lang == "ru" else line))
    return c


# ---------------------------------------------------------------- чистка текста
def clean_gloss(lang, g):
    g = html.unescape(re.sub(r"<[^>]+>", " ", g or ""))
    g = norm(lang, g)
    g = re.sub(r"\s+", " ", g).strip(" .;:")
    if lang == "ru":
        m = re.match(r"^((?:[а-яё\-]+\.\s*(?:,\s*)?)+)(?:·\s*)?", g)
        if m:
            labels = re.findall(r"([а-яё\-]+)\.", m.group(1))
            if any(l in RU_BAD_LABEL for l in labels):
                return ""
            g = g[m.end():].strip()
        g = re.sub(r"^(?:наречие|действие|свойство|состояние) (?:к|по значению)[^;]*;\s*", "", g)
        g = g.replace(" ,", ",").replace(" .", ".")
    g = re.sub(r"^(?:\d+\s*[.)]?\s*)+", "", g)
    g = re.split(r";\s", g)[0].strip(" .,;:")
    if re.search(r"\d|\b(?:fl|ort|q|qar|sm|см|ср)\.|\s-[a-zа-я]|[a-zа-я]-\s", g):
        return ""
    if not g or len(g) > 140 or not LETTERS[lang].match(g) or len(g.split()) < 2 and lang != "en":
        return ""
    if lang == "en" and len(g.split()) < 1:
        return ""
    return g[0].lower() + g[1:] + "." if lang != "en" else g[0].lower() + g[1:] + "."


def ok_sentence(lang, s):
    s = re.sub(r"\s+", " ", s).strip()
    n = len(s.split())
    return 4 <= n <= 16 and LETTERS[lang].match(s) and ".." not in s and "//" not in s and s[0].isupper()


# ---------------------------------------------------------------- словари → компактные записи
def new_rec(w):
    return {"w": w, "pos": set(), "senses": [], "tr": [], "syn": [], "ant": [], "ex": [], "forms": set()}


def add_sense(rec, gloss, pos):
    if gloss and len(rec["senses"]) < 5 and gloss not in [s["gloss"] for s in rec["senses"]]:
        rec["senses"].append({"gloss": gloss, "pos": pos})


def read_kaikki(lang, path, vocab, lex, formmap, src_lang):
    """lang = язык слов; path — JSONL wiktextract. Оставляем записи, если слово или его форма есть в корпусе."""
    n = kept = 0
    marker = f'"lang_code": "{src_lang}"'
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if marker not in line:
                continue
            n += 1
            d = json.loads(line)
            if d.get("lang_code") != src_lang:
                continue
            w = norm(lang, d.get("word", ""))
            if not w or (lang != "uz" and w != w.lower()) or " " in w or re.search(r"[а-яё]", w.lower()) and lang != "ru":
                continue
            k = key(lang, w)
            pos = d.get("pos", "")
            forms = {key(lang, f["form"]) for f in d.get("forms", []) if f.get("form")
                     and not set(f.get("tags", [])) & {"table-tags", "inflection-template", "romanization"}
                     and " " not in f["form"]}
            senses = d.get("senses", [])
            lemma_like = False
            for s in senses:
                fo = s.get("form_of") or (s.get("alt_of") if lang != "uz" else None)
                if fo or "form-of" in s.get("tags", []):
                    for t in fo or []:
                        if t.get("word"):
                            formmap[k].add(key(lang, t["word"]))
                else:
                    lemma_like = True
            if not lemma_like:
                continue
            if lang != "uz" and k not in vocab and not (forms & vocab):
                continue
            rec = lex.get(k) or lex.setdefault(k, new_rec(w))
            rec["pos"].add(pos)
            for f in forms:
                if lang == "uz" or f in vocab:
                    rec["forms"].add(f)
                    formmap[f].add(k)
            if pos not in POS_OK:
                continue
            kept += 1
            top_syn = [(x["word"], x.get("sense_index")) for x in d.get("synonyms", []) if not x.get("tags")]
            top_ant = [(x["word"], x.get("sense_index")) for x in d.get("antonyms", []) if not x.get("tags")]
            good_i = 0
            for i, s in enumerate(senses, 1):
                tags = set(s.get("tags", []))
                if s.get("form_of") or s.get("alt_of") or tags & BAD_EN_TAGS:
                    continue
                gl = (s.get("glosses") or [""])[-1]
                if src_lang == "uz":     # английский перевод узбекского слова
                    gl = re.sub(r"\s*\([^)]*\)", "", gl).split(";")[0].strip(" .")
                    if gl and len(gl) <= 60 and len(rec["tr"]) < 2 and re.match(r"^[A-Za-z ,'\-]+$", gl):
                        rec["tr"].append(gl)
                else:
                    g = clean_gloss(lang, gl)
                    if not g:
                        continue
                    add_sense(rec, g, pos)
                good_i += 1
                # синонимы/антонимы — только к первому значению и только той части речи, что у главного определения
                main_pos = rec["senses"][0]["pos"] if rec["senses"] else pos
                first = good_i == 1 and main_pos == pos
                for x in s.get("synonyms", []):
                    if first and not x.get("tags"):
                        rec["syn"].append(norm(lang, x["word"]))
                for x in s.get("antonyms", []):
                    if first and not x.get("tags"):
                        rec["ant"].append(norm(lang, x["word"]))
                for w2, si in top_syn:
                    if si == i and first:
                        rec["syn"].append(norm(lang, w2))
                for w2, si in top_ant:
                    if si == i and first:
                        rec["ant"].append(norm(lang, w2))
                for e in s.get("examples", []):
                    if len(rec["ex"]) >= 6 or (lang == "en" and e.get("type") != "example"):
                        continue
                    t = norm(lang, e.get("text", "")).strip()
                    if ok_sentence(lang, t):
                        span = (e.get("bold_text_offsets") or [None])[0]
                        rec["ex"].append([t, list(span) if span else None])
    log(lang, src_lang, "entries", n, "kept POS-entries", kept)


def misol_parts(x):
    """{{misol|текст с {{ajrat|слово}}|автор|книга}} → [текст с ⟦слово⟧]."""
    out = []
    for m in re.finditer(r"\{\{misol\|", x):
        i, depth, j = m.end(), 1, m.end()
        while j < len(x) and depth:
            if x.startswith("{{", j):
                depth += 1; j += 2
            elif x.startswith("}}", j):
                depth -= 1; j += 2
            else:
                j += 1
        body = x[i:j - 2]
        body = re.sub(r"\{\{ajrat\|([^{}|]*)\}\}", lambda mm: "⟦" + mm.group(1).strip() + "⟧", body)
        out.append(body.split("|")[0])
    return out


def strip_wiki(x):
    x = html.unescape(x)
    for _ in range(3):
        x = re.sub(r"\{\{[^{}]*\}\}", "", x)
    x = re.sub(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]", r"\1", x)
    x = re.sub(r"<[^>]+>|'''|''", "", x)
    return re.sub(r"\s+", " ", x).strip()


def section(txt, name):
    m = re.search(r"===\s*" + name + r"\s*===\n(.*?)(?=\n==|\Z)", txt, re.S)
    return m.group(1) if m else ""


def read_uzwikt(path, lex, formmap):
    n = 0
    for _, el in ET.iterparse(bz2.open(path), events=("end",)):
        if el.tag.rsplit("}", 1)[-1] != "page":
            continue
        if el.find("{*}ns").text == "0":
            title = norm("uz", el.find("{*}title").text or "")
            txt = el.find("{*}revision/{*}text").text or ""
            if "{{-uz-}}" in txt and re.match(r"^[a-zA-Z']+(-[a-zA-Z']+)?$", title):
                txt = norm("uz", re.split(r"\n=?\s*\{\{-(?!uz-)[a-z-]+-\}\}", txt.split("{{-uz-}}", 1)[1])[0])
                k = key("uz", title)
                rec = lex.get(k) or lex.setdefault(k, new_rec(title))
                for t, p in (("uz-ism", "noun"), ("uz-adj", "adj"), ("uz-verb", "verb"), ("uz-adv", "adv")):
                    if "{{" + t in txt:
                        rec["pos"].add(p)
                if re.search(r"m[oa]q$", k):
                    rec["pos"].add("verb")
                for line in section(txt, "Maʼnosi|Ma'nosi").split("\n"):
                    line = re.sub(r"^[#:*\s]*(?:'''\d+'''\s*)?", "", line)
                    if not line or line.startswith(("q.", "''q.''")):
                        continue
                    for ex in misol_parts(line):
                        s = strip_wiki(ex.replace("⟦", "\x01").replace("⟧", "\x02"))
                        m = re.search("\x01([^\x02]*)\x02", s)
                        plain = s.replace("\x01", "").replace("\x02", "")
                        if m and ok_sentence("uz", plain) and len(rec["ex"]) < 6:
                            st = m.start()
                            rec["ex"].append([plain, [st, st + len(m.group(1))]])
                    g = strip_wiki(re.sub(r"\{\{misol\|.*", "", line))
                    g = re.sub(r"^(?:ko'chma|s\. t\.|shv\.|esk\.|kt\.|din\.|tar\.)\s*", "", g)
                    if g.startswith(("q. ", "Q. ", "ayn.")) or re.search(r"[А-Яа-яЁё]", g):
                        continue
                    g = clean_gloss("uz", g)
                    if g:
                        add_sense(rec, g, None)
                for nm, fld in (("Sinonimlari", "syn"), ("Antonimlari", "ant")):
                    for w2 in re.findall(r"\[\[([^\]|#]+)", section(txt, nm)):
                        if re.match(r"^[a-zA-Z']+$", norm("uz", w2)):
                            rec[fld].append(norm("uz", w2))
                n += 1
        el.clear()
    log("uz wiktionary pages", n)


# ---------------------------------------------------------------- леммы
class Lemmatizer:
    def __init__(self, lang, lex, formmap, vocab=None):
        self.lang, self.lex, self.fm = lang, lex, formmap
        self.vocab = vocab or {}
        self.cache = {}

    def __call__(self, tok):
        if tok in self.cache:
            return self.cache[tok]
        out = set()
        if tok in self.lex and self.lex[tok]["pos"] & set(POS_OK):
            out.add(tok)
        out |= {l for l in self.fm.get(tok, ()) if l in self.lex}
        if not out and self.lang == "uz":
            out = self.uz_strip(tok)
        self.cache[tok] = out
        return out

    def uz_strip(self, tok):
        best, frontier = None, [tok]
        for _ in range(4):
            nxt = []
            for t in frontier:
                for s in UZ_SUFFIXES:
                    if t.endswith(s) and len(t) - len(s) >= 2:
                        st = t[:-len(s)]
                        cands = [st, st + "moq", st + "maq"]
                        if st.endswith("g'"):
                            cands.append(st[:-2] + "q")
                        if st.endswith("g"):
                            cands.append(st[:-1] + "k")
                        if st.endswith("y"):
                            cands += [st[:-1] + "moq", st[:-1] + "maq"]
                        for c in cands:
                            if c in self.lex and self.lex[c]["pos"] & set(POS_OK) and (best is None or len(c) > len(best)):
                                best = c
                        nxt.append(st)
            if best:
                return {best}
            frontier = list(dict.fromkeys(nxt))[:60]
        return set()

    def lemmas_of_text(self, text):
        t = norm(self.lang, text).lower()
        if self.lang == "ru":
            t = t.replace("ё", "е")
        out = set()
        for tok in TOK[self.lang].findall(t):
            out.add(tok)
            out |= self(tok)
        return out

    def clean(self, text):
        """uz: дамп Викисловаря местами из OCR (переносы «ko'pay-moq», «Boiashq»). Пропускаем текст, только если
        каждое слово встречается в корпусе ≥ 2 раз или сводится к лемме словаря."""
        if self.lang != "uz":
            return True
        t = norm("uz", text)
        if re.search(r"[a-z']-[a-z]", t) and not re.search(r"\b(?:" + "|".join(["yer-ko'k", "o'tar-o'tmas"]) + r")\b", t):
            for m in re.finditer(r"([A-Za-z']+)-([a-z']+)", t):
                if m.group(0).lower() not in self.vocab:
                    return False
        if re.search(r"\b(?:ayn|q|n|d|qar|sm|ko'chma)\.|\(|\)", t):
            return False
        for tok in TOK["uz"].findall(t.lower()):
            if len(tok) == 1 and tok not in ("u", "o"):
                return False
            if self.vocab.get(tok, 0) < 2 and not self(tok):
                return False
        return True


# ---------------------------------------------------------------- задания
def blank(lang, ex, w):
    s, span = ex
    if span and 0 <= span[0] < span[1] <= len(s):
        form = s[span[0]:span[1]]
    else:
        m = re.search(r"\b" + re.escape(w) + r"\w*", s, re.I)
        if not m:
            return None
        span, form = [m.start(), m.end()], m.group(0)
    if not form.strip() or " " in form.strip() or len(form) > 25:
        return None
    if len(s.split()) < 5 or re.search(r"[.!?]\s+\S", s.rstrip(".!?\"» ")):      # одно предложение, не список фраз
        return None
    stem = form.lower()[:max(3, len(form) - 3)]
    if s.lower().count(stem) != 1:                                               # слово не подсказано вторым вхождением
        return None
    return s[:span[0]] + "___" + s[span[1]:], form


def match_case(form, w):
    return w[0].upper() + w[1:] if form[:1].isupper() else w


def template_tokens(lang):
    t = " ".join(q for kind in Q.values() for q in kind.get(lang, []))
    t += " " + " ".join(a.get(lang, "") for a in A.values()) + " " + " ".join(" ".join(v) for v in POS_NAME[lang].values())
    t = re.sub(r"\{[a-z]+\}", " ", t)
    return set(TOK[lang].findall(key(lang, t))) | {"an", "a", "the", "is"}


MIN_FREQ = 10          # слово должно встретиться в корпусе «чтение» ≥ 10 раз
MIN_FREQ_SYN = 20      # синоним/антоним в ответе — тоже частое слово, а не редкость Викисловаря


def build(lang, lex, freq_lemma, top, seed=0, strict=True, lem=None, min_freq=MIN_FREQ):
    rnd = random.Random(seed)
    heads = [k for k, _ in sorted(freq_lemma.items(), key=lambda kv: (-kv[1], kv[0]))
             if freq_lemma[k] >= min_freq and k in lex and lex[k]["pos"] & set(POS_OK)
             and not lex[k]["pos"] & FUNC_POS and k not in FUNCTION_WORDS[lang]
             and (lex[k]["senses"] or lex[k]["tr"]) and len(k) >= 2][:top]
    hs = set(heads)
    tmpl = template_tokens(lang)
    heads = [k for k in heads if k not in tmpl]          # слова из шаблонов вопросов/ответов в экзамен не прячем
    exam = {k for k in heads if is_exam(k)}
    known = {x for x, n in freq_lemma.items() if n >= MIN_FREQ_SYN} | hs

    def single_pos(k):
        if k in FUNCTION_WORDS[lang]:
            return None
        p = lex[k]["pos"] - {"", None}
        return next(iter(p)) if len(p) == 1 and next(iter(p)) in POS_OK else None
    by_pos = defaultdict(lambda: {"train": [], "exam": []})
    for k in heads:
        p = single_pos(k)
        if p:
            by_pos[p]["exam" if k in exam else "train"].append(k)
    rows = {"train": [], "exam": []}
    stats = Counter()

    def disp(k):
        return lex[k]["w"]

    def text_exam(text):
        return (lem.lemmas_of_text(text) if strict else set(TOK[lang].findall(key(lang, text)))) & exam

    def leaks(row):
        # утечка = экзаменационное слово (лемма) где угодно в вопросе или ответе, включая «роли» задания (синонимы,
        # варианты «лишнего» и «вставь»). Чтобы не терять слишком много, для учёбы выбираем то значение и тот пример,
        # где спрятанных слов нет (text_exam), а слова из шаблонов вопросов в экзамен не прячем.
        words = set(row["roles"])
        if strict:
            words |= lem.lemmas_of_text(row["q"] + " " + row["a"])
        return words & exam

    for k in heads:
        part = "exam" if k in exam else "train"
        r = random.Random(zlib.crc32(k.encode()) + seed)
        rec, w = lex[k], disp(k)
        out = []
        ok_other = (lambda x: x not in exam) if part == "train" else (lambda x: True)
        glosses = [s["gloss"] for s in rec["senses"] if lem.clean(s["gloss"])]
        if part == "train":           # берём значение, в тексте которого нет спрятанных слов (если такое есть)
            glosses = sorted(glosses, key=lambda g: bool(text_exam(g)))
        if glosses:
            g = glosses[0]
            for q in Q["meaning"][lang]:
                out.append(("meaning", q.format(w=w), A["meaning"][lang].format(w=w, d=g), [k]))
        elif lang == "uz" and rec["tr"]:
            for q in Q["meaning"][lang]:
                out.append(("meaning", q.format(w=w), A["meaning_tr"]["uz"].format(w=w, d=rec["tr"][0]), [k]))
        for kind in ("synonym", "antonym"):
            fld = "syn" if kind == "synonym" else "ant"
            xs = [x for x in dict.fromkeys(rec[fld]) if key(lang, x) != k and key(lang, x) in known
                  and " " not in x and ok_other(key(lang, x))][:2]
            if xs:
                for q in Q[kind][lang]:
                    out.append((kind, q.format(w=w), A[kind][lang].format(w=w, s=SYN_JOIN[lang](xs)),
                                [k] + [key(lang, x) for x in xs]))
        p = single_pos(k)
        pool_part = by_pos  # другие слова: для учёбы — только учебные
        if p:
            same = [x for x in by_pos[p]["train"] if x != k]
            others = [(pp, x) for pp in POS_OK if pp != p for x in by_pos[pp]["train"]]
            if len(same) >= 2 and others:
                pp, odd = r.choice(others)
                group = r.sample(same, 2) + [k]
                if r.random() < 0.3 and part == "train":         # иногда лишнее — само слово
                    pp, odd = p, k
                    p2 = r.choice([x for x in POS_OK if x != p and len(by_pos[x]["train"]) >= 3])
                    group = r.sample(by_pos[p2]["train"], 3)
                    p_main = p2
                else:
                    p_main = p
                items = group + [odd]
                for q in Q["odd"][lang]:
                    r.shuffle(items)
                    lst = ", ".join(disp(x) for x in items)
                    pn, ppn = POS_NAME[lang][pp][0], POS_NAME[lang][p_main][1]
                    art = "an" if pn[0] in "aeiou" else "a"
                    out.append(("odd", q.format(l=lst), A["odd"][lang].format(x=disp(odd), p=pn, pp=ppn, art=art),
                                items[:]))
        exs = sorted(rec["ex"], key=lambda e: bool(text_exam(e[0]))) if part == "train" else rec["ex"]
        n_fill = 0
        for ex in exs:
            if n_fill >= 2:
                break
            if not lem.clean(ex[0]):
                continue
            b = blank(lang, ex, w)
            if not b:
                continue
            sent_blank, form = b
            cand = [(pp, x) for pp in POS_OK if pp not in rec["pos"] for x in by_pos[pp]["train"]]
            if len(cand) < 2:
                continue
            dist_k = [x for _, x in r.sample(cand, 2)]
            dist = [match_case(form, disp(x)) for x in dist_k]
            opts = [form] + dist
            full = ex[0]
            n_fill += 1
            for q in Q["fill"][lang]:
                r.shuffle(opts)
                out.append(("fill", q.format(o=", ".join(opts), s=sent_blank), full,
                            [k] + dist_k))
        for kind, q, a, roles in out:
            row = {"q": q, "a": a, "kind": kind, "lang": lang, "word": w, "roles": roles}
            if part == "train" and leaks(row):
                stats["dropped_leak_" + kind] += 1
                continue
            rows[part].append(row)
            stats[f"{part}_{kind}"] += 1
    for part in rows:
        for row in rows[part]:
            row.pop("roles")
    return heads, exam, rows, stats


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="/home/hermes/mini-ai-raw/words")
    ap.add_argument("--out", default="/home/hermes/mini-ai-raw/words/out")
    ap.add_argument("--freq-dir", default=None, help="папка с корпусом «чтение» {lang}*.txt (по умолчанию data/literature)")
    ap.add_argument("--top", type=int, default=10000)
    ap.add_argument("--langs", default="uz,ru,en")
    ap.add_argument("--exam-git-kb", type=int, default=950)
    ap.add_argument("--extra", action="append", default=[],
                    help="lang=путь — сырые тексты «чтения» для частот (txt/jsonl/parquet), можно несколько раз")
    ap.add_argument("--extra-cap", type=int, default=300_000_000, help="символов на файл --extra")
    a = ap.parse_args()
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gitdir = ROOT / "data/words"
    gitdir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    for lang in a.langs.split(","):
        t0 = time.time()
        files = corpus_files(lang, a.freq_dir)
        extra = [p.split("=", 1)[1] for p in a.extra if p.startswith(lang + "=") and Path(p.split("=", 1)[1]).exists()]
        sf = surface_freq(lang, files, extra, a.extra_cap)
        # память (лимит ~1 ГБ): редкие формы (< 3 раз) не нужны ни для топ-10 тыс., ни для синонимов
        if sum(sf.values()) > 5_000_000:
            sf = Counter({t: n for t, n in sf.items() if n >= 3})
        vocab = set(sf)
        log(lang, "corpus files", [str(f.relative_to(ROOT)) if ROOT in f.parents else str(f) for f in files],
            "extra", extra, "types", len(sf), "tokens", sum(sf.values()))
        files = files + [Path(x) for x in extra]
        lex, formmap = {}, defaultdict(set)
        if lang == "uz":
            read_uzwikt(raw / FILES["uz_w"], lex, formmap)
            read_kaikki("uz", raw / FILES["uz_k"], vocab, lex, formmap, "uz")
        else:
            read_kaikki(lang, raw / FILES[lang], vocab, lex, formmap, lang)
        lem = Lemmatizer(lang, lex, formmap, sf)
        fl = Counter()
        unk = 0
        for tok, n in sf.items():
            ls = lem(tok)
            if not ls:
                unk += n
            for l in ls:
                fl[l] += n
        log(lang, "lex", len(lex), "lemmas with freq", len(fl), "unknown token share",
            round(unk / max(1, sum(sf.values())), 3))
        heads, exam, rows, stats = build(lang, lex, fl, a.top, lem=lem, min_freq=3 if lang == "uz" else MIN_FREQ)
        # полные файлы → Kaggle/Release
        with open(out / f"words_{lang}_train.jsonl", "w", encoding="utf-8") as f:
            for r in rows["train"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(out / f"words_{lang}_exam.jsonl", "w", encoding="utf-8") as f:
            for r in rows["exam"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(out / f"lexicon_{lang}.jsonl", "w", encoding="utf-8") as f:
            for k in heads:
                rec = lex[k]
                f.write(json.dumps({"word": rec["w"], "key": k, "freq": fl[k], "exam": k in exam,
                                    "pos": sorted(p for p in rec["pos"] if p), "senses": rec["senses"], "tr": rec["tr"],
                                    "syn": list(dict.fromkeys(rec["syn"]))[:8], "ant": list(dict.fromkeys(rec["ant"]))[:8],
                                    "examples": rec["ex"]}, ensure_ascii=False) + "\n")
        # маленький экзамен в git: одна формулировка на (слово, тип), лимит по размеру
        seen, small = set(), []
        for r in rows["exam"]:
            kk = (r["word"], r["kind"])
            if kk not in seen:
                seen.add(kk)
                small.append(r)
        random.Random(1).shuffle(small)
        lim, size, keep = a.exam_git_kb * 1024, 0, []
        for r in small:
            s = len((json.dumps(r, ensure_ascii=False) + "\n").encode())
            if size + s > lim:
                break
            size += s
            keep.append(r)
        keep.sort(key=lambda r: (r["word"], r["kind"]))
        with open(gitdir / f"exam_{lang}.jsonl", "w", encoding="utf-8") as f:
            for r in keep:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        (gitdir / f"exam_words_{lang}.txt").write_text("\n".join(sorted(lex[k]["w"] for k in exam)) + "\n", encoding="utf-8")
        chars = {p: sum(len(r["q"]) + len(r["a"]) for r in rows[p]) for p in rows}
        cov = Counter()
        for k in heads:
            rec = lex[k]
            cov["def"] += bool(rec["senses"]); cov["tr_only"] += (not rec["senses"] and bool(rec["tr"]))
            cov["syn"] += bool(rec["syn"]); cov["ant"] += bool(rec["ant"]); cov["ex"] += bool(rec["ex"])
        rnd = random.Random(7)
        summary[lang] = {"corpus_files": [str(f) for f in files], "corpus_tokens": sum(sf.values()),
                         "corpus_types": len(sf), "unknown_token_share": round(unk / max(1, sum(sf.values())), 3),
                         "lexicon_entries": len(lex), "headwords": len(heads), "exam_words": len(exam),
                         "coverage": dict(cov), "rows_train": len(rows["train"]), "rows_exam": len(rows["exam"]),
                         "rows_exam_git": len(keep), "chars": chars, "stats": dict(stats),
                         "examples": rnd.sample(rows["train"], min(5, len(rows["train"]))),
                         "freq_range": [fl[heads[0]], fl[heads[-1]]] if heads else [], "sec": round(time.time() - t0)}
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
        log(lang, json.dumps({k: v for k, v in summary[lang].items() if k != "examples"}, ensure_ascii=False))
        del lex, formmap, lem
    write_report(summary, gitdir)


def write_report(summary, gitdir):
    L = ["# Значения слов (стадия «слова»)", "",
         "Собрано скриптом `tools/make_word_meanings.py` из Викисловаря (CC BY-SA). Задания q→a на языке слова,",
         "по 3 формулировки: **meaning** (что значит), **synonym**, **antonym**, **odd** (лишнее слово — другая часть речи,",
         "части речи по Викисловарю, только слова с одной частью речи), **fill** (вставь слово, 3 варианта ответа).", "",
         "**Экзамен:** слово спрятано целиком, если `crc32(ключ) % 10 == 0` (ключ: нижний регистр, uz-апостроф `'`, ru ё→е).",
         "Учебное задание выбрасывается, если экзаменационная лемма есть ГДЕ УГОДНО в вопросе или ответе (включая",
         "определение, пример, варианты) — проверка по леммам всех слов строки. Тест: `tests/test_word_meanings.py`.", "",
         "| язык | частых лемм (headwords) | из них экзамен | заданий учёба | символов учёба | заданий экзамен (полный / в git) | определение / только перевод / синонимы / антонимы / пример |",
         "|---|---|---|---|---|---|---|"]
    for lang in LANGS:
        s = summary.get(lang)
        if not s:
            continue
        c = s["coverage"]
        L.append(f"| {lang} | {s['headwords']:,} | {s['exam_words']:,} | {s['rows_train']:,} | {s['chars']['train']:,} | "
                 f"{s['rows_exam']:,} / {s['rows_exam_git']:,} | {c.get('def', 0):,} / {c.get('tr_only', 0):,} / "
                 f"{c.get('syn', 0):,} / {c.get('ant', 0):,} / {c.get('ex', 0):,} |")
    L += ["", "## Задания по типам (учёба)", "", "| язык | meaning | synonym | antonym | odd | fill | выброшено (утечка экзамена) |",
          "|---|---|---|---|---|---|---|"]
    for lang in LANGS:
        s = summary.get(lang)
        if not s:
            continue
        st = s["stats"]
        dropped = sum(v for k, v in st.items() if k.startswith("dropped"))
        L.append(f"| {lang} | " + " | ".join(f"{st.get('train_' + k, 0):,}" for k in Q) + f" | {dropped:,} |")
    L += ["", "## Частоты (корпус «чтение»)", ""]
    for lang in LANGS:
        s = summary.get(lang)
        if s:
            L.append(f"- {lang}: {s['corpus_tokens']:,} слов, {s['corpus_types']:,} разных; файлы: "
                     + ", ".join(f"`{Path(f).name}`" for f in s["corpus_files"])
                     + f"; без леммы в словаре: {s['unknown_token_share']:.0%} словоупотреблений; "
                     f"частота самого редкого взятого слова: {s['freq_range'][-1] if s['freq_range'] else '-'}")
    L += ["", "## Источники и лицензии", ""]
    for lang in LANGS:
        for name, url, lic in SOURCES[lang]:
            L.append(f"- {lang}: {name} — {url} — {lic}")
    L += ["", "Лицензия набора: CC BY-SA 4.0 (производное от Викисловаря; авторы — участники Wiktionary, извлечение — "
          "wiktextract, Tatu Ylonen, kaikki.org).", "",
          "## Где лежит", "", "- В git: `data/words/exam_{uz,ru,en}.jsonl` (экзамен, одна формулировка на слово×тип, < 1 МБ), "
          "`data/words/exam_words_{lang}.txt` (список спрятанных слов), этот отчёт.",
          "- Полные файлы (`words_{lang}_train.jsonl`, `words_{lang}_exam.jsonl`, `lexicon_{lang}.jsonl`): см. раздел ниже.", ""]
    up = gitdir / "UPLOADS.md"
    if up.exists():
        L += [up.read_text(encoding="utf-8"), ""]
    L += ["## Примеры (5 случайных учебных на язык)", ""]
    for lang in LANGS:
        s = summary.get(lang)
        if not s:
            continue
        L.append(f"**{lang}**")
        for r in s["examples"]:
            L.append(f"- `{r['kind']}` — Q: {r['q']} → A: {r['a']}")
        L.append("")
    (gitdir / "REPORT.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
