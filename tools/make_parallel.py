"""
Параллельные тексты: ОДИН смысл на трёх языках — чтобы модель связывала языки
(учила на одном — понимала на других).

    .venv/bin/python tools/make_parallel.py     # → data/parallel.json {train, test}

Источник: Tatoeba (tatoeba.org, CC BY 2.0 FR) — короткие бытовые предложения с переводами,
проверенные людьми. Узбекские предложения на кириллице переводим в латиницу скриптом
(однозначная таблица), чтобы в модели был один узбекский алфавит.

Что получаем:
  • пары/тройки переводов (uz–ru, uz–en, ru–en) → задания «переведи» в обе стороны,
    формулировки на языке вопроса: «переведи на узбекский …», «tarjima qil inglizchaga …», «translate to russian …»;
  • 20% предложений целиком спрятаны для экзамена (crc32 по id предложения-источника).
"""
import bz2
import json
import re
import tarfile
import zlib
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache_corpus"
LANGS = {"uzb": "uz", "rus": "ru", "eng": "en"}

CYR2LAT = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "j", "з": "z", "и": "i",
           "й": "y", "к": "k", "л": "l", "м": "m", "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t",
           "у": "u", "ф": "f", "х": "x", "ц": "ts", "ч": "ch", "ш": "sh", "ъ": "'", "ь": "", "э": "e", "ю": "yu",
           "я": "ya", "ў": "o'", "қ": "q", "ғ": "g'", "ҳ": "h"}

ASK = {
    ("uz", "ru"): ["rus tiliga tarjima qil {s}", "ruschaga o'gir {s}"],
    ("uz", "en"): ["ingliz tiliga tarjima qil {s}", "inglizchaga o'gir {s}"],
    ("ru", "uz"): ["переведи на узбекский {s}", "как по-узбекски {s}"],
    ("ru", "en"): ["переведи на английский {s}", "как по-английски {s}"],
    ("en", "uz"): ["translate to uzbek {s}", "how do you say in uzbek {s}"],
    ("en", "ru"): ["translate to russian {s}", "how do you say in russian {s}"],
}


def uz_latin(s):
    if not re.search(r"[а-яёўқғҳА-ЯЁЎҚҒҲ]", s):
        return s.replace("ʻ", "'").replace("‘", "'").replace("’", "'")
    out = []
    for ch in s:
        low = ch.lower()
        t = CYR2LAT.get(low, ch)
        out.append(t.capitalize() if ch != low and t else t)
    return "".join(out).replace("ʻ", "'")


def read_sentences(code):
    out = {}
    with bz2.open(CACHE / f"{code}_sentences.tsv.bz2", "rt", encoding="utf-8") as f:
        for line in f:
            i, _, text = line.rstrip("\n").split("\t", 2)
            out[int(i)] = text
    return out


def main():
    sents = {LANGS[c]: read_sentences(c) for c in LANGS}
    lang_of = {i: l for l, d in sents.items() for i in d}
    links = defaultdict(set)
    with tarfile.open(CACHE / "links.tar.bz2", "r:bz2") as tar:
        f = tar.extractfile([m for m in tar.getmembers() if m.name.endswith(".csv")][0])
        for raw in f:
            a, b = map(int, raw.split(b"\t"))
            if a in lang_of and b in lang_of and lang_of[a] != lang_of[b]:
                links[a].add(b)
                links[b].add(a)
    # группа = узбекское предложение + его переводы (узбекский — самый редкий, от него и строим)
    groups, used = [], set()
    for i, text in sents["uz"].items():
        g = {"uz": uz_latin(text)}
        for j in links.get(i, ()):
            g.setdefault(lang_of[j], sents[lang_of[j]][j])
            for k in links.get(j, ()):              # перевод перевода (uz→en→ru)
                if lang_of[k] != "uz":
                    g.setdefault(lang_of[k], sents[lang_of[k]][k])
        if len(g) >= 2 and g["uz"].lower() not in used:
            used.add(g["uz"].lower())
            groups.append((i, g))
    # пары ru–en из русско-английских связей (их много) — ограничиваем, чтобы узбекский не утонул
    ru_en = []
    for i, text in sents["ru"].items():
        for j in links.get(i, ()):
            if lang_of[j] == "en" and len(text.split()) <= 12:
                ru_en.append((i, {"ru": text, "en": sents["en"][j]}))
                break
    ru_en = sorted(ru_en, key=lambda x: zlib.crc32(str(x[0]).encode()))[: 6 * len(groups) + 3000]

    train, test = [], []
    for sid, g in groups + ru_en:
        bucket = test if zlib.crc32(f"par:{sid}".encode()) % 5 == 0 else train
        for (src, dst), forms in ASK.items():
            if src in g and dst in g:
                form = forms[zlib.crc32(f"{sid}{src}{dst}".encode()) % len(forms)]
                bucket.append({"q": form.format(s=g[src]), "a": g[dst], "kind": f"tr_{src}_{dst}", "lang": src,
                               "group": sid})
    out = ROOT / "data/parallel.json"
    out.write_text(json.dumps({"train": train, "test": test}, ensure_ascii=False), encoding="utf-8")
    full3 = sum(len(g) == 3 for _, g in groups)
    print(f"{out}: групп с узбекским {len(groups)} (все 3 языка: {full3}), ru–en {len(ru_en)}; "
          f"заданий train {len(train)}, test {len(test)}")


if __name__ == "__main__":
    main()
