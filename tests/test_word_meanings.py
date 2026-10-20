"""Значения слов: экзаменационные слова не утекают в учёбу (tools/make_word_meanings.py)."""
import json
import re
import zlib
from collections import defaultdict
from pathlib import Path

import pytest

from tools import make_word_meanings as mw

ROOT = Path(__file__).resolve().parent.parent
WORDS = ROOT / "data/words"
FULL = Path("/home/hermes/mini-ai-raw/words/out")       # полные файлы (на сервере только во время сборки)


def tiny_lex():
    lex, fm = {}, defaultdict(set)
    data = {
        "katta": ("adj", "hajmi ko'p, ulkan narsa.", ["ulkan"], ["kichik"], "Bizning uyimiz juda katta va chiroyli edi."),
        "kichik": ("adj", "hajmi oz, mitti narsa.", ["mitti"], ["katta"], "Kichik bola bog'da o'ynab yurgan edi."),
        "ulkan": ("adj", "juda katta narsa.", [], [], "Ulkan tog'lar osmonga qarab cho'zilib turar edi."),
        "mitti": ("adj", "juda kichik narsa.", [], [], "Mitti qush shoxda sayrab o'tirgan edi."),
        "uy": ("noun", "odam yashaydigan bino.", [], [], "Biz yangi uyga ko'chib o'tdik va xursand bo'ldik."),
        "bog'": ("noun", "daraxtlar o'sadigan joy.", [], [], "Ertalab bog'ga borib gullarni suv bilan sug'ordik."),
        "tog'": ("noun", "yerning baland qismi.", [], [], "Tog'ga chiqqanimizda havo juda toza edi."),
        "qush": ("noun", "qanotli jonzot.", [], [], "Daraxt ustida bir qush sayrab o'tirardi."),
        "yurmoq": ("verb", "oyoq bilan harakat qilmoq.", [], [], "Biz kechqurun daryo bo'yida sekin yurdik."),
        "o'ynamoq": ("verb", "o'yin bilan mashg'ul bo'lmoq.", [], [], "Bolalar hovlida to'p o'ynab yurishdi."),
        "ko'rmoq": ("verb", "ko'z bilan idrok qilmoq.", [], [], "Men kecha do'stimni bozorda ko'rdim."),
        "sekin": ("adv", "shoshmasdan, asta.", [], [], "Toshbaqa yo'ldan juda sekin o'tib ketdi."),
        "tez": ("adv", "shoshilib, chaqqon.", [], ["sekin"], "Ot dala bo'ylab tez chopib ketdi."),
    }
    for w, (p, g, syn, ant, ex) in data.items():
        r = mw.new_rec(w)
        r["pos"].add(p)
        r["senses"].append({"gloss": g, "pos": p})
        r["syn"], r["ant"] = syn, ant
        r["ex"].append([ex, None])
        lex[w] = r
    return lex, fm


def test_exam_rule_is_crc32():
    for w in ["katta", "kitob", "собака", "dog"]:
        assert mw.is_exam(w) == (zlib.crc32(w.encode()) % 10 == 0)
    assert mw.key("uz", "Oʻgʻil") == "o'g'il" and mw.key("ru", "Ёж") == "еж"


def test_no_leak_tiny(monkeypatch):
    lex, fm = tiny_lex()
    exam = {"katta", "qush"}
    monkeypatch.setattr(mw, "is_exam", lambda k: k in exam)
    freq = {k: 100 - i for i, k in enumerate(lex)}
    sf = {t: 5 for k in lex for t in mw.TOK["uz"].findall(" ".join([k] + [s["gloss"] for s in lex[k]["senses"]]
                                                                     + [e[0] for e in lex[k]["ex"]]).lower())}
    lem = mw.Lemmatizer("uz", lex, fm, sf)
    heads, ex, rows, stats = mw.build("uz", lex, freq, 100, lem=lem)
    assert ex == exam
    assert rows["exam"], "экзамен не пустой"
    assert {r["word"] for r in rows["exam"]} == exam
    for r in rows["train"]:
        toks = lem.lemmas_of_text(r["q"] + " " + r["a"])
        assert not toks & exam, r
    # антоним katta↔kichik: «kichik»-задания не должны упоминать экзаменационное «katta»
    assert not any(r["kind"] == "antonym" and r["word"] == "kichik" for r in rows["train"])


def test_blank_and_clean_gloss():
    s = "Bizning uyimiz juda katta va chiroyli edi."
    b = mw.blank("uz", [s, [20, 25]], "katta")
    assert b and b[0] == "Bizning uyimiz juda ___ va chiroyli edi." and b[1] == "katta"
    assert mw.clean_gloss("ru", "устар. старинное слово для обозначения чего-то") == ""
    assert mw.clean_gloss("ru", "домашнее животное семейства псовых").startswith("домашнее")


def _files(lang):
    return [WORDS / f"exam_{lang}.jsonl"]


@pytest.mark.parametrize("lang", mw.LANGS)
def test_git_exam_files(lang):
    f = WORDS / f"exam_{lang}.jsonl"
    words = WORDS / f"exam_words_{lang}.txt"
    if not f.exists():
        pytest.skip("набор ещё не собран")
    assert f.stat().st_size < 1_000_000
    hidden = {mw.key(lang, w) for w in words.read_text(encoding="utf-8").split("\n") if w}
    assert hidden and all(mw.is_exam(k) for k in hidden)
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        assert mw.key(lang, r["word"]) in hidden


@pytest.mark.parametrize("lang", mw.LANGS)
def test_full_train_has_no_exam_words(lang):
    """Полная проверка по собранным файлам (есть на сервере во время сборки, иначе — пропуск)."""
    tr = FULL / f"words_{lang}_train.jsonl"
    words = WORDS / f"exam_words_{lang}.txt"
    if not tr.exists() or not words.exists():
        pytest.skip("полные файлы на Kaggle/Release, локально нет")
    hidden = {mw.key(lang, w) for w in words.read_text(encoding="utf-8").split("\n") if w}
    tok = mw.TOK[lang]
    n = 0
    for line in open(tr, encoding="utf-8"):
        r = json.loads(line)
        assert mw.key(lang, r["word"]) not in hidden
        text = mw.key(lang, r["q"] + " " + r["a"])
        bad = set(tok.findall(text)) & hidden
        assert not bad, (bad, r)
        n += 1
    assert n > 1000
