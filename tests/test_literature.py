"""Тесты для tools/make_literature.py: транслитерация uz, апострофы, фильтры строк, формат стихов."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tools import make_literature as m


def test_translit_basic_words():
    assert m.translit_uz("Ўзбекистон") == "O'zbekiston"
    assert m.translit_uz("қишлоқ") == "qishloq"
    assert m.translit_uz("ғазал") == "g'azal"
    assert m.translit_uz("ҳаёт") == "hayot"
    assert m.translit_uz("шаҳар") == "shahar"
    assert m.translit_uz("чироқ") == "chiroq"
    assert m.translit_uz("маъно") == "ma'no"
    assert m.translit_uz("ёш") == "yosh"
    assert m.translit_uz("йўл") == "yo'l"


def test_translit_ye_and_e():
    assert m.translit_uz("еттi".replace("i", "и")) == "yetti"       # начало слова → ye
    assert m.translit_uz("келди") == "keldi"                          # после согласной → e
    assert m.translit_uz("поезд") == "poyezd"                         # после гласной → ye
    assert m.translit_uz("эшик") == "eshik"
    assert m.translit_uz("цирк") == "tsirk"


def test_translit_case_and_punctuation():
    assert m.translit_uz("Ғафур Ғулом") == "G'afur G'ulom"
    assert m.translit_uz("Шаҳар") == "Shahar"
    assert m.translit_uz("ШАҲАР") == "SHAHAR"
    assert m.translit_uz("Ў") == "O'"
    assert m.translit_uz("Салом, дунё! 2024 — yil.") == "Salom, dunyo! 2024 — yil."


def test_translit_is_ascii_apostrophe_only():
    out = m.translit_uz("Ўзбек ғазали маъно ҳикмати ъ")
    assert not set(out) & set("ʻʼ‘’`")
    assert not any("\u0400" <= c <= "\u04ff" for c in out)         # кириллицы не осталось


def test_norm_apostrophes():
    assert m.norm_uz_apostrophes("O‘zbek, g’ayrat, ma’no, so\u02bbz, ta`lim") == "O'zbek, g'ayrat, ma'no, so'z, ta'lim"
    assert m.norm_uz_apostrophes("tog‘ va bog’") == "tog' va bog'"      # апостроф в конце слова
    assert m.norm_uz_apostrophes("\u02bbe") == "\u02bbe"                     # одиночный знак не в слове не трогаем


def test_to_uz_latin_picks_script():
    assert m.to_uz_latin("Ўзбекистон гўзал юрт") == "O'zbekiston go'zal yurt"
    assert m.to_uz_latin("O‘zbekiston go‘zal yurt") == "O'zbekiston go'zal yurt"


def test_line_filters():
    assert m.line_ok("Once upon a time there was a little girl who lived in a small house.", "en")
    assert not m.line_ok("See http://example.com for more details about this thing here.", "en")
    assert not m.line_ok("CHAPTER ONE AND THE BEGINNING", "en")
    assert not m.line_ok("12 34 56 78 90 12 34 56 78 90 12 34 56 78", "en")
    assert m.line_ok("Жили-были старик со старухой у самого синего моря.", "ru")
    assert not m.line_ok("Once upon a time there was a girl who lived in a house.", "ru")
    assert m.line_ok("Bir bor ekan, bir yo'q ekan, qadim zamonda bir podshoh bo'lgan ekan.", "uz")
    assert not m.line_ok("Жили-были старик со старухой у самого синего моря.", "uz")


def test_verse_stays_one_line_with_slash():
    stanza = "Ты жива еще, моя старушка?\nЖив и я. Привет тебе, привет!\nПусть струится над твоей избушкой\nТот вечерний несказанный свет."
    assert m.block_to_line(stanza, hard_wrapped=False).count(" / ") == 3
    prose = ("It was a bright cold day in April, and the clocks were striking\nthirteen. Winston Smith, his chin nuzzled into his breast in an effort\n"
             "to escape the vile wind, slipped quickly through the glass doors of")
    assert " / " not in m.block_to_line(prose, hard_wrapped=True)


def test_doc_lines_one_paragraph_one_line():
    text = "Первый абзац, в нём достаточно слов для проверки.\n\nВторой абзац тоже длинный и понятный читателю."
    assert len(list(m.doc_lines(text, "ru"))) == 2


def test_heldout_is_deterministic_and_about_5_percent():
    ids = [f"doc{i}" for i in range(4000)]
    held = [m.Heldout.is_held(i) for i in ids]
    assert held == [m.Heldout.is_held(i) for i in ids]
    assert 0.03 < sum(held) / len(ids) < 0.07


def test_uz_lines_are_pure_ascii_latin():
    assert not m.line_ok("Hali ham arabchaning uzun dumlik «ya»sini, Аttorlik rastasining xulyosini bor.".replace("А", "\u0410"), "uz")
    fixed = m.to_uz_latin("Hali ham arabchaning, Аttorlik rastasining xulyosini bor edi o‘sha kuni.")
    assert m.line_ok(fixed, "uz") and "A" in fixed and "\u0410" not in fixed
    assert not m.line_ok("Bu so\u2018z juda uzun va chiroyli edi, deb aytdi u.", "uz")   # неисправленный апостроф не пройдёт


def test_notes_and_footnotes_dropped():
    text = ("Жил-был старик, и было у него три сына, все молодцы.\n\n"
            "3 Речь идет о рукописи, которую автор передал другу в 1836 году.\n\n"
            "Пошёл старший сын в лес и долго не возвращался домой.\n\n"
            "Примечания\n\n"
            "Впервые опубликовано в журнале «Современник» в 1847 году, номер второй.")
    out = list(m.doc_lines(text, "ru"))
    assert out == ["Жил-был старик, и было у него три сына, все молодцы.", "Пошёл старший сын в лес и долго не возвращался домой."]


def test_wikisource_navigation_and_note_headings_dropped():
    nav = "Русская проза / Рассказы / Литература 1896 года / Проза Александра Ивановича Куприна"
    assert list(m.doc_lines(nav + "\n\nОн шёл по дороге и думал о своей матери, о доме, о реке.", "ru")) == [
        "Он шёл по дороге и думал о своей матери, о доме, о реке."]
    for head in ("Комментарии Скуки ради \nВпервые напечатано в «Самарской газете», 1897, номер 275.", "Примечания (С. М. Петров)", "Примечания и комментарии", "Комментарии доктора исторических наук Н. И. Цимбаева."):
        text = "Длинный абзац основного текста, в котором много разных слов.\n\n" + head + "\n\nПоздний комментарий редактора к этому рассказу, тоже длинный."
        assert len(list(m.doc_lines(text, "ru"))) == 1, head
    # «Примечание. …» внутри художественного текста (Салтыков-Щедрин) — это текст автора, не отрезаем
    text = "Примечание. Делать пироги из грязи навсегда возбраняется.\n\nАрина Петровна так и ахнула от такой новости."
    assert len(list(m.doc_lines(text, "ru"))) == 2
