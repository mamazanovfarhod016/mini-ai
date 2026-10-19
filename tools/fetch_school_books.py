"""Скачивает официальные школьные учебники Узбекистана (uzedu.online, копии с сайта Минобразования)
для Mini AI: математика, алгебра, геометрия, физика, информатика, 1–11 класс, языки: русский и узбекский.

    .venv/bin/python tools/fetch_school_books.py      # → ~/Developer/mini-ai-books/pdf/ + catalog_selected.json

Книги — только для обучения (образовательные цели), в git НЕ кладём (большие + чужие права):
каталог и тексты лежат вне репозитория, в ~/Developer/mini-ai-books/.
"""
import html as H
import json
import re
import time
import urllib.request
from pathlib import Path

BOOKS = Path.home() / "Developer/mini-ai-books"
PDF = BOOKS / "pdf"
UA = {"User-Agent": "Mozilla/5.0"}
LANGS = {"Русский": "ru", "Узбекский": "uz"}
SUBJ = {"Математика": "math", "Алгебра": "algebra", "Геометрия": "geometry", "Физика": "physics",
        "Информатика": "informatics"}


def get(u, binary=False, timeout=60):
    data = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=timeout).read()
    return data if binary else data.decode("utf-8", "replace")


def meta(url):
    h = get(url)
    txt = re.sub(r"\s+", " ", H.unescape(re.sub(r"<[^>]+>", " ", h)))
    stop = r"\s+(?:Класс|Авторы|Издательство|Год|Страниц|Язык|Тип|Читать|-->)"
    f = lambda k: (re.search(k + r":\s*(.+?)" + stop, txt) or [None, None])[1]
    pdf = re.search(r'data-src="(/books/[^"]+\.pdf)"', h)
    return {"url": url, "subject": f("Предмет"), "grade": f("Класс"), "authors": f("Авторы"), "year": f("Год"),
            "pages": f("Страниц"), "lang": f("Язык"), "type": f("Тип"),
            "pdf": ("https://uzedu.online" + pdf.group(1)) if pdf else None}


def main():
    PDF.mkdir(parents=True, exist_ok=True)
    cat_path = BOOKS / "catalog.json"
    books = json.loads(cat_path.read_text()) if cat_path.exists() else []
    have = {b["url"] for b in books}
    # физики нет на страницах классов — берём из раздела /fizika/
    for u in sorted(set(re.findall(r'href="(https?://(?:www\.)?uzedu\.online/fizika/\d+/)"', get("https://uzedu.online/fizika/")))):
        u = u.replace("www.", "")
        if u not in have:
            books.append(meta(u)); time.sleep(0.15)
    cat_path.write_text(json.dumps(books, ensure_ascii=False, indent=1))

    sel = []
    for b in books:
        if b["lang"] not in LANGS or b["subject"] not in SUBJ or not b["pdf"] or not b["grade"]:
            continue
        g = int(re.match(r"\d+", b["grade"]).group())
        name = f"{g:02d}_{SUBJ[b['subject']]}_{LANGS[b['lang']]}_{b['url'].rstrip('/').rsplit('/', 1)[-1]}.pdf"
        sel.append({**b, "grade_n": g, "subj": SUBJ[b["subject"]], "lang_code": LANGS[b["lang"]], "file": name})
    sel.sort(key=lambda b: (b["grade_n"], b["subj"], b["lang_code"]))
    ok = 0
    for b in sel:
        f = PDF / b["file"]
        if not f.exists() or f.stat().st_size < 10000:
            try:
                f.write_bytes(get(b["pdf"], binary=True, timeout=300))
                time.sleep(0.5)
            except Exception as e:
                print("не скачалось:", b["file"], e)
                continue
        ok += 1
    (BOOKS / "catalog_selected.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1))
    print(f"книг выбрано {len(sel)}, скачано {ok}")


if __name__ == "__main__":
    main()
