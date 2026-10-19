"""Учебники (PDF) → текст по урокам для Mini AI. Ничего не выдумывает: берёт только то, что напечатано в книге.

    .venv/bin/python tools/books_to_lessons.py     # → ~/Developer/mini-ai-books/lessons.json + report

Для каждой книги:
  1. текст каждой страницы (PyMuPDF); убираем колонтитулы сайта («Все учебники Узбекистана…», номера страниц);
  2. делим на уроки по заголовкам «Урок N» / «N-dars» / «§ N»; если заголовков нет — по страницам;
  3. четверти: уроки книги делим на 4 равные по порядку части (в учебниках четверти не помечены);
  4. «текстовость»: сколько символов на страницу — у 1–2 класса почти всё картинки, текста мало.
Выход: [{book, grade, subj, lang, quarter, lesson, title, text, pages}] в порядке книги.
"""
import json
import re
from pathlib import Path

import fitz  # PyMuPDF

BOOKS = Path.home() / "Developer/mini-ai-books"
NOISE = [r"Все учебники Узбекистана на сайте UZEDU\.ONLINE", r"\*Книга предоставлена исключительно в образовательных целях",
         r"UZEDU\.ONLINE", r"uzedu\.online"]
LESSON_RE = re.compile(r"^\s*(?:Урок|урок|УРОК)\s+(\d+)|^\s*(\d+)\s*-\s*(?:dars|DARS|Dars)\b|^\s*§\s*(\d+)", re.M)


def clean(t):
    for p in NOISE:
        t = re.sub(p, " ", t, flags=re.I)
    t = t.replace("\t", " ")
    lines = [ln.strip() for ln in t.splitlines()]
    lines = [ln for ln in lines if ln and not re.fullmatch(r"\d{1,3}", ln)]   # номера страниц и одиночные цифры
    return "\n".join(lines)


def book_lessons(b):
    doc = fitz.open(BOOKS / "pdf" / b["file"])
    pages = [clean(p.get_text()) for p in doc]
    chars = sum(len(p) for p in pages)
    lessons, cur = [], None
    for i, txt in enumerate(pages):
        m = LESSON_RE.search(txt)
        if m:
            if cur:
                lessons.append(cur)
            first = txt.splitlines()
            title = next((ln for ln in first[first.index(m.group(0).strip()) + 1:] if len(ln) > 3), "") \
                if m.group(0).strip() in first else ""
            cur = {"title": title[:120], "text": txt, "pages": [i + 1]}
        elif cur:
            cur["text"] += "\n" + txt
            cur["pages"].append(i + 1)
        else:
            continue   # обложка/оглавление до первого урока
    if cur:
        lessons.append(cur)
    if len(lessons) < 8:   # нет явных уроков — режем по 2 страницы
        lessons = [{"title": "", "text": "\n".join(pages[i:i + 2]), "pages": [i + 1, i + 2]}
                   for i in range(2, len(pages), 2) if "".join(pages[i:i + 2]).strip()]
    n = len(lessons)
    out = []
    for k, ls in enumerate(lessons):
        out.append({"book": b["file"], "grade": b["grade_n"], "subj": b["subj"], "lang": b["lang_code"],
                    "quarter": 1 + (4 * k) // max(1, n), "lesson": k + 1, **ls})
    return out, chars / max(1, len(pages))


def main():
    sel = json.loads((BOOKS / "catalog_selected.json").read_text())
    allx, rep = [], []
    for b in sel:
        if not (BOOKS / "pdf" / b["file"]).exists():
            continue
        try:
            ls, cpp = book_lessons(b)
        except Exception as e:
            rep.append((b["file"], "ERR " + str(e)[:60]))
            continue
        allx += ls
        rep.append((b["file"], f"уроков {len(ls)}, символов/стр {cpp:.0f}"))
    (BOOKS / "lessons.json").write_text(json.dumps(allx, ensure_ascii=False))
    for r in rep:
        print(*r)
    print("всего уроков:", len(allx))


if __name__ == "__main__":
    main()
