"""
Учебники ЯЗЫКОВ и ЛИТЕРАТУРЫ 1–6 класс для Mini AI (uzedu.online — копии учебников Минобразования Узбекистана).
Решение Фарходa: учить «по реальным учебникам 1–11 по четвертям»; к математике добавляем языки и литературу.

    .venv/bin/python tools/fetch_language_books.py --grade 1          # 3 учебника 1 класса: uz, ru, en
    .venv/bin/python tools/fetch_language_books.py --grade 1 --dry    # только каталог, без PDF

20.10 Фарход: «не качай всё сразу — по классу; англ, рус, узб». Поэтому: ОДИН учебник на язык для ОДНОГО класса
(самый новый). Следующий класс качаем только после того, как этот пройден, а прошлый — удаляем.
  uz: 1 кл. — Alifbe (букварь), дальше Ona tili;  ru: 1 кл. — Букварь, дальше Русский язык;  en: English.

Предметы: букварь, родной язык (Ona tili), узбекский язык, русский язык, чтение (O'qish), литература (Adabiyot),
английский язык. Языки обучения: узбекский и русский (так в школах Узбекистана).
Книги — только для обучения (образовательные цели): в git НЕ кладём, лежат вне репозитория.
PDF качаем по одному, с паузой, потоком на диск.
"""
import argparse
import html as H
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

BOOKS = Path.home() / "Developer/mini-ai-books"
PDF = BOOKS / "pdf_lang"
UA = {"User-Agent": "Mozilla/5.0"}
SECTIONS = {  # раздел сайта → код предмета
    "bukvar": "bukvar", "rodnoj-jazyk": "ona_tili", "uzbekskij-jazyk": "uzbek", "russkij-jazyk": "russian",
    "chtenie": "reading", "literatura": "literature", "anglijskij-jazyk": "english",
}
LANGS = {"Русский": "ru", "Узбекский": "uz", "Английский": "en"}
sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_school_books import get, meta  # noqa: E402


def section_links(sec):
    """Все страницы книг раздела (с учётом пагинации /page/N/)."""
    out, page = [], 1
    while page < 30:
        url = f"https://uzedu.online/{sec}/" + (f"page/{page}/" if page > 1 else "")
        try:
            h = get(url)
        except Exception:
            break
        links = sorted(set(u.replace("www.", "") for u in
                           re.findall(rf'href="(https?://(?:www\.)?uzedu\.online/{sec}/\d+/)"', h)))
        new = [u for u in links if u not in out]
        if not new:
            break
        out += new
        page += 1
        time.sleep(0.2)
    return out


def text_share(path):
    """Доля страниц PDF, где есть текстовый слой (> 200 символов без колонтитула сайта)."""
    import pymupdf
    with pymupdf.open(path) as doc:
        pages = [" ".join(pg.get_text().replace("UZEDU.ONLINE", "").split()) for pg in doc]
    return sum(len(t) > 200 for t in pages) / max(1, len(pages))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grade", type=int, required=True)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    PDF.mkdir(parents=True, exist_ok=True)
    cat_path = BOOKS / "catalog_lang.json"
    books = json.loads(cat_path.read_text()) if cat_path.exists() else []
    have = {b["url"] for b in books}
    for sec, code in SECTIONS.items():
        links = section_links(sec)
        print(f"{sec}: {len(links)} книг", flush=True)
        for u in links:
            if u in have:
                continue
            try:
                b = meta(u)
            except Exception as e:  # страница не открылась — пропускаем
                print("  нет страницы", u, e)
                continue
            b["subj"] = code
            books.append(b)
            have.add(u)
            time.sleep(0.15)
        cat_path.write_text(json.dumps(books, ensure_ascii=False, indent=1))
    sel = []
    for b in books:
        m = re.match(r"\d+", b.get("grade") or "")
        if not m or not b.get("pdf") or b.get("lang") not in LANGS:
            continue
        g = int(m.group())
        if g != a.grade:
            continue
        if b.get("type") and "чебник" not in b["type"] and "Букварь" not in (b.get("subject") or ""):
            continue                                    # только учебники (не тетради/методички)
        name = f"{g:02d}_{b['subj']}_{LANGS[b['lang']]}_{b['url'].rstrip('/').rsplit('/', 1)[-1]}.pdf"
        sel.append({**b, "grade_n": g, "lang_code": LANGS[b["lang"]], "file": name})
    # один учебник на язык. Порядок предпочтения — предмет, потом издание поновее.
    # 20.10: многие PDF — сканы без текста (букварь uz 2019/2021, русский язык 2019, чтение ru) → их не берём:
    # каждый кандидат качаем, проверяем текстовый слой (≥ 60% страниц с текстом), иначе удаляем и берём следующий.
    want = {"uz": [("uz", "ona_tili"), ("uz", "reading"), ("uz", "bukvar")],
            "ru": [("ru", "ona_tili"), ("ru", "russian"), ("ru", "bukvar"), ("ru", "reading")],
            "en": [("uz", "english"), ("ru", "english")]}
    cands = {}
    for target, prefs in want.items():
        cands[target] = [b for lang, subj in prefs
                         for b in sorted((x for x in sel if x["lang_code"] == lang and x["subj"] == subj),
                                         key=lambda x: (x.get("year") or "", x["url"]), reverse=True)]
    sel = [b for c in cands.values() for b in c[:1]]
    (BOOKS / "catalog_lang_selected.json").write_text(json.dumps(sel, ensure_ascii=False, indent=1))
    from collections import Counter
    print("Отобрано:", len(sel), dict(Counter((b["grade_n"], b["subj"], b["lang_code"]) for b in sel)), flush=True)
    if a.dry:
        return
    ok, got = 0, []
    for target, cand in cands.items():
        for b in cand:
            p = PDF / b["file"]
            if not (p.exists() and p.stat().st_size > 50_000):
                try:
                    with urllib.request.urlopen(urllib.request.Request(b["pdf"], headers=UA), timeout=300) as r, open(p, "wb") as f:
                        while chunk := r.read(1 << 20):
                            f.write(chunk)
                except Exception as e:
                    print("  не скачалось", b["file"], e, flush=True)
                    p.unlink(missing_ok=True)
                    continue
                time.sleep(1)
            share = text_share(p)
            if share < 0.6:
                print(f"  {b['file']}: скан без текста ({share:.0%} страниц с текстом) — пропускаю", flush=True)
                p.unlink()
                continue
            print(f"  {target}: {b['file']} ({b.get('subject')}, {b.get('year')}, {p.stat().st_size // 1024} КБ, текст {share:.0%})", flush=True)
            got.append(b)
            ok += 1
            break
        else:
            print(f"  {target}: нет учебника с текстом для {a.grade} класса", flush=True)
    (BOOKS / "catalog_lang_selected.json").write_text(json.dumps(got, ensure_ascii=False, indent=1))
    sel = got
    print(f"Готово: {ok}/{len(sel)} PDF в {PDF}", flush=True)


if __name__ == "__main__":
    main()
