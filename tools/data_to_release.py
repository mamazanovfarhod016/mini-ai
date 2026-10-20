#!/usr/bin/env python3
"""Готовые наборы данных (> 5 МБ или все полные файлы) → приватный Kaggle dataset + GitHub Release, проверка размера,
потом удаление локально.

    .venv/bin/python tools/data_to_release.py --dir /home/hermes/mini-ai-raw/words/out --name words \
        --title "Mini AI: word meanings" --notes data/words/UPLOADS.md [--delete]

Kaggle: mamazanovfarhod016/mini-ai-text (общий для «литературы» и «слов»; каждый набор в своей подпапке <name>/,
новая версия не стирает чужие подпапки — перед `version` скачиваем метаданные и кладём остальное как есть).
GitHub Release: тег data-text-20261020 репо mamazanovfarhod016/mini-ai, файлы <name>_*.gz.
"""
import argparse, gzip, json, os, re, shutil, subprocess, sys, tempfile, time, urllib.request as u
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO, TAG = "mamazanovfarhod016/mini-ai", "data-text-20261020"
KAGGLE = ROOT / ".venv/bin/kaggle"
KSLUG = "mamazanovfarhod016/mini-ai-text"
tok = next(re.search(r"https://[^:]+:([^@]+)@github.com", l).group(1)
           for l in open("/home/hermes/.git-credentials") if "github.com" in l)
H = {"Authorization": "token " + tok, "Accept": "application/vnd.github+json"}


def api(m, url, data=None, hdr=None):
    r = u.Request(url, data=data, headers={**H, **(hdr or {})}, method=m)
    return json.load(u.urlopen(r, timeout=1800))


def release():
    try:
        return api("GET", f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}")
    except Exception:
        return api("POST", f"https://api.github.com/repos/{REPO}/releases", json.dumps({
            "tag_name": TAG, "target_commitish": "feat/teach", "name": "Тексты для Mini AI: литература и значения слов (20.10.2026)",
            "body": "Стадии «чтение» и «слова». Источники и лицензии — data/literature/REPORT.md и data/words/REPORT.md. "
                    "Наборы из Викисловаря — CC BY-SA 4.0.", "prerelease": True}).encode())


def gz(src, dst):
    with open(src, "rb") as f, gzip.open(dst, "wb", compresslevel=6) as g:
        shutil.copyfileobj(f, g, 1 << 20)


def upload_release(files, name):
    rel = release()
    have = {a["name"]: a for a in rel["assets"]}
    out = {}
    for f in files:
        already = f.name.endswith(".gz")
        an = f"{name}_{f.name}" if already else f"{name}_{f.name}.gz"
        tmp = f if already else Path("/home/hermes/mini-ai-raw") / an      # не /tmp: большие файлы
        if not already:
            gz(f, tmp)
        size = tmp.stat().st_size
        if an in have and have[an]["size"] != size:
            api("DELETE", have[an]["url"])
            have.pop(an)
        if an not in have:
            up = rel["upload_url"].split("{")[0] + "?name=" + an
            with open(tmp, "rb") as fh:                                    # потоком, не читая файл в память
                a = api("POST", up, fh, {"Content-Type": "application/gzip", "Content-Length": str(size)})
            print("release uploaded", an, a["size"], flush=True)
        if not already:
            tmp.unlink()
        out[f.name] = (an, size)
    rel = api("GET", f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}")
    have = {a["name"]: a for a in rel["assets"]}
    ok = {fn: (an in have and have[an]["size"] == size) for fn, (an, size) in out.items()}
    return rel["html_url"], {fn: have.get(an, {}).get("browser_download_url") for fn, (an, _) in out.items()}, ok


def kaggle(*args, timeout=3600):
    r = subprocess.run([str(KAGGLE), *args], capture_output=True, text=True, timeout=timeout)
    print("kaggle", " ".join(args[:3]), "→", r.returncode, (r.stdout + r.stderr).strip()[-400:], flush=True)
    return r


def upload_kaggle(files, name, title):
    """Kaggle-версия заменяет весь набор → собираем папку: чужие подпапки скачиваем (если набор уже есть) + наша."""
    work = Path(tempfile.mkdtemp(prefix="kg_", dir="/home/hermes/mini-ai-raw"))
    exists = kaggle("datasets", "metadata", "-p", str(work), KSLUG).returncode == 0
    if exists:
        kaggle("datasets", "download", "-p", str(work / "_old"), "--unzip", KSLUG)
        old = work / "_old"
        if old.exists():
            for p in old.iterdir():
                if p.name != name:
                    shutil.move(str(p), work / p.name)
            shutil.rmtree(old, ignore_errors=True)
    sub = work / name
    sub.mkdir(exist_ok=True)
    for f in files:
        try:
            os.link(f, sub / f.name)                                       # жёсткая ссылка: без лишней копии на диске
        except OSError:
            shutil.copy(f, sub / f.name)
    meta = {"title": "mini-ai-text", "id": KSLUG, "licenses": [{"name": "CC-BY-SA-4.0"}],
            "subtitle": "Mini AI: reading texts and word meanings (uz/ru/en)",
            "description": "Stages 'reading' and 'words' for Mini AI (github.com/mamazanovfarhod016/mini-ai). "
                           "Sources and licenses: data/literature/REPORT.md, data/words/REPORT.md."}
    (work / "dataset-metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    if exists:
        r = kaggle("datasets", "version", "-p", str(work), "-m", f"{name}: {title}", "--dir-mode", "zip")
    else:
        r = kaggle("datasets", "create", "-p", str(work), "--dir-mode", "zip")
    ok = r.returncode == 0
    # проверка: список файлов набора и размеры (Kaggle обрабатывает версию не сразу)
    listing = ""
    for _ in range(30):
        time.sleep(20)
        lr = subprocess.run([str(KAGGLE), "datasets", "files", KSLUG], capture_output=True, text=True, timeout=300)
        listing = lr.stdout
        if lr.returncode == 0 and (name in listing):
            break
    shutil.rmtree(work, ignore_errors=True)
    return ok and name in listing, listing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--glob", default="*.jsonl", help="можно несколько через запятую: '*.txt,*.txt.gz'")
    ap.add_argument("--notes", default=None, help="куда записать ссылки (markdown) для отчёта")
    ap.add_argument("--delete", action="store_true")
    ap.add_argument("--skip-kaggle", action="store_true")
    a = ap.parse_args()
    files = sorted({f for g in a.glob.split(",") for f in Path(a.dir).glob(g.strip())})
    print("files", [(f.name, f.stat().st_size) for f in files], flush=True)
    url, links, rel_ok = upload_release(files, a.name)
    k_ok, listing = (False, "") if a.skip_kaggle else upload_kaggle(files, a.name, a.title)
    print("release", url, rel_ok, "\nkaggle ok", k_ok, flush=True)
    if a.notes:
        L = ["## Загрузки (проверено размером)", "",
             f"- Kaggle (приватный): https://www.kaggle.com/datasets/{KSLUG} — папка `{a.name}/` — "
             + ("загружено ✓" if k_ok else "НЕ ПОДТВЕРЖДЕНО"),
             f"- GitHub Release: {url}", "", "| файл | байт | Release (.gz) |", "|---|---|---|"]
        for f in files:
            L.append(f"| {f.name} | {f.stat().st_size:,} | {'[скачать](' + links[f.name] + ')' if rel_ok[f.name] else 'нет'} |")
        Path(a.notes).write_text("\n".join(L) + "\n", encoding="utf-8")
    if a.delete:
        if all(rel_ok.values()) and (k_ok or a.skip_kaggle):
            for f in files:
                f.unlink()
            print("deleted local", len(files))
        else:
            print("NOT deleted: upload not verified", rel_ok, k_ok)
            return 1


if __name__ == "__main__":
    sys.exit(main())
