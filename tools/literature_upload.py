#!/usr/bin/env python3
"""Mini AI: готовые наборы литературы → приватный Kaggle dataset + GitHub Release, проверка размера, удаление локальных копий.

    .venv/bin/python tools/literature_upload.py kaggle     # mamazanovfarhod016/mini-ai-text (создать или новая версия)
    .venv/bin/python tools/literature_upload.py release    # data-text-20261020 в mamazanovfarhod016/mini-ai (en/ru/uz.txt.gz + rest)
    .venv/bin/python tools/literature_upload.py verify     # сверка размеров на обеих сторонах
Файлы лежат в ~/mini-ai-raw/out (тяжёлые данные в git не кладём)."""
import gzip
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.request as u
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = Path(os.environ.get("MINIAI_RAW", str(Path.home() / "mini-ai-raw")))
OUT = RAW / "out"
KAGGLE = ROOT / ".venv/bin/kaggle"
SLUG = "mamazanovfarhod016/mini-ai-text"
REPO = "mamazanovfarhod016/mini-ai"
TAG = "data-text-20261020"
FILES = ["en.txt", "ru.txt", "uz.txt", "en_tinystories_rest.txt.gz"]


def meta_dir():
    d = RAW / "kaggle_pkg"
    d.mkdir(exist_ok=True)
    for f in FILES:
        if not (d / f).exists():
            os.link(OUT / f, d / f) if (OUT / f).exists() else None
    (d / "dataset-metadata.json").write_text(json.dumps({
        "title": "mini-ai-text", "id": SLUG, "licenses": [{"name": "CC-BY-SA-4.0"}],
        "subtitle": "Literature reading corpus (uz/ru/en) for Mini AI",
        "description": "Mini AI reading corpus, section A (literature). en.txt / ru.txt / uz.txt: one paragraph (or verse stanza, lines joined with ' / ') per line, "
                       "blank line = document boundary; uz is Latin with ASCII apostrophe. en_tinystories_rest.txt.gz: rest of TinyStories V2 train (CDLA-Sharing-1.0). "
                       "Sources and licenses: https://github.com/mamazanovfarhod016/mini-ai/blob/feat/teach/data/literature/REPORT.md",
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    return d


def keep_other_sets(d):
    """Набор общий со «словами» (tools/data_to_release.py кладёт их в words/). Kaggle-версия заменяет ВЕСЬ набор,
    поэтому перед version докачиваем текущие файлы, которых нет у литературы, и кладём рядом."""
    tmp = RAW / "kaggle_old"
    shutil.rmtree(tmp, ignore_errors=True)
    r = subprocess.run([str(KAGGLE), "datasets", "download", "-p", str(tmp), "--unzip", SLUG], capture_output=True, text=True)
    if r.returncode != 0 or not tmp.exists():
        print("WARN: не скачал старую версию — words/ может пропасть", r.stderr[-300:]); return
    for p in tmp.rglob("*"):
        if p.is_file() and p.name not in FILES and p.name != "dataset-metadata.json":
            rel = p.relative_to(tmp)
            dst = d / ("words" if len(rel.parts) == 1 and p.name.startswith(("words_", "lexicon_")) else "") / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), dst)
            print("keep", dst.relative_to(d))
    shutil.rmtree(tmp, ignore_errors=True)


def kaggle():
    d = meta_dir()
    exists = subprocess.run([str(KAGGLE), "datasets", "status", SLUG], capture_output=True, text=True)
    if exists.returncode == 0 and "ready" in exists.stdout.lower():
        keep_other_sets(d)
        cmd = [str(KAGGLE), "datasets", "version", "-p", str(d), "-m", "literature corpus", "--dir-mode", "zip"]
    else:
        cmd = [str(KAGGLE), "datasets", "create", "-p", str(d), "--dir-mode", "zip"]     # приватный по умолчанию
    print(" ".join(cmd), flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout, r.stderr)


def token():
    return next(re.search(r"https://[^:]+:([^@]+)@github.com", l).group(1) for l in open(Path.home() / ".git-credentials") if "github.com" in l)


def gh(m, url, data=None, hdr=None):
    r = u.Request(url, data=data, headers={"Authorization": "token " + token(), "Accept": "application/vnd.github+json", **(hdr or {})}, method=m)
    return json.load(u.urlopen(r, timeout=3600))


class FileStream:
    """Файл → тело запроса потоком (не читаем 800 МБ в память)."""
    def __init__(self, path):
        self.f, self.size = open(path, "rb"), Path(path).stat().st_size
    def read(self, n=-1):
        return self.f.read(n)
    def __len__(self):
        return self.size


def release_files():
    """en/ru/uz жмём gzip (в релизе меньше), остальной TinyStories уже .gz."""
    d = RAW / "release_pkg"
    d.mkdir(exist_ok=True)
    out = []
    for f in FILES:
        src = OUT / f
        if f.endswith(".gz"):
            out.append(src)
        else:
            dst = d / (f + ".gz")
            if not dst.exists():
                with open(src, "rb") as a, gzip.open(dst, "wb", compresslevel=6) as b:
                    shutil.copyfileobj(a, b, 1 << 20)
            out.append(dst)
    return out


def release():
    body = ("Литература для этапа «чтение» (набор A): en.txt.gz, ru.txt.gz, uz.txt.gz + en_tinystories_rest.txt.gz (остаток TinyStories V2 train). "
            "Источники и лицензии: data/literature/REPORT.md в ветке feat/teach. Копия также в Kaggle dataset mamazanovfarhod016/mini-ai-text.")
    try:
        rel = gh("GET", f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}")
    except Exception:
        rel = gh("POST", f"https://api.github.com/repos/{REPO}/releases", json.dumps(
            {"tag_name": TAG, "target_commitish": "feat/teach", "name": "Тексты для чтения: литература uz/ru/en (20.10.2026)", "body": body, "prerelease": True}).encode())
    have = {a["name"]: (a["id"], a["size"]) for a in rel["assets"]}
    for p in release_files():
        if p.name in have and have[p.name][1] == p.stat().st_size:
            continue
        if p.name in have:
            u.urlopen(u.Request(f"https://api.github.com/repos/{REPO}/releases/assets/{have[p.name][0]}", method="DELETE",
                                headers={"Authorization": "token " + token()}))
        url = rel["upload_url"].split("{")[0] + "?name=" + p.name
        a = gh("POST", url, FileStream(p), {"Content-Type": "application/gzip", "Content-Length": str(p.stat().st_size)})
        print("uploaded", p.name, a["size"], flush=True)
    verify()


def verify():
    rel = gh("GET", f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}")
    have = {a["name"]: a["size"] for a in rel["assets"]}
    ok = True
    for p in release_files():
        good = have.get(p.name) == p.stat().st_size
        ok &= good
        print("release", p.name, have.get(p.name), "local", p.stat().st_size, "OK" if good else "MISMATCH")
    print(rel["html_url"])
    r = subprocess.run([str(KAGGLE), "datasets", "files", SLUG], capture_output=True, text=True)
    print(r.stdout, r.stderr)
    return ok


if __name__ == "__main__":
    {"kaggle": kaggle, "release": release, "verify": verify}[sys.argv[1]]()
