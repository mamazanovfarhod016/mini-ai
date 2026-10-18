"""
Тесты lossless raw-режима BPE-токенизатора (ветка feat/bpe-code).

Проверяем:
  1. decode(encode(text, raw=True), raw=True) == text для ВСЕХ ответов/вопросов
     python_tasks.json, корпуса data/corpus/*.txt (сэмпл строк), reasoning.json,
     context_tasks.json (контексты), parallel.json;
  2. старые id (raw=False) НЕ изменились — сравнение на сэмпле вопросов из
     model_transformer_math.json / model_transformer_school.json;
  3. Python raw-encode == JS raw-encode (через tests/js_bpe_encode.js) на 2000
     смешанных строках (код с отступами/переносами, кириллица, узбекские
     апострофы, эмодзи);
  4. скорость кодирования data/corpus/en.txt в raw-режиме.
"""
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mini_ai.bpe import BPETokenizer, UNK  # noqa: E402
from mini_ai.transformer import clean_question  # noqa: E402


def _load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _python_tasks_texts():
    d = _load("data/python_tasks.json")
    texts = []
    for split in ("train", "test"):
        for it in d.get(split, []):
            texts.append(it["q"])
            texts.append(it["a"])
    return texts


def _reasoning_texts():
    d = _load("data/reasoning.json")
    texts = []
    for split in ("train", "test"):
        for it in d.get(split, []):
            texts.append(it["q"])
            texts.append(it["a"])
    return texts


def _context_texts():
    d = _load("data/context_tasks.json")
    texts = []
    for split in ("train", "test"):
        for it in d.get(split, []):
            for k in ("context", "q", "a"):
                v = it.get(k)
                if isinstance(v, str) and v:
                    texts.append(v)
    return texts


def _parallel_texts():
    d = _load("data/parallel.json")
    texts = []
    for split in ("train", "test"):
        for it in d.get(split, []):
            texts.append(it["q"])
            texts.append(it["a"])
    return texts


def _corpus_lines(limit=4000):
    lines = []
    for fname in ("en.txt", "ru.txt", "uz.txt"):
        p = ROOT / "data/corpus" / fname
        if p.exists():
            lines += p.read_text(encoding="utf-8").splitlines()
    random.Random(0).shuffle(lines)
    return lines[:limit]


@pytest.fixture(scope="module")
def tok():
    """Обучаем небольшой базовый токенизатор, затем extend() на код+тексты —
    так же, как это будет использоваться в реальном пайплайне."""
    t = BPETokenizer()
    base_texts = [clean_question(q) for q in ["привет как дела", "what is two plus two", "salom qalaysiz"]] * 3
    t.train(base_texts, vocab_size=120)
    code_corpus = _python_tasks_texts()[:400] + _corpus_lines(800) + _reasoning_texts()[:200]
    t.extend(code_corpus, extra=3000, min_count=2)
    return t


# ======================================================================================
# 1. Lossless round-trip raw=True на всех датасетах
# ======================================================================================
@pytest.mark.parametrize("name,loader", [
    ("python_tasks", _python_tasks_texts),
    ("reasoning", _reasoning_texts),
    ("context_tasks", _context_texts),
    ("parallel", _parallel_texts),
])
def test_raw_roundtrip_datasets(tok, name, loader):
    texts = loader()
    bad = []
    for t in texts:
        got = tok.decode(tok.encode(t, raw=True), raw=True)
        if got != t:
            bad.append((t[:80], got[:80]))
    assert not bad, f"[{name}] {len(bad)}/{len(texts)} несовпадений round-trip, пример: {bad[:3]}"


def test_raw_roundtrip_corpus_lines(tok):
    lines = _corpus_lines(4000)
    bad = []
    for ln in lines:
        got = tok.decode(tok.encode(ln, raw=True), raw=True)
        if got != ln:
            bad.append((ln[:80], got[:80]))
    assert not bad, f"{len(bad)}/{len(lines)} несовпадений, пример: {bad[:3]}"


def test_raw_roundtrip_misc_strings(tok):
    """Код с отступами/табами/переносами, Cyrillic, узбекские апострофы, эмодзи."""
    samples = [
        "def foo(x):\n    if x > 0:\n        return x\n    return -x\n",
        "a = 1\n\tb = 2\n  \t  c=3   \n\n\nend",
        "Привет,   как дела?\n\nХорошо!",
        "Salom, bu o‘zbekcha matn: g‘isht, ko‘p.",
        "emoji test 😀🚀🔥 multi   space    run",
        "    indented\n        double indented\n",
        "mixed\ttabs and    spaces\t\tend",
        "",
        "   ",
        "\n\n\n",
        "CamelCase ALLCAPS lowercase MiXeD",
    ]
    for s in samples:
        got = tok.decode(tok.encode(s, raw=True), raw=True)
        assert got == s, (s, got)


def test_raw_no_silent_unk(tok):
    """В raw-режиме незнакомые символы уходят в байтовый fallback, а не в <unk>."""
    s = "неизвестный эмодзи 🦄🧩 текст"
    ids = tok.encode(s, raw=True)
    assert UNK not in ids
    assert tok.decode(ids, raw=True) == s


# ======================================================================================
# 2. Старые id (raw=False) не изменились — регрессия на старых моделях
# ======================================================================================
@pytest.mark.parametrize("model_name", ["model_transformer_math.json", "model_transformer_school.json"])
def test_old_model_encode_unchanged(model_name):
    path = ROOT / model_name
    if not path.exists():
        pytest.skip(f"{model_name} отсутствует")
    d = json.loads(path.read_text(encoding="utf-8"))
    tk = d.get("tokenizer")
    if tk != "bpe" and not isinstance(d.get("vocab"), list):
        pytest.skip("не BPE-модель")
    old_tok = BPETokenizer.from_dict({"vocab": d["vocab"], "merges": d.get("merges", [])})
    new_tok = BPETokenizer.from_dict({"vocab": d["vocab"], "merges": d.get("merges", [])})
    new_tok.extend(_python_tasks_texts()[:100] + _corpus_lines(200), extra=500, min_count=2)

    questions = []
    tq = d.get("test_questions")
    if isinstance(tq, list):
        for q in tq:
            questions.append(q if isinstance(q, str) else q.get("q", ""))
    if not questions:
        questions = ["привет", "как дела", "сколько будет 2 + 2", "сколько сейчас времени"]

    mismatches = []
    for q in questions[:50]:
        cq = clean_question(q)
        old_ids = old_tok.encode(cq)
        new_ids = new_tok.encode(cq)   # raw=False по умолчанию — старое поведение
        if old_ids != new_ids:
            mismatches.append((q, old_ids, new_ids))
    assert not mismatches, f"{len(mismatches)} несовпадений старых id после extend(): {mismatches[:3]}"
    # первые n0 id словаря не изменились
    assert new_tok.itos[:len(old_tok.itos)] == old_tok.itos


# ======================================================================================
# 3. Python raw-encode == JS raw-encode на 2000 смешанных строках
# ======================================================================================
def _mixed_strings(n=2000):
    rng = random.Random(42)
    code_snippets = [
        "def f(x):\n    return x + 1\n",
        "for i in range(10):\n    print(i)\n    if i % 2 == 0:\n        continue\n",
        "class Foo:\n\tdef __init__(self):\n\t\tself.x = 1\n",
        "a    =    1\nb\t=\t2\nresult = a+b   # comment\n",
    ]
    extra = ["Привет мир", "Salom olam, bu o‘zbekcha gap", "emoji 😀🚀", "UPPER lower MiXeD",
              "многие    пробелы   подряд", "\ttab\tindent\t", "line1\nline2\nline3"]
    base = _python_tasks_texts() + _corpus_lines(500) + extra + code_snippets
    out = []
    for i in range(n):
        s = rng.choice(base)
        if rng.random() < 0.3:
            s = s + rng.choice(code_snippets)
        out.append(s)
    return out


@pytest.mark.skipif(shutil.which("node") is None, reason="нет node")
def test_python_js_raw_parity(tok, tmp_path):
    texts = _mixed_strings(2000)
    model_path = tmp_path / "bpe_tok.json"
    model_path.write_text(json.dumps(tok.to_dict(), ensure_ascii=False), encoding="utf-8")

    out = subprocess.run(
        ["node", str(ROOT / "tests/js_bpe_encode.js"), str(model_path), "raw"],
        input=json.dumps(texts, ensure_ascii=False),
        capture_output=True, text=True, check=True,
    ).stdout
    js_results = json.loads(out)
    assert len(js_results) == len(texts)

    mismatches = []
    decode_mismatches = []
    for t, js in zip(texts, js_results):
        py_ids = tok.encode(t, raw=True)
        if py_ids != js["ids"]:
            mismatches.append((t[:60], py_ids[:20], js["ids"][:20]))
            continue
        py_decoded = tok.decode(py_ids, raw=True)
        if py_decoded != js["decoded"]:
            decode_mismatches.append((t[:60], py_decoded[:60], js["decoded"][:60]))
    assert not mismatches, f"{len(mismatches)} несовпадений encode, пример: {mismatches[:3]}"
    assert not decode_mismatches, f"{len(decode_mismatches)} несовпадений decode, пример: {decode_mismatches[:3]}"


# ======================================================================================
# 4. Скорость: кодирование data/corpus/en.txt
# ======================================================================================
def test_encode_speed_en_corpus(tok):
    path = ROOT / "data/corpus/en.txt"
    if not path.exists():
        pytest.skip("en.txt отсутствует")
    text = path.read_text(encoding="utf-8")
    t0 = time.perf_counter()
    ids = tok.encode(text, raw=True)
    dt = time.perf_counter() - t0
    mb = len(text.encode("utf-8")) / (1024 * 1024)
    rate = mb / dt if dt > 0 else float("inf")
    print(f"\nencode speed: {mb:.2f} MB in {dt:.2f}s -> {rate:.2f} MB/s, {len(ids)} ids")
    # экстраполяция на 20 МБ (требование: разумное время для 20 МБ текста)
    est_20mb = 20.0 / rate if rate > 0 else float("inf")
    print(f"estimated time for 20 MB: {est_20mb:.1f}s")
    assert est_20mb < 180, f"20 МБ оценочно заняли бы {est_20mb:.1f}с — слишком долго"
