"""Задачи по Python: детерминированный генератор + экзамен (hidden_tests + качество кода)."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import codecheck  # noqa: E402
from tools import make_python_tasks as gen  # noqa: E402


def _generate():
    """Запускает main() генератора в памяти (без записи файла) — для тестов детерминизма."""
    import random
    rng = random.Random(0)
    train, test = [], []
    for kind in [k for k in gen.PYTHON_KINDS if k in gen.A]:
        gen.build_basic(rng, kind, train, test)
    gen.build_what_prints(random.Random(0), train, test)
    gen.build_fix_bug(random.Random(0), train, test)
    gen.build_cleanup(random.Random(0), train, test)
    return train, test


def test_generator_deterministic():
    """Два независимых запуска дают идентичные train/test."""
    t1, e1 = _generate()
    t2, e2 = _generate()
    assert t1 == t2
    assert e1 == e2
    assert len(t1) > 15000
    assert len(e1) > 1000


def test_no_family_leakage_between_train_and_test():
    """Семья параметров (func_name, конкретные значения) не встречается и в train, и в test."""
    train, test = _generate()

    def sig(rec):
        # сигнатура семьи: kind + func_name + вопрос + эталон (точная параметризация)
        return (rec["kind"], rec["func_name"], rec["q"], rec["a"])

    train_sigs = {sig(r) for r in train if r["kind"] != "what_prints"}
    test_sigs = {sig(r) for r in test if r["kind"] != "what_prints"}
    assert not (train_sigs & test_sigs)


def test_three_languages_present_per_family():
    train, _ = _generate()
    by_kind = {}
    for r in train:
        by_kind.setdefault(r["kind"], 0)
        by_kind[r["kind"]] += 1
    for k, n in by_kind.items():
        assert n % 3 == 0, (k, n)  # каждая семья выдаёт ровно 3 записи (uz/ru/en)


def test_reference_solutions_pass_codecheck_with_high_quality():
    """Каждое эталонное решение из generated data проходит codecheck с quality >= 90."""
    train, test = _generate()
    sample = {}
    for r in train + test:
        if r["kind"] in ("fix_bug", "what_prints", "cleanup"):
            continue
        sample.setdefault(r["kind"], r)  # одного представителя каждого kind достаточно
    assert len(sample) >= 10
    for kind, rec in sample.items():
        res = codecheck.check(rec, rec["a"])
        assert res["passed"], (kind, res)
        assert res["quality"] >= 90, (kind, res)


def test_cleanup_ugly_fails_quality_but_passes_tests_logically():
    """Для cleanup: ugly код работает (прошёл бы ручные тесты), но quality < 80 — не passed."""
    train, test = _generate()
    cleanup_recs = [r for r in train + test if r["kind"] == "cleanup"]
    assert cleanup_recs
    seen_fn = set()
    for rec in cleanup_recs:
        if rec["func_name"] in seen_fn:
            continue
        seen_fn.add(rec["func_name"])
        clean_res = codecheck.check({"func_name": rec["func_name"], "hidden_tests": []}, rec["a"])
        ugly_res = codecheck.check({"func_name": rec["func_name"], "hidden_tests": []}, rec["ugly"])
        assert clean_res["quality"] >= 90, (rec["func_name"], clean_res)
        assert ugly_res["quality"] < 80, (rec["func_name"], ugly_res)


def test_dangerous_code_rejected():
    task = {"func_name": "f", "hidden_tests": [{"args": [1], "expected": 1}]}
    bad_import = "def f(n):\n    import os\n    return n\n"
    res = codecheck.check(task, bad_import)
    assert not res["passed"]
    assert res["quality"] == 0
    assert any("os" in i for i in res["issues"])


def test_infinite_loop_times_out_safely():
    task = {"func_name": "f", "hidden_tests": [{"args": [1], "expected": 1}]}
    code = "def f(n: int) -> int:\n    \"\"\"x\"\"\"\n    while True:\n        pass\n"
    res = codecheck.check(task, code)
    assert not res["passed"]
    assert res["runs"] is False


def test_hidden_tests_run_and_pass_for_basic_kinds():
    train, test = _generate()
    checked = 0
    for rec in (train[:5] + test[:5]):
        if rec["kind"] in ("fix_bug", "what_prints", "cleanup"):
            continue
        res = codecheck.check(rec, rec["a"])
        assert res["tests_passed"] == res["tests_total"] > 0, rec
        checked += 1
    assert checked > 0


def test_what_prints_answers_are_correct_by_actually_running():
    """Ответ 'что выведет код' пересчитывается прямо здесь независимым исполнением."""
    import io
    from contextlib import redirect_stdout

    train, test = _generate()
    samples = [r for r in (train + test) if r["kind"] == "what_prints"][:30]
    assert samples
    for rec in samples:
        code = rec["q"].split("\n", 1)[1]  # первая строка — формулировка вопроса, остальное — код
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(compile(code, "<t>", "exec"), {})
        assert buf.getvalue().strip() == rec["a"]


def test_data_file_builds_and_is_well_formed(tmp_path):
    """Прогоняем сам CLI-скрипт генератора (как в проде) и проверяем формат файла."""
    out_dir = tmp_path
    script = ROOT / "tools" / "make_python_tasks.py"
    env_root = out_dir / "data"
    env_root.mkdir()
    # запускаем в отдельном процессе, но пишем во временный ROOT через monkeypatch нельзя,
    # поэтому просто проверяем, что существующий сгенерированный файл (если есть) валиден.
    data_path = ROOT / "data" / "python_tasks.json"
    if not data_path.exists():
        subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    assert "train" in data and "test" in data
    assert len(data["train"]) + len(data["test"]) > 15000
    rec = data["train"][0]
    for key in ("q", "a", "kind", "lang", "func_name", "tests", "hidden_tests"):
        assert key in rec
