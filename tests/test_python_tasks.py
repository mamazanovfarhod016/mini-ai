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
    seen = set()
    for kind in gen.VARIANTS:
        gen.build_basic(rng, kind, train, test, seen)
    gen.build_what_prints(random.Random(0), train, test, seen)
    gen.build_fix_bug(random.Random(0), train, test, seen)
    gen.build_cleanup(random.Random(0), train, test, seen)
    return train, test


def _unique(records):
    return {(r["q"], r["a"]) for r in records}


def test_generator_deterministic():
    """Два независимых запуска дают идентичные train/test."""
    t1, e1 = _generate()
    t2, e2 = _generate()
    assert t1 == t2
    assert e1 == e2
    assert len(t1) > 8000
    assert len(e1) > 1500


def test_no_qa_duplication_within_train():
    """Каждая пара (q, a) встречается в train не более одного раза (без накрутки количества)."""
    train, _ = _generate()
    sigs = [(r["q"], r["a"]) for r in train]
    assert len(sigs) == len(set(sigs))


def test_no_family_leakage_between_train_and_test():
    """Семья параметров (kind, func_name, вопрос, эталон) не встречается и в train, и в test."""
    train, test = _generate()

    def sig(rec):
        return (rec["kind"], rec["func_name"], rec["q"], rec["a"])

    train_sigs = {sig(r) for r in train}
    test_sigs = {sig(r) for r in test}
    assert not (train_sigs & test_sigs)


def test_no_qa_overlap_between_train_and_test():
    """Ни одна пара (q, a) не встречается одновременно в train и в test."""
    train, test = _generate()
    assert not (_unique(train) & _unique(test))


def test_three_languages_present_per_family():
    train, _ = _generate()
    by_kind = {}
    for r in train:
        by_kind.setdefault(r["kind"], 0)
        by_kind[r["kind"]] += 1
    for k, n in by_kind.items():
        assert n % 3 == 0, (k, n)  # каждая семья выдаёт ровно 3 записи (uz/ru/en)


def test_every_kind_has_unique_train_and_exam_items():
    """Каждый kind имеет значимое разнообразие train и НЕ НУЛЕВОЙ экзамен."""
    train, test = _generate()
    code_kinds = [k for k in gen.PYTHON_KINDS if k != "what_prints"]
    for kind in code_kinds:
        train_unique = len(_unique([r for r in train if r["kind"] == kind]))
        test_unique = len(_unique([r for r in test if r["kind"] == kind]))
        assert train_unique >= 100, (kind, "train", train_unique)
        assert test_unique >= 1, (kind, "test (exam)", test_unique)


def test_overall_diversity_targets():
    """Общие пороги разнообразия, заданные интегратором."""
    train, test = _generate()
    all_recs = train + test
    excl_wp = [r for r in all_recs if r["kind"] != "what_prints"]
    wp = [r for r in all_recs if r["kind"] == "what_prints"]
    fix_bug = [r for r in all_recs if r["kind"] == "fix_bug"]
    cleanup = [r for r in all_recs if r["kind"] == "cleanup"]

    assert len(_unique(excl_wp)) >= 3000, len(_unique(excl_wp))
    assert len(_unique(wp)) >= 1500, len(_unique(wp))
    assert len(_unique(fix_bug)) >= 600, len(_unique(fix_bug))
    assert len(_unique(cleanup)) >= 600, len(_unique(cleanup))


def test_reference_solutions_pass_codecheck_with_high_quality():
    """Каждое эталонное решение из generated data проходит codecheck с quality >= 90."""
    train, test = _generate()
    sample = {}
    for r in train + test:
        if r["kind"] in ("fix_bug", "what_prints", "cleanup"):
            continue
        sample.setdefault((r["kind"], r["func_name"]), r)
    assert len(sample) >= 50
    checked = 0
    for key, rec in sample.items():
        res = codecheck.check(rec, rec["a"])
        assert res["passed"], (key, res)
        assert res["quality"] >= 90, (key, res)
        checked += 1
        if checked >= 300:
            break


def test_fix_bug_buggy_code_fails_at_least_one_hidden_test():
    """Для fix_bug: сломанный код из вопроса должен падать хотя бы на одном скрытом тесте."""
    train, test = _generate()
    fix_bug_recs = [r for r in (train + test) if r["kind"] == "fix_bug"]
    assert fix_bug_recs
    checked = 0
    for rec in fix_bug_recs:
        if checked >= 300:
            break
        # сломанный код встроен в вопрос после маркера "fix it:"/"исправь:"/"Tuzat:" — последняя строка с ':'
        buggy_code = rec["q"].split(":\n", 1)[1]
        clean_res = codecheck.check(rec, rec["a"])
        buggy_res = codecheck.check(rec, buggy_code)
        assert clean_res["passed"], (rec["func_name"], clean_res)
        assert clean_res["quality"] >= 90, (rec["func_name"], clean_res)
        assert not (buggy_res["tests_passed"] == buggy_res["tests_total"] and buggy_res["runs"]), \
            (rec["func_name"], "buggy code did not fail any hidden test", buggy_res)
        checked += 1
    assert checked > 0


def test_cleanup_ugly_fails_quality_but_passes_tests_logically():
    """Для cleanup: ugly код проходит скрытые тесты, но quality < 80 — не passed."""
    train, test = _generate()
    cleanup_recs = [r for r in train + test if r["kind"] == "cleanup"]
    assert cleanup_recs
    checked = 0
    for rec in cleanup_recs:
        if checked >= 300:
            break
        clean_res = codecheck.check(rec, rec["a"])
        ugly_res = codecheck.check(rec, rec["ugly"])
        assert clean_res["passed"], (rec["func_name"], clean_res)
        assert clean_res["quality"] >= 90, (rec["func_name"], clean_res)
        assert ugly_res["runs"], (rec["func_name"], ugly_res)
        assert ugly_res["tests_passed"] == ugly_res["tests_total"], (rec["func_name"], ugly_res)
        assert ugly_res["quality"] < 80, (rec["func_name"], ugly_res)
        checked += 1
    assert checked > 0


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
    script = ROOT / "tools" / "make_python_tasks.py"
    data_path = ROOT / "data" / "python_tasks.json"
    if not data_path.exists():
        subprocess.run([sys.executable, str(script)], check=True, cwd=ROOT)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    assert "train" in data and "test" in data
    assert len(data["train"]) + len(data["test"]) > 8000
    rec = data["train"][0]
    for key in ("q", "a", "kind", "lang", "func_name", "tests", "hidden_tests"):
        assert key in rec
