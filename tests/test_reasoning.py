"""Рассуждения проверяет СКРИПТ (не ИИ): генератор детерминирован, нет утечки
экзаменационных чисел в обучение, каждый сгенерированный ответ `a` проходит
reasoncheck на 100%, reasoncheck ловит подставленные ошибки."""
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import reasoncheck as rc  # noqa: E402
from mini_ai.transformer import clean_question  # noqa: E402

make_reasoning = importlib.import_module("tools.make_reasoning")


def _gen():
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        make_reasoning.main()
    data = json.loads((ROOT / "data/reasoning.json").read_text(encoding="utf-8"))
    return data


def test_generator_deterministic():
    d1 = _gen()
    d2 = _gen()
    assert d1 == d2
    assert len(d1["train"]) > 0 and len(d1["test"]) > 0


def test_sizes_in_range():
    d = _gen()
    total = len(d["train"]) + len(d["test"])
    assert 60_000 <= total <= 110_000, total
    frac = len(d["test"]) / total
    assert 0.10 <= frac <= 0.25, frac


def test_no_leakage_between_train_and_test():
    """Набор чисел (kind + steps, то есть по-факту числа задачи) спрятанный в test
    не встречается ни разу в train — ни на одном языке."""
    d = _gen()

    def key(x):
        return (x["kind"], tuple(x["steps"]))

    train_keys = {key(x) for x in d["train"]}
    test_keys = {key(x) for x in d["test"]}
    assert not (train_keys & test_keys)


def test_every_generated_answer_passes_reasoncheck():
    d = _gen()
    bad = []
    for item in d["train"] + d["test"]:
        r = rc.check(item, item["a"], item["lang"])
        if not (r["answer_ok"] and r["steps_ok"] and r["check_ok"]):
            bad.append((item["kind"], item["lang"], item["a"], r["errors"]))
    assert not bad, bad[:5]


def test_reasoncheck_catches_wrong_step():
    item = {"answer": "38"}
    good = "57 − 10 = 47, 47 − 9 = 38. Проверка: 38 + 19 = 57 ✓. Ответ: 38"
    bad = "57 − 10 = 48, 47 − 9 = 38. Проверка: 38 + 19 = 57 ✓. Ответ: 38"
    assert rc.check(item, good, "ru")["steps_ok"] is True
    r = rc.check(item, bad, "ru")
    assert r["steps_ok"] is False
    assert any("wrong_step" in e for e in r["errors"])


def test_reasoncheck_catches_wrong_answer():
    item = {"answer": "38"}
    text = "57 − 10 = 47, 47 − 9 = 38. Проверка: 38 + 19 = 57 ✓. Ответ: 39"
    r = rc.check(item, text, "ru")
    assert r["answer_ok"] is False
    assert any("wrong_answer" in e for e in r["errors"])


def test_reasoncheck_catches_missing_check():
    item = {"answer": "38"}
    text = "57 − 10 = 47, 47 − 9 = 38. Ответ: 38"
    r = rc.check(item, text, "ru")
    assert r["check_ok"] is False
    assert "missing_check_line" in r["errors"]


def test_reasoncheck_works_for_all_three_languages():
    item = {"answer": "38"}
    texts = {
        "ru": "57 − 10 = 47, 47 − 9 = 38. Проверка: 38 + 19 = 57 ✓. Ответ: 38",
        "uz": "57 − 10 = 47, 47 − 9 = 38. Tekshirish: 38 + 19 = 57 ✓. Javob: 38",
        "en": "57 − 10 = 47, 47 − 9 = 38. Check: 38 + 19 = 57 ✓. Answer: 38",
    }
    for lang, text in texts.items():
        r = rc.check(item, text, lang)
        assert r["answer_ok"] and r["steps_ok"] and r["check_ok"], (lang, r)


def test_clean_question_keeps_math_intact():
    d = _gen()
    for item in d["train"][:200]:
        cleaned = clean_question(item["q"])
        # цифры должны сохраниться, вопрос не должен стать пустым
        assert any(ch.isdigit() for ch in cleaned)
        assert cleaned.strip() != ""
