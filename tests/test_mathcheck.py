"""Примеры проверяет СКРИПТ (не ИИ): правильный ответ, типы, слабые места, тренажёр."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import mathcheck, store  # noqa: E402
from tests.test_learn import req, srv  # noqa: E402,F401


def test_right_answer():
    cases = {"1+2": "3", "1+2=": "3", "2+2=4": "Верно, 4.", "2+2=5": "Нет, будет 4.", "347+589": "936",
             "сколько будет 7 умножить на 8": "56", "10:4": "2,5", "1/0": "На ноль делить нельзя.",
             "привет": None, "2026": None, "covid-19": None, "iphone 15 pro": None}
    assert {q: mathcheck.right_answer(q) for q in cases} == cases


def test_kinds_and_weak(tmp_path):
    for got in ("936", "1000", "925"):                       # 1 верно, 2 ошибки на трёхзначных
        mathcheck.log_answer(tmp_path, "347+589", got)
    for _ in range(5):
        mathcheck.log_answer(tmp_path, "2+3", "5")
    assert mathcheck.weak_kinds(tmp_path) == ["add_3x3"]
    st = store.read_json(tmp_path / "math_stats.json", {})
    assert st["add_3x3"]["ok"] == 1 and st["add_3x3"]["wrong"] == 2 and st["add_1x1"]["ok"] == 5


def test_drill_correct_and_heldout_disjoint():
    for kind in ("add_3x3", "sub_2x2", "mul_2x1", "div_2x1"):
        tr, te = mathcheck.drill(kind)
        assert len(tr) >= 200 and len(te) >= 50 and len(tr) <= 500 and len(te) <= 100
        core = lambda q: q.replace("сколько будет ", "").replace("посчитай ", "").rstrip(" =")
        assert not {core(q) for q, _ in tr} & {core(q) for q, _ in te}, kind
        for q, a in tr[:50] + te[:50]:
            assert mathcheck.right_answer(q) == a, (q, a)            # ответ = то, что посчитал скрипт
            d = mathcheck.kind_of(*mathcheck.parse_math(q)[:3])
            assert d == kind, (q, d)


def test_server_checks_math_and_skips_mentor(srv):  # noqa: F811
    base, data = srv
    code, r = req(base + "/api/log", {"q": "347+589", "a": "1000", "lang": "ru"})
    assert code == 200 and r["math"] == {"kind": "add_3x3", "ok": False, "right": "936"}
    assert store.read_jsonl(data / "questions.jsonl")[0]["ok"] is False
    code, r = req(base + "/api/mentor", {"q": "2+2=", "a": "x"})
    assert r["id"] is None and not (data / "mentor_inbox.json").exists()   # ИИ-наставник не тратится
