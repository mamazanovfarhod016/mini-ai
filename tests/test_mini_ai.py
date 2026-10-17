"""Тесты: численная проверка градиента + модель отвечает правильно + Python == JS."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mini_ai.brain import Brain  # noqa: E402
from mini_ai.features import fnv1a, normalize, tokens  # noqa: E402
from mini_ai.mlp import MLP  # noqa: E402


# ---------- 1. gradient check ----------
def test_gradient_check():
    """
    Сравниваем градиент из backprop с численным:
        dL/dw ≈ (L(w + eps) - L(w - eps)) / (2*eps)
    Если backprop написан правильно, относительная ошибка < 1e-6.
    """
    rng = np.random.default_rng(0)
    net = MLP(12, 7, 5, seed=1)
    X = rng.normal(size=(6, 12))
    y = rng.integers(0, 5, size=6)
    l2 = 1e-2
    _, cache = net.loss(X, y, l2)
    grads = net.backward(cache, y, l2)
    eps = 1e-5
    for name, W in net.params.items():
        num = np.zeros_like(W)
        it = np.nditer(W, flags=["multi_index"])
        for _ in it:
            i = it.multi_index
            old = W[i]
            W[i] = old + eps
            lp, _ = net.loss(X, y, l2)
            W[i] = old - eps
            lm, _ = net.loss(X, y, l2)
            W[i] = old
            num[i] = (lp - lm) / (2 * eps)
        rel = np.linalg.norm(num - grads[name]) / (np.linalg.norm(num) + np.linalg.norm(grads[name]) + 1e-12)
        assert rel < 1e-6, f"{name}: относительная ошибка {rel:.2e}"


def test_features():
    assert normalize("Привет!!! Как ДЕЛА?") == "привет как дела"
    assert normalize("Yo'q") == normalize("yoʻq") == "yoq"
    assert "c:как" in tokens("как дила")
    assert fnv1a("") == 2166136261


# ---------- 2. обученная модель ----------
CONTROL = [
    ("привет", "greeting"), ("превет", "greeting"), ("salom", "greeting"),
    ("как дела", "how_are_you"), ("как дила", "how_are_you"), ("qalaysan", "how_are_you"),
    ("пока", "goodbye"), ("xayr", "goodbye"),
    ("кто ты", "who_are_you"), ("как тебя зовут", "name"),
    ("кто тебя создал", "creator"), ("seni kim yaratgan", "creator"),
    ("спасибо", "thanks"), ("rahmat", "thanks"),
    ("да", "yes"), ("нет", "no"), ("yo'q", "no"),
    ("сколько тебе лет", "age"), ("что ты умеешь", "abilities"),
    ("расскажи шутку", "joke"), ("какая погода", "weather"),
    ("который час", "time"), ("какое сегодня число", "date"),
    ("мне грустно", "sad"), ("что делаешь", "what_doing"),
]


@pytest.fixture(scope="module")
def brain():
    return Brain()


@pytest.mark.parametrize("phrase,intent", CONTROL)
def test_control_phrases(brain, phrase, intent):
    t = brain.think(phrase)
    assert t["intent"] == intent, (phrase, t["top3"])


def test_unknown_is_honest(brain):
    # Сложные вопросы → класс out_of_scope или порог уверенности → честное «не знаю»
    for q in ["объясни квантовую хромодинамику", "xyzzy qwrtp", "реши уравнение x квадрат", "столица германии"]:
        text, t = brain.reply(q)
        assert t["intent"] in (None, "out_of_scope"), (q, t)
        assert any(w in text for w in ("не знаю", "сложно", "bilmayman", "murakkab")), text


def test_creator_is_farhod(brain):
    text, _ = brain.reply("кто тебя создал")
    assert "Фарход" in text


def test_model_small():
    assert (ROOT / "model.json").stat().st_size < 500 * 1024


# ---------- 3. Python и JavaScript думают одинаково ----------
@pytest.mark.skipif(shutil.which("node") is None, reason="нет node")
def test_python_js_parity(brain):
    data = json.loads((ROOT / "data/intents.json").read_text(encoding="utf-8"))
    phrases = [p for it in data["intents"] for p in it["patterns"][:6]]
    phrases += [p for p, _ in CONTROL] + ["объясни квантовую хромодинамику", "Привет!!! КАК ДЕЛА?", "Yoʻq rahmat", ""]
    out = subprocess.run(["node", str(ROOT / "tests/js_think.js")], input=json.dumps(phrases, ensure_ascii=False),
                         capture_output=True, text=True, check=True).stdout
    js = json.loads(out)
    assert len(js) == len(phrases)
    for p, j in zip(phrases, js):
        t = brain.think(p)
        assert t["intent"] == j["intent"], p
        assert abs(t["confidence"] - j["confidence"]) < 1e-9, p
        assert [a for a, _ in t["top3"]] == [a for a, _ in j["top3"]], p
        assert abs(t["familiar"] - j["familiar"]) < 1e-12, p
