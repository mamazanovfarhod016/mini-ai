"""
Самоулучшение Mini AI: «решает сама → скрипт проверяет → учится на своих ВЕРНЫХ решениях».
(В статьях это называют rejection sampling / STaR / expert iteration — простейшая форма обучения с наградой.)

    .venv/bin/python -m server.selfplay --model M.json --tasks data/reasoning.json --out data/selfplay.json \
        [--n 2000 --tries 4 --temp 0.7]

Как работает (без ИИ-учителя — правильность решает только скрипт):
  1. Берём задания из ОБУЧАЮЩЕЙ части (экзамен не трогаем — иначе честной проверки не будет).
  2. Модель решает каждое задание `tries` раз: первый раз «уверенно» (greedy), остальные — с температурой
     (немного случайности, чтобы пробовать разные пути решения).
  3. Скрипт проверяет: ответ верный? каждый шаг «X op Y = Z» верный? есть проверка?
     (server/reasoncheck.py, если он есть, иначе — проверка финального ответа).
  4. Верные решения — в обучающий набор (её собственные, проверенные). Неверные — в статистику:
     по каким видам задач ошибается → туда больше заданий в следующем круге.
  5. Дальше обычное дообучение на этом наборе + повторение старого (чтобы не забыть) и экзамен
     на спрятанных заданиях. Новая версия публикуется, только если экзамен лучше.
Награда здесь простая: 1 — решение верное (попадает в обучение), 0 — нет.
"""
import argparse
import json
import random
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from mini_ai.transformer import KVModel, clean_question, load_model, BOS, SEP

ANSWER_WORDS = ("ответ", "javob", "answer")


def build_prompt(tok, q, max_q):
    return [BOS] + tok.encode(clean_question(q))[:max_q] + [SEP]


def final_answer(text):
    """Текст после последнего «Ответ:/Javob:/Answer:» (без точки в конце)."""
    low = text.lower()
    pos = max(low.rfind(w) for w in ANSWER_WORDS)
    if pos < 0:
        return None
    tail = text[pos:].split(None, 1)
    return tail[1].strip().rstrip(".").strip() if len(tail) > 1 else ""


def _num(s):
    s = s.replace("−", "-").replace(",", ".").strip()
    try:
        return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None


def simple_check(item, text):
    """Запасная проверка, если server/reasoncheck.py нет: финальный ответ + все равенства «a op b = c»."""
    got, want = final_answer(text), str(item.get("answer", item.get("a", "")))
    want = final_answer(want) or want
    ok = got is not None and (got.lower() == want.lower() or (_num(got) is not None and _num(got) == _num(want)))
    steps_ok = True
    for a, op, b, c in re.findall(r"(−?\d+(?:[./]\d+)?)\s*([+−*/])\s*(−?\d+(?:[./]\d+)?)\s*=\s*(−?\d+(?:[./]\d+)?)", text):
        x, y, z = _num(a), _num(b), _num(c)
        if None in (x, y, z) or (op == "/" and y == 0):
            continue
        r = {"+": x + y, "−": x - y, "*": x * y, "/": x / y if y else None}[op]
        steps_ok &= r == z
    return {"answer_ok": ok, "steps_ok": steps_ok, "passed": ok and steps_ok}


def checker():
    try:
        from server import reasoncheck
        def f(item, text):
            r = reasoncheck.check(item, text, item.get("lang", "ru"))
            r["passed"] = bool(r.get("answer_ok") and r.get("steps_ok") and r.get("check_ok", True))
            return r
        return f
    except ImportError:
        return simple_check


def solve(kv, tok, cfg, max_q, q, temp, rng):
    ids = build_prompt(tok, q, max_q)
    out, _ = kv.generate(ids, max_new=min(cfg["block"], 400) - len(ids), temperature=temp, rng=rng)
    return tok.decode(out)


def run(model, tasks_path, out, n=2000, tries=4, temp=0.7, seed=0, weak_boost=True, log=print):
    p, cfg, tok, meta = load_model(model)
    max_q = meta.get("max_q", 24)
    kv = KVModel(p, cfg)
    data = json.loads(Path(tasks_path).read_text(encoding="utf-8"))
    train = data["train"]
    rng_py, rng = random.Random(seed), np.random.default_rng(seed)
    check = checker()
    prev = Path(out).with_suffix(".stats.json")
    weak = json.loads(prev.read_text())["fail_rate"] if weak_boost and prev.exists() else {}
    # больше заданий тех видов, где в прошлый раз чаще ошибалась
    w = [1.0 + 3.0 * weak.get(t["kind"], 0.5) for t in train]
    picked = rng_py.choices(train, weights=w, k=n)
    good, tried, solved = [], Counter(), Counter()
    for i, item in enumerate(picked):
        tried[item["kind"]] += 1
        seen = set()
        for t in range(tries):
            text = solve(kv, tok, cfg, max_q, item["q"], 0.0 if t == 0 else temp, rng)
            if text in seen:
                continue
            seen.add(text)
            if check(item, text)["passed"]:
                good.append({"q": item["q"], "a": text, "kind": item["kind"], "lang": item.get("lang"),
                             "source": "selfplay", "greedy": t == 0})
                solved[item["kind"]] += 1
                break
        if i % 200 == 0:
            log(f"  самоучёба: {i}/{n}, верных решений {len(good)}")
    fail = {k: round(1 - solved[k] / tried[k], 3) for k in tried}
    Path(out).write_text(json.dumps({"train": good}, ensure_ascii=False), encoding="utf-8")
    prev.write_text(json.dumps({"fail_rate": fail, "solved": dict(solved), "tried": dict(tried)},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"самоучёба: решила верно {len(good)}/{n} ({len(good) / max(1, n) * 100:.0f}%); "
        f"слабые виды: {', '.join(k for k, v in sorted(fail.items(), key=lambda x: -x[1])[:4])}")
    return good, fail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    run(a.model, a.tasks, a.out, a.n, a.tries, a.temp, a.seed)


if __name__ == "__main__":
    main()
