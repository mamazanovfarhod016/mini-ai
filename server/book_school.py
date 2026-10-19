"""Учёба ПО КНИГЕ, четверть за четвертью (решение Фархода 19.10.2026: «1 четверть 1 класса полностью по книге,
пока не пройдёт — дальше к следующей нельзя»).

    .venv/bin/python -m server.book_school            # один шаг: учит текущую четверть и сдаёт экзамен
    .venv/bin/python -m server.book_school --status   # где сейчас

Состояние — server_data/book_school.json:
  {"model": путь к текущей модели, "grade": 1, "quarter": 1, "attempts": n, "layers": 13, "history": [...]}
  layers — 19.10 Фарход: «вырасти хотя бы до 6 млн» → 13 слоёв × 192 ≈ 6,1 млн (рост без потери знаний:
  новые слои сначала «молчат», проверено — ответы 8 из 8 те же).
Шаг:
  1. берём задания текущей четверти (учить) + 30% повторения всех прошлых четвертей (чтобы не забывала);
  2. учим на Kaggle (server.kaggle_train, с текущей модели — знания не теряются);
  3. ЭКЗАМЕН здесь, скриптом: спрятанные задания текущей четверти + выборка спрятанных из прошлых;
  4. ≥ PASS (90%) по текущей И ≥ 85% по прошлым → следующая четверть, модель принимается;
     иначе — та же четверть ещё раз (модель-кандидат всё равно берём, если не хуже по прошлым).
Задания — только из учебников (tools/book_tasks.py → ~/Developer/mini-ai-books/tasks.json); ответы считает Python.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BOOKS = Path.home() / "Developer/mini-ai-books"
STATE = ROOT / "server_data/book_school.json"
PASS, KEEP = 0.90, 0.85
STEPS = 3000


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"model": str(ROOT / "server_data/models/_candidate_en2.json"), "grade": 1, "quarter": 1,
            "attempts": 0, "history": []}


def stages():
    return json.loads((BOOKS / "tasks.json").read_text())


def idx_of(st, g, q):
    return next(i for i, s in enumerate(st) if s["grade"] == g and s["quarter"] == q)


def exam(model_path, items):
    """Ответ модели = последнее число после «Ответ/Javob:», иначе весь текст. Проверка — сравнение строк."""
    import numpy as np  # noqa: F401
    from mini_ai.transformer import KVModel, load_model
    from train_transformer import greedy_answer
    p, cfg, tok, _ = load_model(model_path)
    kv, block = KVModel(p, cfg), cfg["block"]
    ok = 0
    wrong = []
    for x in items:
        out = greedy_answer(kv, tok, x["q"], block).strip()
        got = out.split(":")[-1].strip() if ("Ответ" in out or "Javob" in out) else out
        if got == x["answer"]:
            ok += 1
        elif len(wrong) < 8:
            wrong.append((x["q"], x["answer"], out[:60]))
    return ok / max(1, len(items)), wrong


def preflight(train, test, log=print):
    """Совет Карпаты: «посмотри на данные прямо перед сетью». Перед КАЖДОЙ отправкой на Kaggle:
    1) каждый ответ пересчитан Python заново (ошибочный ответ в данных = сеть учит ошибку);
    2) ни одно экзаменационное задание не попало в учёбу (иначе экзамен нечестный);
    3) 5 случайных заданий — в лог, глазами."""
    import re
    from tools.book_tasks import calc
    test_q = {x["q"] for x in test}
    leak = [x["q"] for x in train if x["q"] in test_q]
    bad = []
    for x in train:
        t = x["q"].split()
        v = calc(t[0::2], t[1::2])
        final = re.split(r"(?:Ответ|Javob|Answer):\s*", x["a"])[-1].strip()
        if v is None or str(int(v)).replace("-", "−") != final:
            bad.append((x["q"], x["a"]))
    for x in random.Random(0).sample(train, min(5, len(train))):
        log(f"  данные: {x['q']}  →  {x['a']}")
    if leak or bad:
        raise RuntimeError(f"Данные не прошли проверку: утечек в экзамен {len(leak)}, неверных ответов {len(bad)}: {bad[:3]}")
    log(f"Проверка данных: {len(train)} заданий, все ответы верны, утечек в экзамен нет")


def step(log=print):
    from server import kaggle_train
    st = stages()
    s = load_state()
    i = idx_of(st, s["grade"], s["quarter"])
    cur, past = st[i], st[:i]
    rng = random.Random(len(s["history"]))
    # 19.10: ОЧЕНЬ много заданий — но только для ЭТОЙ четверти (по видам и числам из книги), после — удаляем
    from server.book_augment import augment
    all_test_q = {x["q"] for p in st for x in p["test"]}
    extra = augment(cur, seed=len(s["history"]), banned_q=all_test_q)
    review = [x for p in past for x in p["train"] if x["q"] not in all_test_q]
    review = rng.sample(review, min(len(review), max(30, int(0.3 * (len(cur["train"]) + len(extra)) / 0.7))))
    train = [x for x in cur["train"] if x["q"] not in all_test_q] + extra + review
    # экзамен по текущей — ещё и на ru/uz-вариантах; прошлые — до 100 спрятанных
    past_test = [x for p in past for x in p["test"]]
    past_test = rng.sample(past_test, min(100, len(past_test)))
    pairs = ROOT / "server_data/book_train.json"
    preflight(train, cur["test"] + past_test, log=log)
    pairs.write_text(json.dumps({"train": [{"q": x["q"], "a": x["a"], "kind": x["kind"], "lang": x["lang"]} for x in train],
                                 "test": [{"q": x["q"], "a": x["a"], "kind": x["kind"]} for x in cur["test"]]},
                                ensure_ascii=False))
    out = ROOT / f"server_data/models/_book_g{s['grade']}q{s['quarter']}_try{s['attempts'] + 1}.json"
    log(f"Учёба: {s['grade']} класс, {s['quarter']} четверть, попытка {s['attempts'] + 1}: "
        f"из книги {len(cur['train'])} + новых по образцу книги {len(extra)} + повторение {len(review)}, "
        f"экзамен {len(cur['test'])} + прошлые {len(past_test)}")
    kaggle_train.run(math=str(pairs), steps=STEPS, out=str(out), init=s["model"], school=False, log=log,
                     extra=["--heads", "8", "--layers", str(s.get("layers", 13)), "--math_frac", "0.7", "--lr", "3e-4",
                            "--warmup", "200", "--eval_every", "1000"])
    acc_cur, wrong = exam(str(out), cur["test"])
    pairs.unlink(missing_ok=True)   # материалы этой четверти больше не нужны — к следующей соберём заново
    acc_past, _ = exam(str(out), past_test) if past_test else (1.0, [])
    old_past, _ = exam(s["model"], past_test) if past_test else (1.0, [])
    passed = acc_cur >= PASS and acc_past >= KEEP
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M"), "grade": s["grade"], "quarter": s["quarter"],
           "attempt": s["attempts"] + 1, "acc_cur": round(acc_cur, 3), "acc_past": round(acc_past, 3),
           "passed": passed, "model": str(out), "wrong": wrong}
    s["history"].append(rec)
    if acc_past >= old_past - 0.02:          # прошлое не забыла → берём модель как текущую
        s["model"] = str(out)
    if passed:
        nxt = st[i + 1] if i + 1 < len(st) else None
        s["attempts"] = 0
        if nxt:
            s["grade"], s["quarter"] = nxt["grade"], nxt["quarter"]
        else:
            s["done"] = True
    else:
        s["attempts"] += 1
    STATE.write_text(json.dumps(s, ensure_ascii=False, indent=1))
    log(f"Экзамен: текущая четверть {acc_cur:.0%}, прошлые {acc_past:.0%} → {'СДАЛА' if passed else 'не сдала, повтор'}")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status:
        s = load_state()
        print(json.dumps({k: v for k, v in s.items() if k != "history"}, ensure_ascii=False))
        for h in s["history"][-5:]:
            print(h["ts"], f"{h['grade']} кл {h['quarter']} четв, попытка {h['attempt']}: {h['acc_cur']:.0%} / "
                  f"прошлые {h['acc_past']:.0%}", "СДАЛА" if h["passed"] else "")
        return
    step()


if __name__ == "__main__":
    main()
