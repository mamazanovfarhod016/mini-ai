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
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BOOKS = Path.home() / "Developer/mini-ai-books"
STATE = ROOT / "server_data/book_school.json"
PASS, KEEP, FINAL = 0.90, 0.85, 0.95
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
        if x.get("kind") == "gsm8k":      # задачи со словами: каждое равенство решения уже пересчитано при загрузке
            continue
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


def finish_grade(grade, model, log=print):
    """Класс сдан на ≥95%: бенчмарки (server.benchmarks) → результаты в отчёт → материалы класса удаляем.
    Удаляем: PDF учебников класса, текст уроков, задания для учёбы. Оставляем: (1) спрятанные экзамены —
    чтобы проверять, что не забыла; (2) маленький набор повторения ≤300 заданий — исследование (2403.08763):
    без повторения старого сеть забывает, а повторять больше нечего, если удалить всё."""
    from server import benchmarks
    rec = benchmarks.run(model, label=f"после {grade} класса")
    summ = "; ".join(f"{k} {v['acc'] * 100:.0f}%" for k, v in rec["results"].items())
    log(f"БЕНЧМАРК после {grade} класса: {summ}")
    (ROOT / "server_data/benchmark_report.txt").write_text(f"{grade} класс сдан. {summ}\n", encoding="utf-8")
    st = stages()
    rng = random.Random(grade)
    pool = [x for p in st if p["grade"] == grade for x in p["train"]]
    keep = rng.sample(pool, min(300, len(pool)))
    for p in st:
        if p["grade"] == grade:
            p["train"] = [x for x in keep if x["quarter"] == p["quarter"]]
    (BOOKS / "tasks.json").write_text(json.dumps(st, ensure_ascii=False))
    removed = 0
    for f in (BOOKS / "pdf").glob(f"{grade:02d}_*.pdf"):
        f.unlink()
        removed += 1
    ls_path = BOOKS / "lessons.json"
    if ls_path.exists():
        ls = [x for x in json.loads(ls_path.read_text()) if x["grade"] != grade]
        ls_path.write_text(json.dumps(ls, ensure_ascii=False))
    log(f"Материалы {grade} класса удалены: учебников {removed}, заданий {len(pool) - len(keep)} "
        f"(оставлено {len(keep)} для повторения и все экзамены)")


def _tens(x):
    """Задание с переходом через десяток или с 10 в ответе/числах."""
    import re
    nums = [int(n) for n in re.findall(r"\d+", x["q"])]
    try:
        ans = int(str(x.get("answer", "")).replace("−", "-"))
    except ValueError:
        ans = None
    return 10 in nums or ans == 10 or (len(nums) == 2 and "+" in x["q"] and nums[0] < 10 and nums[1] < 10 and sum(nums) >= 10)


def _div(x):
    """Задание с делением (20.10: 2 кл. 3 четв. — первое знакомство с «/», 3 попытки подряд 33%,
    тот же пример «45 / 9» каждый раз неверный — модели не хватило повторений нового действия)."""
    return "/" in x["q"] and x.get("kind") != "gsm8k"


def step(log=print):
    from server import kaggle_train
    st = stages()
    s = load_state()
    i = idx_of(st, s["grade"], s["quarter"])
    cur, past = st[i], st[:i]
    if s.get("review_grade"):     # не сдала итог класса → учим все четверти класса вместе
        g = s["review_grade"]
        qs = [p for p in st if p["grade"] == g]
        cur = {"grade": g, "quarter": qs[-1]["quarter"], "train": [x for p in qs for x in p["train"]],
               "test": [x for p in qs for x in p["test"]]}
        past = [p for p in st if p["grade"] < g]
    # 19.10 (вечер): продолжаем с ЛУЧШЕЙ попытки этой четверти (не с последней) — иначе откат 82% → 74–76%
    same = [h for h in s["history"] if h["grade"] == s["grade"] and h["quarter"] == s["quarter"]
            and h.get("acc_past", 0) >= KEEP - 0.05 and Path(h["model"]).exists()]
    best = max(same, key=lambda h: h["acc_cur"] + h.get("acc_past", 0)) if same else None
    init = best["model"] if best else s["model"]
    if init != s["model"]:
        log(f"Старт с лучшей попытки: {Path(init).name} (текущая {best['acc_cur']:.0%}, прошлые {best.get('acc_past', 0):.0%})")
    rng = random.Random(len(s["history"]))
    # 19.10: ОЧЕНЬ много заданий — но только для ЭТОЙ четверти (по видам и числам из книги), после — удаляем
    from server.book_augment import augment
    all_test_q = {x["q"] for p in st for x in p["test"]}
    extra = augment(cur, seed=len(s["history"]), banned_q=all_test_q)
    review = [x for p in past for x in p["train"] if x["q"] not in all_test_q]
    # 19.10 (ночь): 2-я четверть сдана на 100%, но 1-ю забыла (82% < 85%) — повторения было всего 71.
    # Добавляем задания ПО ОБРАЗЦУ прошлых четвертей (те же виды и числа из книги), чтобы повторение было ~30%.
    want = max(30, int(0.3 * (len(cur["train"]) + len(extra)) / 0.7))
    if past and len(review) < want:
        per = max(50, (want - len(review)) // len(past) + 1)
        for j, p in enumerate(past):
            review += augment(p, factor=100, cap=per, seed=1000 + j + len(s["history"]), banned_q=all_test_q)
    review = rng.sample(review, min(len(review), want))
    train = [x for x in cur["train"] if x["q"] not in all_test_q] + extra + review
    # 19.10 (вечер): ошибки — на переходе через 10 (7 + 3, 8 + 2, 10 − 4): такие задания повторяем ×4
    hard = [x for x in train if x.get("kind") != "gsm8k" and _tens(x)]
    train += hard * 3
    if hard:
        log(f"Переход через 10: {len(hard)} заданий ×4")
    # 20.10: деление — новое действие этой четверти, 3 попытки подряд 33% (один и тот же пример
    # «45 / 9» каждый раз неверный) → повторяем деление ×6, как переход через 10 помог прошлой четверти
    div = [x for x in train if _div(x)]
    train += div * 5
    if div:
        log(f"Деление (новое действие): {len(div)} заданий ×6")
    # 19.10: задачи со словами GSM8K (OpenAI, MIT; ru-перевод d0rj, MIT) — с 3 класса, по ступени четверти.
    # Экзамен по ним — ОТДЕЛЬНЫЙ и НЕ блокирует переход: для 6 млн параметров это очень трудно (честный замер роста).
    gsm_train, gsm_test = [], []
    gl = BOOKS / "gsm8k/levels.json"
    if gl.exists() and s["grade"] >= 3:
        g = json.loads(gl.read_text())
        pick = lambda xs: [x for x in xs if x["grade"] == min(s["grade"], 7) and x["quarter"] <= s["quarter"]]  # noqa: E731
        gsm_train = rng.sample(pick(g["train"]), min(1500, len(pick(g["train"]))))
        gsm_test = rng.sample(pick(g["test"]), min(60, len(pick(g["test"]))))
        train += [{**x, "kind": "gsm8k"} for x in gsm_train]
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
    kaggle_train.run(math=str(pairs), steps=STEPS, out=str(out), init=init, school=False, log=log,
                     gpu=os.environ.get("MINIAI_GPU", "1") != "0",   # 20.10: лимит GPU кончился → MINIAI_GPU=0 (CPU Kaggle)
                     extra=["--heads", "8", "--layers", str(s.get("layers", 13)), "--math_frac", "0.7", "--lr", str(s.get("lr", 1.5e-4)),
                            "--warmup", "200", "--eval_every", "1000"] + (["--max_q_min", "200"] if gsm_train else []))
    acc_cur, wrong = exam(str(out), cur["test"])
    acc_gsm = round(exam(str(out), gsm_test)[0], 3) if gsm_test else None
    pairs.unlink(missing_ok=True)   # материалы этой четверти больше не нужны — к следующей соберём заново
    acc_past, _ = exam(str(out), past_test) if past_test else (1.0, [])
    old_past, _ = exam(init, past_test) if past_test else (1.0, [])
    passed = acc_cur >= PASS and acc_past >= KEEP
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M"), "grade": s["grade"], "quarter": s["quarter"],
           "attempt": s["attempts"] + 1, "acc_cur": round(acc_cur, 3), "acc_past": round(acc_past, 3),
           "passed": passed, "model": str(out), "wrong": wrong, "acc_gsm8k": acc_gsm}
    s["history"].append(rec)
    if acc_past >= old_past - 0.02:          # прошлое не забыла → берём модель как текущую
        s["model"] = str(out)
    if passed:
        nxt = st[i + 1] if i + 1 < len(st) else None
        s["attempts"] = 0
        if not nxt or nxt["grade"] != s["grade"]:
            # 19.10: класс пройден по четвертям → итоговый экзамен класса (все спрятанные задания класса).
            # ≥95% → бенчмарки, отчёт, удаление материалов класса, следующий класс. Иначе — повтор всего класса.
            final = [x for p in st if p["grade"] == s["grade"] for x in p["test"]]
            acc_final, _ = exam(str(out), final)
            rec["acc_class_final"] = round(acc_final, 3)
            log(f"ИТОГОВЫЙ экзамен {s['grade']} класса: {acc_final:.0%} (нужно 95%)")
            if acc_final >= FINAL:
                s.pop("review_grade", None)
                finish_grade(s["grade"], str(out), log)
                if nxt:
                    s["grade"], s["quarter"] = nxt["grade"], nxt["quarter"]
                else:
                    s["done"] = True
            else:
                s["review_grade"] = s["grade"]          # повтор всего класса (все четверти вместе)
        elif nxt:
            s["grade"], s["quarter"] = nxt["grade"], nxt["quarter"]
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
