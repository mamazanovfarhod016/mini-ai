"""GSM8K (OpenAI, MIT) + русский перевод gsm8k-ru (d0rj, MIT) → задачи со словами по ступеням (19.10.2026, Фарход «Да»).

    .venv/bin/python tools/gsm8k_tasks.py    # → ~/Developer/mini-ai-books/gsm8k/levels.json + сводка

• Решения уже по шагам (так их писали люди). Убираем служебные <<48/2=24>>, в конце «Ответ: N».
• ПРОВЕРКА: каждое равенство «a op b = c» в решении пересчитывает Python; задачи с ошибкой в решении выкидываем.
• Ступень = число шагов решения (больше шагов — труднее) и размер чисел:
      2 шага, числа ≤ 100   → 3 класс       3 шага → 4 класс      4 шага → 5 класс
      5 шагов → 6 класс      6+ шагов → 7 класс
  Внутри класса — по четвертям по возрастанию трудности.
• Экзамен — официальный test GSM8K (модель его не видит никогда), учёба — train.
Узбекского перевода с проверенной лицензией нет — пока только ru/en.
"""
import json
import re
from fractions import Fraction
from pathlib import Path

import pyarrow.parquet as pq

D = Path.home() / "Developer/mini-ai-books/gsm8k"
KW = {"ru": "Ответ", "en": "Answer"}
EQ = re.compile(r"<<([^=<>]+)=([^<>]+)>>")


def _val(s):
    s = s.replace(",", "").replace("$", "").strip()
    try:
        return Fraction(s) if re.fullmatch(r"-?\d+(\.\d+)?(/\d+)?", s) else Fraction(eval(s, {"__builtins__": {}}))  # noqa: S307
    except Exception:
        return None


def convert(q, a, lang):
    body, _, final = a.partition("####")
    final = final.strip().replace(",", "")
    steps, ok = [], True
    for ln in body.strip().splitlines():
        for expr, res in EQ.findall(ln):
            if not re.fullmatch(r"[\d.+\-*/() ]+", expr):
                ok = False
                continue
            l, r = _val(expr), _val(res)
            if l is None or r is None or l != r:
                ok = False
        ln = EQ.sub("", ln).strip()
        if ln:
            steps.append(ln)
    if not ok or not re.fullmatch(r"-?\d+", final):
        return None
    n_steps = len(EQ.findall(a))
    big = max([int(x) for x in re.findall(r"\d+", q)] or [0])
    return {"q": " ".join(q.split()), "a": " ".join(steps) + f" {KW[lang]}: {final}", "answer": final, "lang": lang,
            "steps": n_steps, "big": big}


def level(x):
    s = x["steps"]
    grade = 3 if s <= 2 else min(7, s + 1)
    if s <= 2 and x["big"] > 100:
        grade = 4
    return grade


def main():
    out = {"train": [], "test": []}
    dropped = 0
    for split in ("train", "test"):
        for lang in ("en", "ru"):
            for r in pq.read_table(D / f"{lang}_{split}.parquet").to_pylist():
                x = convert(r["question"], r["answer"], lang)
                if x is None:
                    dropped += 1
                    continue
                x["grade"] = level(x)
                out[split].append(x)
    for split in out:
        by = {}
        for x in out[split]:
            by.setdefault(x["grade"], []).append(x)
        for g, xs in by.items():
            xs.sort(key=lambda x: (x["steps"], x["big"], len(x["q"])))
            for k, x in enumerate(xs):
                x["quarter"] = 1 + (4 * k) // len(xs)
    (D / "levels.json").write_text(json.dumps(out, ensure_ascii=False))
    print(f"выкинуто задач с ошибкой в решении или нечисловым ответом: {dropped}")
    for g in range(3, 8):
        tr = sum(x["grade"] == g for x in out["train"])
        te = sum(x["grade"] == g for x in out["test"])
        print(f"{g} класс: учить {tr}, экзамен {te}")


if __name__ == "__main__":
    main()
