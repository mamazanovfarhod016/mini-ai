"""exp69 — ОДНА МАТЕРИЯ (analogy.py) против списка правил (understand.py). Учитель даёт только РЕШЁННЫЕ ПРИМЕРЫ с именами —
никаких «формула», «обратное к», «любое сравнение переносится». Обобщение на «тяжелее/сильнее/дешевле/ниже» должно прийти
из смысла слов (meaning_ru) + формы. Те же 15 held-out задач exp67b, написанные до кода. Число = решено из 15; рядом контроль
без примеров и вариант без базы смысла (чтобы видеть, что даёт именно смысл)."""
import sys, os, re, time; sys.path.insert(0, ".")
from mini_ai.analogy import Analogy
from tools.exp67b_heldout import T

EX = ["смотри: Диёр старше Сардора на 2, Сардор старше Камрона на 4 — значит Диёр старше Камрона на 6",
      "Риштан в Ферганской области, Ферганская область в Узбекистане — значит Риштан в Узбекистане",
      "если Нигора младше Лолы на 1, значит Лола старше Нигоры на 1", "если Лола старше Нигоры на 1, значит Нигора младше Лолы на 1",
      "если Малика выше Севары на 3, значит Севара ниже Малики на 3", "если Севара ниже Малики на 3, значит Малика выше Севары на 3",
      "если самса дороже чая на 1000, значит чай дешевле самсы на 1000", "если чай дешевле самсы на 1000, значит самса дороже чая на 1000",
      "Велик едет 15 км/ч, до парка 6 км — значит ехать 0.4 ч",
      "У меня было 250000 сум, потратил 40000 — значит осталось 210000 сум",
      "высокая — то же, что выше", "младший — то же, что младше", "отдал — то же, что потратил", "есть — то же, что было"]

def run(A, label):
    ok = 0; miss = []
    for t, gold in T:
        r = A.solve(t); a = str(r["answer"]) if r else None
        hit = a is not None and (gold.lower() in a.lower() or a.lower() in gold.lower() or (re.fullmatch(r"[\d.]+", gold) and a and abs(float(re.sub(r"[^\d.]", "", a) or "nan") - float(gold)) < 0.02))
        ok += bool(hit); (None if hit else miss.append(f"  ✗ {t[:60]} → {a}  (эталон {gold})"))
    print(f"[{label}] {ok}/15"); print("\n".join(miss)); return ok

def meaning():
    from mini_ai.meaning import Meaning
    from mini_ai.russian import RU_BASE_NPZ
    M = Meaning(None); M.load(RU_BASE_NPZ); return M

if __name__ == "__main__":
    print("=== контроль: ни одного примера ==="); c = run(Analogy(None, None, None), "control")
    print("\n=== 11 примеров + 4 слова, БЕЗ базы смысла (только форма + сравнительная степень) ===")
    A0 = Analogy(None, None, None)
    for e in EX: print("  учитель:", e, "\n  она:    ", A0.tell(e)[1])
    n0 = run(A0, "no-meaning")
    print("\n=== 11 примеров + 4 слова + СМЫСЛ слов (meaning_ru) ===")
    t0 = time.time(); M = meaning(); print(f"  meaning loaded {time.time()-t0:.0f}s")
    A1 = Analogy(None, None, M)
    for e in EX: A1.tell(e)
    n1 = run(A1, "meaning")
    print(f"\nEXP69: контроль {c}/15 · 11 примеров + 4 слова без смысла {n0}/15 · 11 примеров + 4 слова + смысл {n1}/15 | {A1.report()}")
    r = A1.solve(T[5][0]); print("\nВслух:", A1.say_steps(r) if r else None)
