"""Эксперимент 25: АГЕНТ, КОТОРЫЙ УЧИТСЯ ПРИЁМАМ (саморазвитие без градиента).
Фаза 1 — обучение: 8 задач с проверкой. Приёма нет → агент перебирает генофонд (13 одиночных + 156 пар),
  выполняет, проверяет; прошедший становится приёмом.
Фаза 2 — перенос: те же типы задач, но ДРУГИЕ слова, пути, числа, темы, и ДРУГАЯ формулировка.
  Успех = решил из памяти приёмов (без перебора). Цифра: доля решённых + сколько взято «из памяти».
Фаза 3 — перезапуск: новый процесс грузит procs.json → всё умеет сразу (не забывает).
"""
import sys, os, json, time, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_ai.agent import Agent

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
HOME = os.path.join(OUT, "exp25_agent"); shutil.rmtree(HOME, ignore_errors=True)
W = os.path.join(HOME, "work"); os.makedirs(W, exist_ok=True)
open(f"{W}/a.txt", "w").write("alpha\nbeta\ngamma\nbeta\n")
open(f"{W}/b.txt", "w").write("one two three\nfour five\n")

TRAIN = [
    (f"create file {W}/hello.txt with text «Hello world»", lambda o: open(f"{W}/hello.txt").read() == "Hello world"),
    (f"how many lines in {W}/a.txt", lambda o: o.strip() == "4"),
    (f"how many words in {W}/b.txt", lambda o: o.strip() == "5"),
    ("calculate 17*3+4", lambda o: o.strip() == "55"),
    (f"count 'beta' in {W}/a.txt", lambda o: o.strip() == "2"),
    (f"copy {W}/a.txt to {W}/a2.txt", lambda o: os.path.exists(f"{W}/a2.txt")),
    ("read wikipedia about Bukhara and tell what is Bukhara", lambda o: "bukhara" in o.lower() and len(o) > 30),
    (f"show first 2 lines of {W}/a.txt", lambda o: o.strip() == "alpha\nbeta"),
]
TRANSFER = [   # другая формулировка, другие данные
    (f"make a file {W}/note.md containing «todo: ship v3»", lambda o: open(f"{W}/note.md").read() == "todo: ship v3"),
    (f"number of lines in {W}/b.txt", lambda o: o.strip() == "2"),
    (f"word count for {W}/a.txt", lambda o: o.strip() == "4"),
    ("compute 100-58", lambda o: o.strip() == "42"),
    (f"how many times 'one' appears in {W}/b.txt", lambda o: o.strip() == "1"),
    (f"copy {W}/b.txt to {W}/b2.txt", lambda o: os.path.exists(f"{W}/b2.txt")),
    ("read wikipedia about Khiva and tell what is Khiva", lambda o: "khiva" in o.lower() and len(o) > 30),
    (f"print first 1 lines of {W}/b.txt", lambda o: o.strip() == "one two three"),
]


def phase(agent, tasks, name, allow_learning=True):
    ok = from_mem = 0; t0 = time.time()
    print(f"--- {name} ---")
    for req, check in tasks:
        out, how = agent.run(req, check, learn=allow_learning)
        good = out is not None and check(out)
        ok += good; from_mem += how == "память приёмов"
        print(f"  {'OK ' if good else 'FAIL'} {req[:60]:60} ← {how[:70]}")
    print(f"  {name}: {ok}/{len(tasks)} решено, {from_mem} из памяти приёмов, {time.time()-t0:.1f}s")
    return ok, from_mem


a = Agent(HOME)
r1 = phase(a, TRAIN, "ФАЗА 1 обучение (перебор генофонда)")
a.save()
r2 = phase(a, TRANSFER, "ФАЗА 2 перенос на новые формулировки (учиться разрешено)")
a.save()
b = Agent(HOME)   # новый процесс — грузит с диска
print(f"перезапуск: загружено приёмов {len(b.procs)}, знаний {len(b.mem.L.stext)} предл.")
r3 = phase(b, TRANSFER, "ФАЗА 3 после перезапуска, БЕЗ права учиться (только память)", allow_learning=False)
json.dump(dict(train=r1, transfer=r2, restart=r3, procs=[{k: v for k, v in p.items() if k != "vec"} for p in b.procs]),
          open(f"{OUT}/exp25_agent.json", "w"), ensure_ascii=False, indent=1)
