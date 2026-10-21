"""Эксперимент 28: САМОРАЗВИТИЕ 2-го ПОРЯДКА — агент сам расширяет генофонд из прочитанной документации.
Задачи, которых НЕТ в генофонде (13 генов): последние N строк, сортировка, уникальные строки, размер файла,
перевернуть строку, md5. Шаг 1: агент пробует → не может. Шаг 2: читает шпаргалку по shell (как человек
читает man / StackOverflow) → learn_genes_from_text вынимает шаблоны команд. Шаг 3: те же задачи → перебор
уже по расширенному генофонду. Шаг 4: перезапуск — гены и приёмы на диске.
Успех: 0/6 → ≥5/6; после перезапуска из памяти приёмов ≥5/6.
"""
import sys, os, json, time, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_ai.agent import Agent

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
HOME = os.path.join(OUT, "exp28_agent"); shutil.rmtree(HOME, ignore_errors=True)
W = os.path.join(HOME, "work"); os.makedirs(W, exist_ok=True)
open(f"{W}/a.txt", "w").write("pear\napple\nfig\napple\nkiwi\n")
open(f"{W}/b.txt", "w").write("hello\n")

DOC = """
Shell cheat-sheet (примеры команд, как в любом учебнике):
$ tail -n 2 notes.txt        # last lines of a file
$ sort data.txt              # sort lines
$ sort -u data.txt           # unique sorted lines
$ wc -c < file.txt           # size in bytes
$ rev word.txt               # reverse each line
$ md5sum report.txt          # checksum
$ tac log.txt                # lines in reverse order
$ nl file.txt                # number lines
"""

TASKS = [
    (f"show last 2 lines of {W}/a.txt", lambda o: o.strip() == "apple\nkiwi"),
    (f"sort lines of {W}/a.txt", lambda o: o.strip() == "apple\napple\nfig\nkiwi\npear"),
    (f"unique sorted lines of {W}/a.txt", lambda o: o.strip() == "apple\nfig\nkiwi\npear"),
    (f"size in bytes of {W}/b.txt", lambda o: o.strip() == "6"),
    (f"reverse the line in {W}/b.txt", lambda o: o.strip() == "olleh"),
    (f"lines of {W}/a.txt in reverse order", lambda o: o.strip() == "kiwi\napple\nfig\napple\npear"),
]


def phase(agent, name, learn=True):
    ok = mem = 0; t0 = time.time(); print(f"--- {name} (генов {len(agent.genes)}) ---")
    for req, check in TASKS:
        out, how = agent.run(req, check, learn=learn)
        good = out is not None and check(out); ok += good; mem += how == "память приёмов"
        print(f"  {'OK ' if good else 'FAIL'} {req.replace(W, '…')[:45]:45} ← {how[:60]}")
    print(f"  {name}: {ok}/{len(TASKS)}, из памяти {mem}, {time.time()-t0:.1f}s"); return ok, mem


a = Agent(HOME)
r1 = phase(a, "ШАГ 1: исходный генофонд")
n = a.learn_genes_from_text(DOC)
print(f"ШАГ 2: прочитал шпаргалку → новых генов {n}: {[g[1] for g in a.genes[len(a.genes)-n:]]}")
r3 = phase(a, "ШАГ 3: после чтения")
a.save()
b = Agent(HOME)
r4 = phase(b, "ШАГ 4: перезапуск, без права учиться", learn=False)
json.dump(dict(before=r1, after=r3, restart=r4, new_genes=n), open(f"{OUT}/exp28_self_genes.json", "w"), indent=1)
