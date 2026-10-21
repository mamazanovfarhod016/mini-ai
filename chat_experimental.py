#!/usr/bin/env python3
"""Mini AI — экспериментальная модель (память без градиента). Чат для проверки руками.

  python3 chat_experimental.py            # учебник (SciQ), ~10 с загрузки
  python3 chat_experimental.py --wiki     # + Simple Wikipedia (300k фактов, ~30 с)

Команды в чате:
  любой вопрос              → ответ по фактам + цепочка-источник (или честное «не знаю»)
  /tell X                   → рассказать про X своими словами
  /learn <текст>            → выучить текст прямо сейчас (сразу можно спрашивать)
  /do <задача>              → действие инструментами (shell/файлы/wiki)
  /code <задача> | f | тесты→ написать функцию; тесты как JSON: [[[args], expected], ...]
  /facts X                  → показать сырые факты про X
  /stats                    → размер памяти
  /quit
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mini_ai.experimental import ExperimentalMind

wiki = "--wiki" in sys.argv
print("загрузка…", flush=True)
m = ExperimentalMind(school=True, wiki=wiki)
print(f"готово: {m.boot['facts']} фактов за {m.boot['seconds']} с; источники {m.boot['sources']}")
print("спрашивай (help: /help)\n")

while True:
    try: line = input("ты> ").strip()
    except (EOFError, KeyboardInterrupt): print(); break
    if not line: continue
    if line in ("/quit", "/exit"): break
    if line == "/help": print(__doc__); continue
    if line == "/stats": print(f"  фактов {len(m.kb.facts)}, слов {len(m.kb.df)}, источники {dict(m.kb.sources.most_common(6))}, история {len(m.history)}"); continue
    if line.startswith("/tell "):
        r = m.tell(line[6:]); print(f"  [{r['how']}, фактов {r['facts']}]\n  {r['text']}"); continue
    if line.startswith("/learn "):
        r = m.learn(line[7:]); print(f"  выучено: +{r['new_facts']} фактов (всего {r['total']})"); continue
    if line.startswith("/facts "):
        key = " ".join(line[7:].lower().split()); ks = m.kb.by_x.get(key, [])[:15]
        for k in ks: f = m.kb.facts[k]; print(f"  {f[3]} —{f[1]}→ {f[4]}   [{f[5]}]")
        if not ks: print("  нет фактов с таким субъектом")
        continue
    if line.startswith("/do "):
        r = m.act(line[4:]); print(f"  [{r['how']}]\n  {r['output']}"); continue
    if line.startswith("/code "):
        try:
            task, fname, tests = [p.strip() for p in line[6:].split("|")]; tests = json.loads(tests)
        except Exception: print("  формат: /code задача | имя_функции | [[[аргументы], ожидаемое], ...]"); continue
        r = m.code(task, fname, tests)
        print(f"  [{r['status']}, проб {r['tried']}" + (f", из {'+'.join(r['parts'])}]\n" + r['code'] if r['status'] == 'ok' else "]")); continue
    r = m.ask(line)
    if r["answer"]:
        print(f"  → {r['answer']}   [{r['how']}, уверенность {r['confidence']}]")
        for f in r["facts"]: print(f"     • {f}")
    else:
        print(f"  → не знаю. (разбор: slot={r['frame']['slot']} X={r['frame']['X']} rel={r['frame']['rel']} Y={r['frame']['Y']})\n     подскажи: /learn <предложение> — и спроси снова")
