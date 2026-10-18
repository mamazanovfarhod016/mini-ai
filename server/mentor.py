"""
Инструмент наставника (запускает ассистент-наставник по cron или вручную).

    python3 -m server.mentor inbox                 # новые вопросы (детерминированный вывод, для cron-монитора)
    python3 -m server.mentor topics                # уже выученные темы (чтобы добавлять в существующую)
    python3 -m server.mentor teach 3 --ru "Париж — столица Франции." --uz "Fransiya poytaxti — Parij." \
        --para "главный город франции" --para "какой город столица франции" ... (минимум 8 разных)
        [--topic k_stolica_francii]                # добавить в существующую тему
    python3 -m server.mentor skip 3                # отклонить (личное без ответа / спам)

Почему 8+ формулировок: модель должна понять СМЫСЛ. Часть формулировок прячется
для экзамена (server/knowledge.py) — знание засчитывается, только если модель
отвечает на те, которых не видела.
"""
import argparse
import sys
from pathlib import Path

from server import knowledge, store

DATA = Path(__file__).resolve().parent.parent / "server_data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inbox")
    sub.add_parser("topics")
    t = sub.add_parser("teach")
    t.add_argument("id", type=int)
    t.add_argument("--ru"); t.add_argument("--uz")
    t.add_argument("--para", action="append", default=[])
    t.add_argument("--topic")
    s = sub.add_parser("skip"); s.add_argument("id", type=int)
    args = ap.parse_args()
    data = Path(args.data)

    if args.cmd == "inbox":
        from server import mathcheck
        all_items = store.read_json(data / "mentor_inbox.json", [])
        for i in all_items:                      # примеры проверяет скрипт — ИИ-наставника не будим
            if i["status"] == "new" and mathcheck.parse_math(i["q"]):
                i["status"] = "skipped_math"
        store.write_json(data / "mentor_inbox.json", all_items)
        items = [i for i in all_items if i["status"] == "new"]
        for i in items:
            web = i.get("web") or {}
            if i.get("kind") == "confirm":
                print(f"#{i['id']} [{i['lang']}] ПОДТВЕРЖДЕНО пользователем (ответ из интернета)\n"
                      f"  вопрос: {i['q']}\n  ответ: {i.get('a')}\n  источник: {web.get('url') or '—'}")
            else:
                found = f"интернет: «{web.get('answer')}» ({web.get('url')})" if web.get("found") else "интернет: не нашла"
                print(f"#{i['id']} [{i['lang']}] ВОПРОС НАСТАВНИКУ\n  вопрос: {i['q']}\n"
                      f"  модель ответила: {i.get('a') or '—'}\n  {found}")
        return
    if args.cmd == "topics":
        for tp in knowledge.load(data):
            ans = (tp["responses"].get("ru") or tp["responses"].get("uz") or [""])[0]
            print(f"{tp['tag']}: {tp['description']} → {ans}  ({len(tp['train'])} обуч. + {len(tp['heldout'])} экзамен.)")
        return
    if args.cmd == "teach":
        for a in (args.ru, args.uz):
            if a and len(a) > 160:
                sys.exit("ответ до 160 символов — маленькая модель длинное не выучит")
        items = store.read_json(data / "mentor_inbox.json", [])
        item = next((i for i in items if i["id"] == args.id), None)
        if not item:
            sys.exit(f"нет вопроса #{args.id}")
        try:
            tp, created = knowledge.add_or_merge(data, item["q"], args.para, args.ru, args.uz, args.topic,
                                                 source=item.get("kind", "question"))
        except ValueError as e:
            sys.exit(str(e))
        answer = args.ru if (item["lang"] == "ru" and args.ru) else (args.uz or args.ru)
        store.answer_mentor_item(data, args.id, knowledge.clean_answer(answer), teach=False)
        print(f"#{args.id} → тема {tp['tag']} ({'новая' if created else 'дополнена'}): "
              f"{len(tp['train'])} формулировок для обучения, {len(tp['heldout'])} спрятано для экзамена")
        return
    if args.cmd == "skip":
        items = store.read_json(data / "mentor_inbox.json", [])
        for i in items:
            if i["id"] == args.id:
                i["status"] = "skipped"
        store.write_json(data / "mentor_inbox.json", items)
        print(f"#{args.id} отклонён")


if __name__ == "__main__":
    main()
