"""
Инструмент наставника (запускает ассистент-наставник по cron или вручную).

    python3 -m server.mentor inbox                 # новые вопросы (детерминированный вывод, для cron-монитора)
    python3 -m server.mentor answer 3 "Париж — столица Франции."   # ответить (+ в очередь обучения)
    python3 -m server.mentor answer 3 "..." --no-teach            # ответить, но не учить на этом
    python3 -m server.mentor skip 3                # отклонить (спам/личное)
"""
import argparse
import sys
from pathlib import Path

from server import store

DATA = Path(__file__).resolve().parent.parent / "server_data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATA))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("inbox")
    a = sub.add_parser("answer"); a.add_argument("id", type=int); a.add_argument("text"); a.add_argument("--no-teach", action="store_true")
    s = sub.add_parser("skip"); s.add_argument("id", type=int)
    args = ap.parse_args()
    data = Path(args.data)

    if args.cmd == "inbox":
        items = [i for i in store.read_json(data / "mentor_inbox.json", []) if i["status"] == "new"]
        for i in items:
            web = i.get("web") or {}
            found = f"интернет: «{web.get('answer')}» ({web.get('source')})" if web.get("found") else "интернет: не нашла"
            print(f"#{i['id']} [{i['lang']}] {i['ts']}\n  вопрос: {i['q']}\n  модель ответила: {i.get('a') or '—'}\n  {found}")
        return
    if args.cmd == "answer":
        text = args.text.strip()
        if not text or len(text) > 200:
            sys.exit("ответ должен быть 1–200 символов (модель маленькая, длинное не выучит)")
        it = store.answer_mentor_item(data, args.id, text, teach=not args.no_teach)
        print(f"#{it['id']} отвечен{' и добавлен в очередь обучения' if not args.no_teach else ''}: «{it['q']}» → «{text}»")
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
