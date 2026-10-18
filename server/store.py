"""Файловое хранилище сервера Mini AI (JSON/JSONL в server_data/, атомарная запись)."""
import json
import os
import time
from pathlib import Path


def read_json(path, default):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def append_jsonl(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_jsonl(path):
    try:
        with open(path, encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        return []


def add_mentor_item(data_dir, rec):
    """Новый вопрос наставнику: id по порядку, статус new."""
    inbox = Path(data_dir) / "mentor_inbox.json"
    items = read_json(inbox, [])
    item = {"id": (max([i["id"] for i in items]) + 1) if items else 1, "status": "new", **rec}
    items.append(item)
    write_json(inbox, items)
    return item


def answer_mentor_item(data_dir, item_id, answer, teach=True):
    """Наставник ответил: ответ уходит в приложение и (если teach) — в очередь обучения."""
    data_dir = Path(data_dir)
    inbox = data_dir / "mentor_inbox.json"
    items = read_json(inbox, [])
    item = next((i for i in items if i["id"] == item_id), None)
    if item is None:
        raise KeyError(f"нет вопроса #{item_id}")
    item["status"] = "answered"
    item["answer"] = answer
    item["answered_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(inbox, items)
    answers = read_json(data_dir / "mentor_answers.json", {})
    answers[str(item_id)] = {"q": item["q"], "answer": answer, "at": item["answered_at"]}
    write_json(data_dir / "mentor_answers.json", answers)
    if teach:
        queue = read_json(data_dir / "teach_queue.json", [])
        queue.append({"q": item["q"], "a": answer, "from": f"mentor#{item_id}"})
        write_json(data_dir / "teach_queue.json", queue)
    return item
