"""
Mini AI — сервер самообучения (только стандартная библиотека Python).

    python3 -m server.app --port 8787 --data server_data

Эндпоинты (JSON, CORS открыт — приложение работает из file:// внутри WebView):
  POST /api/log          {q, a, ok, lang, device}      — каждый вопрос из приложения → questions.jsonl
  POST /api/research     {q, lang}                     — поиск ответа в интернете (server/research.py)
  POST /api/confirm      {q, a, web, lang}             — «Верно» под ответом из интернета → наставнику на обобщение
  GET  /api/progress                                   — история экзаменов (график «Как я умнею»)
  POST /api/mentor       {q, a, web, lang, device}     — «отправить наставнику» → mentor_inbox.jsonl
  GET  /api/mentor/answers?ids=1,2                     — ответы наставника для приложения
  GET  /api/model/latest                               — {version, url, test_intent_acc}
  GET  /models/<file>.json                             — сами веса
  GET  /health

Все данные (вопросы, очередь наставника) лежат в server_data/ — она в .gitignore (личное).
"""
import argparse
import json
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from server import research as research_mod
from server import store

MAX_BODY = 16 * 1024


class Handler(BaseHTTPRequestHandler):
    server_version = "MiniAI/0.3"
    data: Path = None
    lock = threading.Lock()

    def log_message(self, fmt, *args):  # тише в логе: только метод/путь/код
        print(time.strftime("%H:%M:%S"), self.command, self.path.split("?")[0], args[1] if len(args) > 1 else "")

    def _send(self, code, obj=None, body=None, ctype="application/json; charset=utf-8"):
        data = body if body is not None else json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _json_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_BODY:
            raise ValueError("bad body size")
        obj = json.loads(self.rfile.read(n).decode("utf-8"))
        if not isinstance(obj, dict):
            raise ValueError("body must be an object")
        return obj

    def do_OPTIONS(self):
        self._send(204, body=b"")

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/health":
            return self._send(200, {"ok": True})
        if u.path == "/api/progress":
            return self._send(200, store.read_json(self.data / "progress.json", []))
        if u.path == "/api/model/latest":
            latest = store.read_json(self.data / "models/latest.json", None)
            return self._send(200, latest or {"version": 0})
        if u.path.startswith("/models/"):
            name = u.path.rsplit("/", 1)[-1]
            f = self.data / "models" / name
            if "/" in name or ".." in name or not name.endswith(".json") or not f.exists():
                return self._send(404, {"error": "not found"})
            return self._send(200, body=f.read_bytes())
        if u.path == "/api/mentor/answers":
            ids = urllib.parse.parse_qs(u.query).get("ids", [""])[0]
            want = {int(x) for x in ids.split(",") if x.strip().isdigit()}
            answers = store.read_json(self.data / "mentor_answers.json", {})
            return self._send(200, {k: v for k, v in answers.items() if int(k) in want})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        try:
            body = self._json_body()
        except Exception as e:
            return self._send(400, {"error": str(e)})
        q = str(body.get("q", "")).strip()[:300]
        if not q:
            return self._send(400, {"error": "q is required"})
        lang = "uz" if body.get("lang") == "uz" else "ru"
        rec = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "q": q, "lang": lang,
               "a": str(body.get("a", ""))[:500], "device": str(body.get("device", ""))[:40]}
        if u.path == "/api/log":
            rec["ok"] = bool(body.get("ok"))
            with self.lock:
                store.append_jsonl(self.data / "questions.jsonl", rec)
            return self._send(200, {"ok": True})
        if u.path == "/api/research":
            res = research_mod.research(q, lang)
            with self.lock:
                store.append_jsonl(self.data / "research.jsonl", {**rec, "result": res})
            return self._send(200, res)
        if u.path == "/api/confirm":
            # пользователь нажал «Верно» под ответом из интернета → наставник превратит его в тему
            web = body.get("web") if isinstance(body.get("web"), dict) else {}
            with self.lock:
                item = store.add_mentor_item(self.data, {**rec, "kind": "confirm", "web": web})
            return self._send(200, {"ok": True, "id": item["id"]})
        if u.path == "/api/mentor":
            web = body.get("web") if isinstance(body.get("web"), dict) else None
            with self.lock:
                item = store.add_mentor_item(self.data, {**rec, "kind": "question", "web": web})
            return self._send(200, {"ok": True, "id": item["id"]})
        return self._send(404, {"error": "not found"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parent.parent / "server_data"))
    args = ap.parse_args()
    Handler.data = Path(args.data)
    (Handler.data / "models").mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Mini AI server на http://{args.host}:{args.port}, данные: {Handler.data}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
