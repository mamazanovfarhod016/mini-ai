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
import re
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from server import research as research_mod
from server import mathcheck, store

MAX_BODY = 16 * 1024
MAX_UPLOAD = 40 * 1024 * 1024   # модель из Colab


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

    def _colab_upload(self, u):
        """19.10.2026: модель из Google Colab → карантин server_data/colab_incoming/ (в приложение сама НЕ идёт:
        сначала экзамен, замена только если не хуже). Проверка: размер ≤ 40 МБ, валидный JSON с config/weights."""
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0 or n > MAX_UPLOAD:
            return self._send(413, {"error": "bad size"})
        raw = self.rfile.read(n)
        kind = urllib.parse.parse_qs(u.query).get("kind", ["model"])[0]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        inc = self.data / "colab_incoming"
        inc.mkdir(parents=True, exist_ok=True)
        if kind == "log":
            (inc / f"{stamp}.log.txt").write_bytes(raw[:2_000_000])
            return self._send(200, {"ok": True})
        try:
            obj = json.loads(raw.decode("utf-8"))
            assert isinstance(obj, dict) and "config" in obj
        except Exception:
            return self._send(400, {"error": "not a Mini AI model"})
        (inc / f"{stamp}.json").write_bytes(raw)
        return self._send(200, {"ok": True, "saved": f"{stamp}.json"})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/api/colab/upload":
            return self._colab_upload(u)
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
                m = mathcheck.log_answer(self.data, q, rec["a"], lang)   # пример → проверяет скрипт, не ИИ
                if m:
                    rec["ok"], rec["math"] = m["ok"], m
                store.append_jsonl(self.data / "questions.jsonl", rec)
            return self._send(200, {"ok": True, "math": m})
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
            if mathcheck.parse_math(q):            # примеры наставнику не нужны — их проверяет скрипт
                return self._send(200, {"ok": True, "id": None, "math": True})
            web = body.get("web") if isinstance(body.get("web"), dict) else None
            with self.lock:
                item = store.add_mentor_item(self.data, {**rec, "kind": "question", "web": web})
            return self._send(200, {"ok": True, "id": item["id"]})
        # ---- ЭКСПЕРИМЕНТАЛЬНАЯ МОДЕЛЬ «память без градиента» (mini_ai/experimental.py, 21.10.2026) ----
        if u.path == "/api/memory/ask":
            mind = self._memory_mind()
            text = (body.get("q") or "").strip()
            sid = str(body.get("sid") or rec.get("device") or "anon")[:64]     # нить разговора — по sid (exp40)
            with self.lock:
                mind.dialog = self._session(mind, sid)
                if text.lower().startswith(("/learn ", "выучи ", "запомни ")):
                    r = mind.learn(text.split(" ", 1)[1])
                    return self._send(200, {"answer": f"Выучила: +{r['new_facts']} фактов (всего {r['total']}).", "text": f"Выучила: +{r['new_facts']} фактов (всего {r['total']}).", "kind": "learn", "how": "learn", "facts": [], "confidence": 1.0, "sid": sid})
                me = re.match(r"^(?:example|пример)\s*[:\-]?\s*(.+?\?)\s*(?:→|->|=>|answer:|ответ:)\s*(.+?)\.?$", text.strip(), re.I)
                if me:
                    t = mind.show_example(me.group(1).strip(), me.group(2).strip()); return self._send(200, {"answer": t, "text": t, "kind": "rule", "how": "few-shot rule", "facts": [], "confidence": 1.0, "sid": sid})
                mw = re.match(r"^(?:/word\s+|what does\s+|what do you know about the word\s+|что значит\s+|что такое слово\s+)[«\"']?([\w\-]+)[»\"']?(?:\s+mean)?\??$", text.strip(), re.I)
                if mw:
                    t = mind.word(mw.group(1)); return self._send(200, {"answer": t, "text": t, "kind": "word", "how": "meaning from contexts", "facts": [], "confidence": 1.0, "sid": sid})
                if text.lower().startswith(("/tell ", "расскажи про ", "tell me about ")):
                    r = mind.tell(text.split(" ", 2 if text.lower().startswith("расскажи") else 1)[-1])
                    return self._send(200, {"answer": r["text"] or "Про это у меня нет фактов.", "text": r["text"] or "Про это у меня нет фактов.", "kind": "answer" if r["text"] else "unknown", "how": r["how"], "facts": [], "read": r.get("read", []), "confidence": 1.0 if r["text"] else 0.0, "sid": sid})
                r = mind.ask(text)
            store.append_jsonl(self.data / "memory_ask.jsonl", {**rec, "sid": sid, "answer": r.get("answer"), "kind": r.get("kind"), "how": r.get("how"), "confidence": r.get("confidence")})
            return self._send(200, {"answer": r.get("text") or r.get("answer"), "text": r.get("text"), "short": r.get("answer"), "kind": r.get("kind"), "how": r.get("how"),
                                    "facts": r.get("facts", []), "read": r.get("read", []), "confidence": r.get("confidence", 0), "frame": r.get("frame"),
                                    "note": r.get("note"), "thoughts": r.get("thoughts", []), "n_facts": len(mind.kb.facts), "sid": sid})
        return self._send(404, {"error": "not found"})

    _mind = None
    _sessions = {}            # sid → (Dialog, last_ts); нить разговора в памяти, TTL 30 мин, не больше 500 нитей
    SESSION_TTL = 30 * 60

    @classmethod
    def _session(cls, mind, sid):
        now = time.time()
        if len(cls._sessions) > 500 or (cls._sessions and now - min(t for _, t in cls._sessions.values()) > cls.SESSION_TTL):
            for k in [k for k, (_, t) in cls._sessions.items() if now - t > cls.SESSION_TTL]: cls._sessions.pop(k, None)
        d, t = cls._sessions.get(sid, (None, 0))
        if d is None or now - t > cls.SESSION_TTL:
            from mini_ai.dialog import Dialog
            d = Dialog(mind.kb)
        cls._sessions[sid] = (d, now)
        return d

    @classmethod
    def _memory_mind(cls):
        if cls._mind is None:
            from mini_ai.experimental import ExperimentalMind
            cls._mind = ExperimentalMind(school=True, wiki=True, wiki_limit=300000)
            print(f"[memory] экспериментальная модель загружена: {cls._mind.boot}", flush=True)
        return cls._mind


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
