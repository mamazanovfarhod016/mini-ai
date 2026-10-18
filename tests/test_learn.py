"""
Тесты самообучения (server/ + app/www/learn.js).

  1. сервер: журнал вопросов, очередь наставника, ответ наставника → приложению и в очередь обучения;
     защита от мусора (пустой вопрос, не-JSON, выход за папку models/);
  2. поиск в интернете — с подменой сети (без реальных запросов): найдено / нерелевантно / нет Wikidata / сеть упала;
  3. learn.js (через node): какие вопросы считаются «незнакомыми» → идёт в интернет;
     знакомые темы (с опечатками) — нет.
"""
import json
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import app as server_app  # noqa: E402
from server import research, store  # noqa: E402


@pytest.fixture()
def srv(tmp_path):
    server_app.Handler.data = tmp_path
    (tmp_path / "models").mkdir()
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), server_app.Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}", tmp_path
    httpd.shutdown()


def req(url, body=None, raw=None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    r = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=5) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_log_and_mentor_roundtrip(srv):
    base, data = srv
    assert req(base + "/api/log", {"q": "привет", "a": "Привет!", "ok": True})[0] == 200
    assert store.read_jsonl(data / "questions.jsonl")[0]["q"] == "привет"

    code, r = req(base + "/api/mentor", {"q": "как зовут мою кошку", "a": "не знаю", "lang": "ru"})
    assert code == 200 and r["id"] == 1
    assert req(base + "/api/mentor/answers?ids=1")[1] == {}           # пока не ответил

    store.answer_mentor_item(data, 1, "Мурка")
    _, ans = req(base + "/api/mentor/answers?ids=1")
    assert ans["1"]["answer"] == "Мурка"
    assert store.read_json(data / "teach_queue.json", [])[0] == {"q": "как зовут мою кошку", "a": "Мурка", "from": "mentor#1"}


def test_bad_requests_rejected(srv):
    base, _ = srv
    assert req(base + "/api/log", {"q": "  "})[0] == 400
    assert req(base + "/api/log", raw=b"not json")[0] == 400
    assert req(base + "/api/log", raw=b"[1,2]")[0] == 400
    assert req(base + "/models/..%2Fquestions.jsonl")[0] == 404
    assert req(base + "/models/latest.txt")[0] == 404


def test_model_latest_and_download(srv):
    base, data = srv
    assert req(base + "/api/model/latest")[1] == {"version": 0}
    (data / "models/model_v1.json").write_text('{"tokenizer":"bpe"}')
    store.write_json(data / "models/latest.json", {"version": 1, "file": "model_v1.json"})
    assert req(base + "/api/model/latest")[1]["version"] == 1
    assert req(base + "/models/model_v1.json")[1] == {"tokenizer": "bpe"}


def fake_fetch(pages):
    def fetch(url):
        for key, val in pages.items():
            if key in url:
                if isinstance(val, Exception):
                    raise val
                return val
        return {}
    return fetch


SEARCH = {"query": {"search": [{"title": "Париж"}]}}
SUMMARY = {"type": "standard", "extract": "Пари́ж — столица и крупнейший город Франции. Второе предложение.",
           "wikibase_item": "Q90", "content_urls": {"mobile": {"page": "https://ru.m.wikipedia.org/wiki/Париж"}}}


def test_research_found():
    r = research.research("столица франции", "ru", fetch=fake_fetch({"list=search": SEARCH, "summary": SUMMARY}))
    assert r["found"] and "столица" in r["answer"] and "Q90" in r["source"]


def test_research_rejects_irrelevant_and_unverified():
    off = dict(SUMMARY, extract="Совсем другая статья про кино.")
    assert not research.research("столица франции", "ru", fetch=fake_fetch({"list=search": SEARCH, "summary": off}))["found"]
    nowd = {k: v for k, v in SUMMARY.items() if k != "wikibase_item"}
    assert not research.research("столица франции", "ru", fetch=fake_fetch({"list=search": SEARCH, "summary": nowd}))["found"]


def test_research_network_error_is_honest():
    r = research.research("столица франции", "ru", fetch=fake_fetch({"list=search": OSError("down")}))
    assert not r["found"] and "интернет" in r["reason"]


@pytest.mark.skipif(shutil.which("node") is None, reason="нет node")
def test_learn_js_needs_web():
    js = r"""
    global.window = global; global.localStorage = { _s: {}, getItem(k){return this._s[k]??null}, setItem(k,v){this._s[k]=String(v)} };
    global.fetch = () => Promise.reject(new Error("offline"));
    require(process.argv[1] + '/app/www/bpe.js'); window.MiniAIBPE = module.exports;
    eval(require('fs').readFileSync(process.argv[1] + '/app/www/transformer.js','utf8'));
    eval(require('fs').readFileSync(process.argv[1] + '/app/www/known_words.js','utf8'));
    eval(require('fs').readFileSync(process.argv[1] + '/app/www/learn.js','utf8'));
    const qs = JSON.parse(process.argv[2]);
    console.log(JSON.stringify(qs.map(q => window.MiniLearn.needsWeb(q))));
    """
    web = ["столица франции", "кто такой эйнштейн", "что такое фотосинтез", "o'zbekiston poytaxti",
           "сколько планет в солнечной системе", "где находится эйфелева башня"]
    chat = ["привет", "превет как дила", "кто тебя создал", "ну как ты там", "хочу анекдот", "salom qalaysan",
            "что ты умеешь", "который час"]
    out = subprocess.run(["node", "-e", js, str(ROOT), json.dumps(web + chat, ensure_ascii=False)],
                         capture_output=True, text=True, check=True).stdout
    res = json.loads(out)
    assert res[:len(web)] == [True] * len(web), list(zip(web, res))
    assert res[len(web):] == [False] * len(chat), list(zip(chat, res[len(web):]))
