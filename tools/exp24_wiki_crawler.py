"""Эксперимент 24: АВТОНОМНЫЙ ЧИТАТЕЛЬ ВИКИПЕДИИ — «запустил, и она сама шарится».
Цикл без градиента и без GPU:
  1. очередь тем (seed = несколько тем) → Wikipedia API (plain text, без разметки — exp20 показал, что
     markdown-мусор убивает поиск; API даёт чистый extract)
  2. читаем статью → memory_v2.Memory.read_text (предложения в долговременную память, IDF растёт)
  3. из статьи достаём ССЫЛКИ (API prop=links) → новые темы в очередь (так «шарится по всем частям»)
  4. периодически сохраняем на диск (не забывает между запусками)
  5. замер роста: число предложений, слов в сите, скорость чтения, и ТЕСТ на фиксированных вопросах —
     отвечает ли на то, чего не знала до чтения.
Всё — numpy + urllib. Железо: 2 CPU, без GPU.
"""
import sys, os, json, time, re, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import urllib.request, urllib.parse, urllib.error
import numpy as np
from mini_ai.memory_v2 import Memory, toks, content_words, C, unit

UA = {"User-Agent": "MiniAI-memory-core/0.1 (research project; github NuX-team/mini-ai)"}
API = "https://en.wikipedia.org/w/api.php"


def api(params, retries=4):
    params = dict(params, format="json"); url = API + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=UA)
    for attempt in range(retries):
        try:
            return json.loads(urllib.request.urlopen(req, timeout=20).read())
        except urllib.error.HTTPError as e:
            if e.code == 429: time.sleep(2 * (attempt + 1)); continue   # вежливо ждём, не долбим
            raise
    raise RuntimeError("429 после повторов")


def fetch_article(title):
    d = api(dict(action="query", prop="extracts|links", explaintext=1, titles=title, pllimit="max", plnamespace=0, redirects=1))
    p = list(d["query"]["pages"].values())[0]
    text = p.get("extract", "") or ""
    links = [l["title"] for l in p.get("links", [])]
    return p.get("title", title), text, links


def clean_extract(text):
    # обрезаем служебные разделы в конце статьи
    cut = re.search(r"\n==+\s*(References|See also|External links|Bibliography|Notes|Further reading|Citations)\s*==+", text)
    if cut: text = text[:cut.start()]
    text = re.sub(r"==+[^=]+==+", " ", text)   # заголовки разделов
    return text


def title_vec(title):
    cw = content_words(toks(title))
    return unit(sum(C(w) for w in cw)) if cw else np.zeros(2048)


def crawl(mem, seeds, max_articles=30, save_every=10, log=print, focus=True):
    """focus=True: следующая статья = ссылка, ближайшая по косинусу к «центру интереса» (среднему
    вектору заголовков уже прочитанного). Это чистая геометрия — ни одного правила про темы.
    focus=False: обычная очередь (BFS) — контроль."""
    queue = list(seeds); seen = set(m.get("src") for m in mem.L.smeta)
    center = np.zeros(2048); stats = []; t_start = time.time()
    while queue and len(stats) < max_articles:
        if focus and np.linalg.norm(center) > 0 and len(queue) > 1:
            Q = np.stack([title_vec(t) for t in queue]); i = int(np.argmax(Q @ unit(center)))
        else: i = 0
        title = queue.pop(i)
        if title in seen: continue
        try:
            t0 = time.time(); real_title, text, links = fetch_article(title); t_net = time.time() - t0
        except Exception as e:
            log(f"  ! {title}: {e}"); continue
        if real_title in seen or len(text) < 500: seen.add(real_title); continue
        seen.add(real_title)
        t0 = time.time(); n = mem.read_text(clean_extract(text), source=real_title); t_read = time.time() - t0
        center = center + title_vec(real_title)
        for l in links:
            if l not in seen and l not in queue and ":" not in l and not l.startswith("List of") \
               and not re.match(r"^\d{4}", l) and not l.startswith("."): queue.append(l)
        stats.append(dict(title=real_title, sentences=n, net_s=round(t_net, 2), read_s=round(t_read, 3),
                          total_sentences=len(mem.L.stext), vocab=len(mem.sieve.words), queue=len(queue)))
        log(f"  [{len(stats):3}] {real_title[:40]:40} +{n:4} предл. (сеть {t_net:.1f}s, чтение {t_read:.3f}s) | "
            f"всего {len(mem.L.stext):6} предл., словарь {len(mem.sieve.words):6}, очередь {len(queue)}")
        if len(stats) % save_every == 0: mem.save()
        time.sleep(0.5)                                      # вежливость к API
    mem.save()
    return stats, time.time() - t_start


TEST_QUESTIONS = [   # написаны ДО запуска; темы — те, до которых краулер должен ДОЙТИ по ссылкам, не seed
    ("What is the population of Samarkand?", ["882", "population"]),
    ("Who built the Registan?", ["registan"]),
    ("What river flows through Tashkent?", ["chirchik", "river"]),
    ("When was Bukhara founded?", ["bukhara"]),
    ("What is the capital of Uzbekistan?", ["tashkent"]),
    ("Who was Timur?", ["timur", "tamerlane"]),
    ("What language is spoken in Uzbekistan?", ["uzbek"]),
    ("What is the Silk Road?", ["silk road", "trade"]),
]


def evaluate(mem, log=print):
    ok1 = ok3 = 0
    for q, golds in TEST_QUESTIONS:
        top = mem.ask(q, topk=3)
        if not top: log(f"  MISS {q} (память пуста)"); continue
        hit1 = any(g in top[0][0].lower() for g in golds)
        hit3 = any(any(g in s.lower() for g in golds) for s, _, _ in top)
        ok1 += hit1; ok3 += hit3
        log(f"  {'OK ' if hit1 else ('R3 ' if hit3 else 'MISS')} {q}\n       -> ({top[0][1]:.2f}, {top[0][2].get('src','')}) {top[0][0][:120]}")
    log(f"  ИТОГО: R@1 {ok1}/{len(TEST_QUESTIONS)}  R@3 {ok3}/{len(TEST_QUESTIONS)}")
    return ok1 / len(TEST_QUESTIONS), ok3 / len(TEST_QUESTIONS)


if __name__ == "__main__":
    OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
    STORE = os.path.join(OUT, "wiki_memory")
    mem = Memory(path=STORE)
    print(f"память при старте: {len(mem.L.stext)} предложений, словарь {len(mem.sieve.words)}", flush=True)
    print("--- ДО чтения ---"); r_before = evaluate(mem)
    seeds = sys.argv[1:] or ["Uzbekistan"]
    N = int(os.environ.get("N_ARTICLES", 30))
    stats, dt = crawl(mem, seeds, max_articles=N)
    print(f"--- прочитано {len(stats)} статей за {dt:.0f}s ({sum(s['sentences'] for s in stats)} предложений, "
          f"{sum(s['sentences'] for s in stats)/max(1,dt):.0f} предл/с с учётом сети) ---")
    print("--- ПОСЛЕ чтения ---"); r_after = evaluate(mem)
    json.dump(dict(before=r_before, after=r_after, stats=stats, seconds=dt),
              open(f"{OUT}/exp24_wiki_crawler.json", "w"), indent=1, ensure_ascii=False)
