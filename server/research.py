"""
Поиск ответа в интернете для Mini AI (сервер). Только стандартная библиотека Python.

Источник — Википедия (ru или uz, по языку вопроса), API без ключей.
Ответ считается НАЙДЕННЫМ, только если он прошёл две проверки:
  1. релевантность: ≥ половины смысловых слов вопроса (по корню, первые 5 букв)
     встречаются в заголовке+тексте статьи;
  2. сверка: у статьи есть элемент Wikidata (второй, независимо редактируемый
     реестр фактов) — так мы отсекаем страницы-неоднозначности и мусор.
Если хоть одна проверка не прошла — честно «не нашла».
"""
import json
import re
import urllib.parse
import urllib.request

UA = "MiniAI/0.3 (https://github.com/mamazanovfarhod016/mini-ai; study project)"
STOP = set("""что кто как где когда какой какая какое какие сколько почему зачем это такое такой
ли же бы а и в во на по о об от до из за с со у к ко не ни мне меня тебя ты я он она они мы вы
расскажи скажи объясни подскажи знаешь можешь пожалуйста про есть был была было
nima kim qanday qayer qachon nega necha bu shu u men sen biz siz haqida ayt aytib ber bormi""".split())


def _get(url, timeout=8):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def content_words(q):
    words = re.findall(r"[a-zа-яёʻ'’]+", q.lower())
    return [w for w in words if len(w) >= 3 and w not in STOP]


def relevance(q, text):
    words = content_words(q)
    if not words:
        return 0.0
    t = text.lower().replace("ё", "е")
    hit = sum(1 for w in words if w.replace("ё", "е")[:5] in t)
    return hit / len(words)


def first_sentences(text, max_chars=220):
    text = text.replace("\u0301", "").replace("\u0300", "")   # ударения (Пари́ж) — их нет в словаре модели
    text = re.sub(r"\s*\([^()]*\)", "", text)          # убираем скобки (транскрипции, даты)
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+", text)
    out = ""
    for p in parts:
        if len(out) + len(p) > max_chars and out:
            break
        out = (out + " " + p).strip()
    return out[:max_chars]


def research(question, lang="ru", fetch=_get):
    """→ {found, answer, source, url, relevance, checks}"""
    wiki = "uz" if lang == "uz" else "ru"
    words = content_words(question)
    if not words:
        return {"found": False, "reason": "в вопросе нет слов для поиска"}
    query = " ".join(words)
    try:
        s = fetch(f"https://{wiki}.wikipedia.org/w/api.php?action=query&list=search&format=json&srlimit=3"
                  f"&srsearch={urllib.parse.quote(query)}")
        hits = s.get("query", {}).get("search", [])
        for h in hits:
            title = h["title"]
            summ = fetch(f"https://{wiki}.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(title)}")
            if summ.get("type") == "disambiguation":
                continue
            extract = summ.get("extract") or ""
            rel = relevance(question, title + " " + extract)
            has_wikidata = bool(summ.get("wikibase_item"))
            if rel >= 0.5 and has_wikidata and extract:
                return {
                    "found": True,
                    "answer": first_sentences(extract),
                    "source": f"Википедия ({wiki}) + Wikidata {summ['wikibase_item']}",
                    "title": title,
                    "url": summ.get("content_urls", {}).get("mobile", {}).get("page")
                           or f"https://{wiki}.wikipedia.org/wiki/{urllib.parse.quote(title)}",
                    "relevance": round(rel, 2),
                }
        return {"found": False, "reason": "в Википедии нет подходящей статьи"}
    except Exception as e:  # сеть/таймаут — тоже честно «не нашла»
        return {"found": False, "reason": f"интернет недоступен ({type(e).__name__})"}
