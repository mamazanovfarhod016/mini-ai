"""Эксперимент 20: «сама зайдёт в интернет и выучит» — маленькая честная проба.
Берём 2 РЕАЛЬНЫЕ страницы Википедии (не синтетику, не bAbI с закрытым словарём 150 слов) через
web_extract, чистим markdown (ссылки/сноски) до обычной прозы, режем на предложения, читаем их
в Memory ядра (мешок канонических кодов — exp16a/exp17 показали, что это самая надёжная и быстрая
схема поиска предложения: R@1 100% на bAbI, top10 15.9% на синонимах). НИКАКОГО градиента, один проход.
Проверяем: находит ли ядро ПРАВИЛЬНОЕ предложение (R@1) на вопросы, которые Я написал заранее,
читая текст (не подогнаны под то, что ядро умеет) — и содержит ли найденное предложение golden-слово.
Честно: ответ-слово не извлекаем (открытый словарь — известная дыра, round2 §1.3/критик), меряем
только R@1 (нашла нужное предложение) — это и есть «выучила», если понимать обучение как «запомнила
факт и может его найти по вопросу», а не как «генерирует связный текст».
"""
import sys, os, re, json, time, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_core import unit

D = 2048
_C = {}


def C(w):
    if w not in _C: _C[w] = unit(np.random.default_rng(zlib.crc32(w.encode())).standard_normal(D))
    return _C[w]


STOP = set("the a an to of is are was were in on at as by for with from and or but that this which who whom "
           "it its he she they their his her them not no into over under between after before during "
           "has have had be been being will would can could may might also than then there these those "
           "such about one two three first second third most many some all when where what which how".split())


def clean_wiki_markdown(md):
    """Снимает markdown-мусор: [текст](ссылка), сноски [12], картинки, HTML-таблицы — оставляет прозу.
    Дополнительно отрезает ВСЁ после первого служебного раздела (References/See also/External links/
    Bibliography/Notes) — там живут сотни строк библиографии, которые доминируют по совпадению слов
    с заголовком статьи («Istanbul», «Everest») и забивают поиск мусором (0% на первой пробе)."""
    cut = re.search(r"\n#{1,3}\s*(References|See also|External links|Bibliography|Notes|Further reading|Citations)\b",
                     md, re.IGNORECASE)
    if cut: md = md[:cut.start()]
    t = md
    t = re.sub(r"!\[.*?\]\(.*?\)", " ", t)                      # картинки
    # [текст](url "title") -> текст; url может содержать ОДИН уровень вложенных скобок (wiki/X_(Y))
    t = re.sub(r"\[([^\]]*)\]\([^()]*(?:\([^()]*\)[^()]*)*\)", r"\1", t)
    t = re.sub(r"\(https?://\S+?\)", " ", t)                    # остаточные "(https://...)" после первой чистки
    t = re.sub(r"https?://\S+", " ", t)                         # любой оставшийся голый URL
    t = re.sub(r"\[\\?\[?\d+\\?\]?\]", " ", t)                  # [12], [\[12\]]
    t = re.sub(r"\\\[.*?\\\]", " ", t)                          # \[note 1\] и т.п.
    t = re.sub(r"\|.*\|", " ", t)                               # строки таблиц markdown
    t = re.sub(r"[#*_`↑\[\]]", " ", t)
    return t


def is_clean_prose(sentence):
    """Отфильтровывает остаточный мусор (заголовки-ссылки, даты доступа, координаты) по эвристикам
    без ML: доля цифр высокая, нет глагола-связки, начинается с даты/номера — выбрасываем."""
    s = sentence
    digit_ratio = sum(c.isdigit() for c in s) / max(1, len(s))
    if digit_ratio > 0.15: return False
    if re.match(r"^(Retrieved|Archived|ISBN|doi|pp\.|Jump up)", s): return False
    if s.count('"') >= 2 and len(s) < 90: return False       # короткие "Title". — похоже на библиографию
    return True


def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZА-Я])", text)
    return [p.strip() for p in parts if 25 < len(p.strip()) < 400 and is_clean_prose(p.strip())]


def toks(s): return re.findall(r"[a-zA-Z']+", s.lower())
def content_words(ws): return [w for w in ws if w not in STOP and len(w) > 1]


def bag_vec(cwords, idf=None):
    if not cwords: return np.zeros(D)
    if idf is None:
        v = sum(C(w) for w in cwords)
    else:
        v = sum(idf.get(w, 1.0) * C(w) for w in cwords)
    return unit(v)


def build_idf(all_cwords):
    """IDF без градиента: редкое слово (встретилось в малой доле предложений) весит больше —
    классическая мера информативности 1972 (Sparck Jones), посчитана одним проходом по корпусу.
    Лечит провал exp20-v1: частые слова (название статьи «Istanbul», «population») глушили
    редкие информативные слова в простом мешке кодов."""
    N = len(all_cwords)
    df = {}
    for cw in all_cwords:
        for w in set(cw): df[w] = df.get(w, 0) + 1
    return {w: np.log(N / c) for w, c in df.items()}


def read_page(path):
    raw = open(path, encoding="utf-8").read()
    prose = clean_wiki_markdown(raw)
    sents = split_sentences(prose)
    return sents


def build_memory(sents):
    cws = [content_words(toks(s)) for s in sents]
    idf = build_idf(cws)
    vecs = np.stack([bag_vec(cw, idf) for cw in cws])
    return sents, cws, vecs, idf


def ask(question, sents, vecs, idf, topk=3):
    qcw = content_words(toks(question))
    qv = bag_vec(qcw, idf)
    sims = vecs @ qv
    order = np.argsort(-sims)[:topk]
    return [(sents[i], float(sims[i])) for i in order]


# ---------- проба: вопросы написаны ЗАРАНЕЕ, читая текст (не подогнаны под алгоритм) ----------
QUESTIONS = {
    "istanbul": [
        ("What is the population of Istanbul?", ["15 million", "million"]),
        ("Who founded Byzantium?", ["greek", "megarian", "colonists"]),
        ("When did the Ottomans capture Constantinople?", ["1453"]),
        ("Who built Topkapi Palace?", ["mehmed"]),
        ("What two continents does Istanbul span?", ["europe", "asia"]),
        ("What river or waterway does Istanbul straddle?", ["bosphorus"]),
    ],
    "everest": [
        ("How tall is Mount Everest?", ["8,848", "8848", "29,031"]),
        ("What is Mount Everest named after?", ["george everest"]),
        ("What is the Nepali name for Everest?", ["sagarmatha"]),
        ("What is the Tibetan name for Everest?", ["qomolangma", "chomolungma", "mo gla"]),
    ],
}

if __name__ == "__main__":
    OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
    pages = {
        "istanbul": "/opt/hermes-data/cache/web/en.wikipedia.org-941ce62fe1.md",
        "everest": "/opt/hermes-data/cache/web/en.wikipedia.org-435cee9b0b.md",
    }
    results = {}
    for name, path in pages.items():
        t0 = time.time()
        sents = read_page(path)
        s2, cws, vecs, idf = build_memory(sents)
        t_read = time.time() - t0
        print(f"=== {name}: прочитано {len(s2)} предложений за {t_read:.2f}s ===", flush=True)
        qres = []
        hits = 0; hits3 = 0
        for q, gold_substrings in QUESTIONS[name]:
            top = ask(q, s2, vecs, idf, topk=3)
            found_sent, score = top[0]
            ok = any(g.lower() in found_sent.lower() for g in gold_substrings)
            ok3 = any(any(g.lower() in s.lower() for g in gold_substrings) for s, _ in top)
            hits += ok; hits3 += ok3
            print(f"  {'OK ' if ok else ('R3 ' if ok3 else 'MISS')} Q: {q}", flush=True)
            print(f"       -> ({score:.2f}) {found_sent[:160]}", flush=True)
            qres.append(dict(q=q, found=found_sent, score=score, ok=ok, ok_at_3=ok3))
        acc = hits / len(QUESTIONS[name]); acc3 = hits3 / len(QUESTIONS[name])
        print(f"  ИТОГО {name}: R@1 {hits}/{len(QUESTIONS[name])}={acc*100:.0f}%   R@3 {hits3}/{len(QUESTIONS[name])}={acc3*100:.0f}%\n", flush=True)
        results[name] = dict(n_sentences=len(s2), read_sec=round(t_read, 2), questions=qres, acc=acc, acc_at_3=acc3)
    json.dump(results, open(f"{OUT}/exp20_web_read.json", "w"), indent=1, ensure_ascii=False)
