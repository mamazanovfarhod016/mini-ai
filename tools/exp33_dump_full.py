"""Эксперимент 33: ЛОКАЛЬНЫЙ ДАМП ВИКИПЕДИИ → бинарная память (вместо API с 429).
Поток: bz2 XML → статья → чистка wiki-разметки → предложения → ключ (IDF-мешок, бинарно) + факты (читатель).
Два прохода: (1) IDF по словам (счётчик документов) на первых N статьях; (2) запись. Всё — numpy, bz2, re.
Память: ключи 256 Б/предложение; текст jsonl. Замер: статей/с, предложений, МБ, затем 16 вопросов exp32 +
8 новых (другие темы — проверка, что дамп шире API-краулера).
"""
import sys, os, re, bz2, json, time, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_v2 import toks, content_words, C, unit, D, STOP
from mini_ai.binary_store import BinaryStore, pack
from mini_ai.reader import Reader

DUMP = os.path.expanduser("~/Developer/mini-ai-data/simplewiki.xml.bz2")
OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
STORE = f"{OUT}/simplewiki_full"
N_ART = int(os.environ.get("N_ART", 20000))

# ---------- чистка вики-разметки (без внешних библиотек) ----------
def clean_wikitext(t):
    t = t.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&").replace("&quot;", '"').replace("&nbsp;", " ")
    t = re.sub(r"(?s)<!--.*?-->", " ", t)
    t = re.sub(r"(?s)<ref[^>]*/>", " ", t); t = re.sub(r"(?s)<ref[^>]*>.*?</ref>", " ", t)
    t = re.sub(r"(?s)\{\{[^{}]*\}\}", " ", t); t = re.sub(r"(?s)\{\{[^{}]*\}\}", " ", t)   # шаблоны (2 уровня)
    t = re.sub(r"(?s)\{\|.*?\|\}", " ", t)                                              # таблицы
    t = re.sub(r"\[\[(?:File|Image|Category):(?:[^\[\]]|\[\[[^\]]*\]\])*\]\]", " ", t, flags=re.I)   # с вложенными [[ ]]
    t = re.sub(r"\[\[([^\]|]*)\|([^\]]*)\]\]", r"\2", t); t = re.sub(r"\[\[([^\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\s*([^\]]*)\]", r"\1", t); t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"<[^>]+>", " ", t); t = re.sub(r"'{2,}", "", t)
    t = re.split(r"\n==+\s*(?:References|Related pages|Other websites|Notes|Sources|See also)\s*==+", t)[0]
    t = re.sub(r"==+[^=\n]+==+", " ", t); t = re.sub(r"^\s*[\*#:;].*$", " ", t, flags=re.M)
    t = re.sub(r"&nbsp;|&amp;|&quot;", " ", t)
    t = re.sub(r"\(\s*[;,]?\s*\)", " ", t)                                       # пустые скобки после удаления шаблонов
    return re.sub(r"[ \t]+", " ", t)

def sentences(text):
    for para in text.split("\n"):
        para = para.strip()
        if len(para) < 30: continue
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"])", para):
            s = s.strip()
            if 25 < len(s) < 400 and sum(c.isdigit() for c in s) / len(s) < 0.15 and not s.startswith(("|", "{", "}")): yield s

def articles(path, limit):
    """Потоковый парсер: <page> … <title>, <ns>, <text>. Только ns=0, без redirect."""
    n = 0; title = None; ns = None; buf = None; in_text = False
    with bz2.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if "<title>" in line: title = re.search(r"<title>(.*?)</title>", line).group(1)
            elif "<ns>" in line: ns = re.search(r"<ns>(\d+)</ns>", line).group(1)
            elif "<redirect" in line: title = None
            elif "<text" in line:
                in_text = True; buf = [line.split(">", 1)[1]]
                if "</text>" in line: in_text = False; buf[0] = buf[0].split("</text>")[0]
            elif in_text:
                if "</text>" in line: buf.append(line.split("</text>")[0]); in_text = False
                else: buf.append(line)
            if buf is not None and not in_text:
                if title and ns == "0":
                    yield title, "".join(buf); n += 1
                    if n >= limit: return
                buf = None; title = None

if __name__ == "__main__":
    # ---- проход 1: IDF ----
    t0 = time.time(); df = collections.Counter(); n_docs = 0
    for title, wt in articles(DUMP, N_ART):
        ws = set(content_words(toks(clean_wikitext(wt)))); df.update(ws); n_docs += 1
    idf = {w: float(np.log((1 + n_docs) / (1 + c)) + 1.0) for w, c in df.items()}
    print(f"проход 1 (IDF): {n_docs} статей, словарь {len(df)}, {time.time()-t0:.0f}s", flush=True)

    # ---- проход 2: запись ----
    store = BinaryStore(STORE); reader = Reader(); facts = []; t0 = time.time(); n_s = 0
    for ai, (title, wt) in enumerate(articles(DUMP, N_ART)):
        for s in sentences(clean_wikitext(wt)):
            cw = content_words(toks(s))
            if not cw: continue
            key = sum(idf.get(w, 1.0) * C(w) for w in cw)
            store.add(key, s, {"src": title}); n_s += 1
            fs, _ = reader.read(s, title)
            facts.extend(list(f) + [title] for f in fs if "@" not in f[1])     # + источник (тема статьи)
        if (ai + 1) % 2000 == 0:
            print(f"  {ai+1} статей, {n_s} предл., {len(facts)} фактов, {time.time()-t0:.0f}s, {(ai+1)/(time.time()-t0):.0f} ст/с", flush=True)
    store.save(); json.dump(idf, open(f"{STORE}/idf.json", "w"))
    with open(f"{STORE}/facts.jsonl", "w") as f:
        for x in facts: f.write(json.dumps(x, ensure_ascii=False) + "\n")
    dt = time.time() - t0
    print(f"проход 2: {n_s} предложений, {len(facts)} фактов, ключи {store.B.nbytes/1e6:.0f} МБ, {dt:.0f}s "
          f"({n_s/dt:.0f} предл/с, {N_ART/dt:.0f} ст/с)", flush=True)

    # ---- вопросы (16 из exp32 + 8 новых тем, написаны до запуска) ----
    Q = [("What is the population of Samarkand?", ["population"]), ("What is the capital of Uzbekistan?", ["tashkent"]),
         ("Who was Timur?", ["timur", "tamerlane"]), ("What is the Silk Road?", ["silk road"]),
         ("What strait divides Istanbul?", ["bosphorus", "bosporus"]), ("What is the largest city in Kazakhstan?", ["almaty"]),
         ("What lake is in Kyrgyzstan?", ["issyk"]), ("What sea borders Turkey?", ["black sea", "mediterranean", "aegean"]),
         ("Who wrote Hamlet?", ["shakespeare"]), ("What is the chemical symbol for gold?", ["au"]),
         ("How many legs does a spider have?", ["eight", "8"]), ("What is the boiling point of water?", ["100"]),
         ("Who painted the Mona Lisa?", ["leonardo", "da vinci"]), ("What is the largest planet?", ["jupiter"]),
         ("What is the capital of France?", ["paris"]), ("Who discovered penicillin?", ["fleming"])]
    r1 = r3 = 0; t0 = time.time()
    for q, golds in Q:
        cw = content_words(toks(q)); qv = sum(idf.get(w, 1.0) * C(w) for w in cw)
        top = store.search(qv, 3); hit = [any(g in t.lower() for g in golds) for t, _, _ in top]
        r1 += hit[0] if hit else 0; r3 += any(hit)
        print(f"  {'OK ' if hit and hit[0] else ('R3 ' if any(hit) else 'MISS')} {q} → {top[0][0][:90] if top else ''}")
    print(f"R@1 {r1}/{len(Q)}  R@3 {r3}/{len(Q)}  {(time.time()-t0)/len(Q)*1000:.0f} мс/запрос на {len(store)} предложениях")
    json.dump(dict(articles=N_ART, sentences=n_s, facts=len(facts), mb=store.B.nbytes/1e6, r1=r1, r3=r3, n=len(Q)),
              open(f"{OUT}/exp33_dump.json", "w"), indent=1)
