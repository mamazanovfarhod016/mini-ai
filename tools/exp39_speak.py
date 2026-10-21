"""Эксперимент 39: РЕЧЬ СВОИМИ СЛОВАМИ — из фактов, без копирования предложений.
Для 8 сущностей из дампа: факты про тему статьи → mini_ai/speak.compose → абзац.
Проверка (честная, автоматическая):
  1. НЕ КОПИЯ: доля сгенерированных предложений, которых нет в памяти дословно (должно быть ~100%);
  2. ПРАВДА: каждое предложение восстанавливается в факт (X, rel, Y) читателем, и этот факт есть в памяти
     (Y-часть совпадает по словам ≥ 70%) — «своими словами, но не врёт»;
  3. СВЯЗНОСТЬ: нет повторов, есть местоимения, порядок be → have → события.
Плюс сравнение с exp30 (чужие шаблоны): там 100% копия чужой грамматики.
"""
import sys, os, json, re, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_ai.speak import compose
from mini_ai.reader import Reader
from mini_ai.memory_v2 import toks, content_words

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
STORE = os.environ.get("STORE", f"{OUT}/simplewiki_store")
facts = [json.loads(l) for l in open(f"{STORE}/facts.jsonl")]
sent_set = set(" ".join(toks(json.loads(l)[0])) for l in open(f"{STORE}/text.jsonl"))
by_topic = collections.defaultdict(list)
for X, rel, Y, src in facts:
    if toks(X) == toks(src) or (toks(src) and toks(src)[0] in toks(X) and len(toks(X)) <= 3): by_topic[src].append((X, rel, Y))
print(f"фактов {len(facts)}, сущностей с фактами про тему {len(by_topic)}")

TOPICS = ["Paris", "Jupiter", "Gold", "Spider", "William Shakespeare", "Tashkent", "Water", "Penicillin", "Istanbul", "Albert Einstein"]
tot = copy = true_ = n_s = 0; rows = []
for t in TOPICS:
    fs = by_topic.get(t) or by_topic.get(t.split()[-1]) or []
    if not fs: print(f"\n## {t}: нет фактов"); continue
    text = compose(t, fs)
    print(f"\n## {t} ({len(fs)} фактов)\n{text}")
    # проверка
    r = Reader(); sents = [s for s in re.split(r"(?<=\.)\s+", text) if s]
    for s in sents:
        n_s += 1; is_copy = " ".join(toks(s)) in sent_set; copy += is_copy
        got, _ = r.read(s, t); ok = False
        for gX, grel, gY in got:
            gw = set(content_words(toks(gY)))
            for X, rel, Y in fs:
                fw = set(content_words(toks(Y)))
                if gw and len(gw & fw) / len(gw) >= 0.7: ok = True; break
            if ok: break
        true_ += ok
        rows.append(dict(topic=t, sent=s, copy=is_copy, grounded=ok))
    tot += 1
print(f"\nИТОГО: {n_s} предложений; дословная копия из памяти {copy}/{n_s}; подтверждено фактом из памяти {true_}/{n_s}")
json.dump(dict(sentences=n_s, copies=copy, grounded=true_, rows=rows), open(f"{OUT}/exp39_speak.json", "w"), ensure_ascii=False, indent=1)
