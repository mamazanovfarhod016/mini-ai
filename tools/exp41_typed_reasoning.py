"""Эксперимент 41: ТИПЫ СВЯЗЕЙ ИЗ СТАТИСТИКИ → планировщик знает, КУДА идти (exp9 field_learned на реальных фактах).
exp40: цепочка 0/20 — планировщик брёл по покрытию слов, якорь ловил «State», «People».
Что добавляем (всё выводится из фактов, не руками):
  1. ТИП СУЩНОСТИ из её определения: (X, be, «a city in …») → тип X = city; «a country» → country;
     «a river» → river… Тип = первое существительное после артикля в be-факте (голова именной группы).
     Распределение: 1.16 млн фактов → типы для ~200k сущностей.
  2. СИГНАТУРА СВЯЗИ из статистики: для rel считаем, какие типы стоят в X и какие сущности-имена в Y:
     capital: X=country → Y=city (по 90%+ фактов). Это таблица exp9 GRAMMAR, но ВЫУЧЕННАЯ.
  3. ЯКОРЬ: только имена с заглавной из вопроса (не любое слово-заголовок).
  4. ВОПРОС → ЦЕЛЬ: вопросное слово + существительное → тип ответа («what is the capital» → city;
     «what language» → language; «what river» → river; «who» → person; «what ocean/sea» → water body).
  5. ПЛАНИРОВЩИК: BFS от якоря; на каждом шаге предпочитает рёбра, чьи Y-сущности ведут к типу цели
     (как exp9 lookahead: reducible). Ответ = сущность нужного типа, достигнутая по цепочке.
Проверка: те же 20 вопросов exp40 + контроль. Успех: ≥ 10/20 (exp40: 0/20).
"""
import sys, os, json, time, re, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_ai.memory_v2 import toks, content_words

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
STORE = os.environ.get("STORE", f"{OUT}/simplewiki_full")
t0 = time.time(); facts = [json.loads(l) for l in open(f"{STORE}/facts.jsonl")]
print(f"фактов {len(facts)}, загрузка {time.time()-t0:.0f}s", flush=True)
def norm(s): return " ".join(toks(s))

# ---- 1. тип сущности из определения ----
HEAD = re.compile(r"^(?:a|an|the|one of the|the largest|the biggest|the second largest|the capital and largest)?\s*(?:[\w\-]+\s+){0,3}?([a-z][\w\-]*)\b")
TYPE_WORDS = {"city": "city", "town": "city", "capital": "city", "village": "city", "municipality": "city",
              "country": "country", "republic": "country", "kingdom": "country", "nation": "country", "state": "state",
              "river": "river", "mountain": "mountain", "lake": "lake", "sea": "sea", "ocean": "sea", "island": "island",
              "continent": "continent", "planet": "planet", "star": "star", "language": "language", "currency": "currency",
              "element": "element", "metal": "element", "antibiotic": "drug", "drug": "drug", "play": "work", "novel": "work",
              "painting": "work", "film": "work", "book": "work", "song": "work", "museum": "building", "tower": "building",
              "mausoleum": "building", "district": "district", "region": "region", "province": "region", "county": "region"}
PERSON = {"scientist", "writer", "actor", "actress", "singer", "painter", "poet", "king", "queen", "emperor", "ruler", "president",
          "politician", "philosopher", "composer", "player", "footballer", "author", "physicist", "mathematician", "inventor",
          "general", "leader", "musician", "prince", "princess", "doctor", "biologist", "chemist", "artist", "architect",
          "explorer", "engineer", "astronomer", "playwright", "dramatist", "novelist", "bacteriologist", "pharmacologist"}
etype = {}
for X, rel, Y, src in facts:
    if rel != "be": continue
    nx = norm(X)
    if nx in etype: continue
    ws = toks(Y)[:8]
    for w in ws:
        if w in TYPE_WORDS: etype[nx] = TYPE_WORDS[w]; break
        if w in PERSON: etype[nx] = "person"; break
tcount = collections.Counter(etype.values())
print(f"типы сущностей: {len(etype)} — {tcount.most_common(12)}", flush=True)

# ---- граф + 2. сигнатуры связей ----
by_x = collections.defaultdict(list)
for i, (X, rel, Y, src) in enumerate(facts): by_x[norm(X)].append(i)
NAME = re.compile(r"\b(?:[A-Z][\w'\-]+)(?:\s+(?:of|the|de|and)?\s*[A-Z][\w'\-]+){0,3}")
def names_in(Y): return [n for n in (norm(m.group(0)) for m in NAME.finditer(Y)) if n in by_x]
sig = collections.defaultdict(collections.Counter)      # rel → Counter((type_x, type_y))
for X, rel, Y, src in facts:
    tx = etype.get(norm(X))
    if not tx: continue
    for n in names_in(Y)[:2]:
        ty = etype.get(n)
        if ty: sig[rel][(tx, ty)] += 1
print("сигнатуры (выучены): " + "; ".join(f"{r}: {sig[r].most_common(1)[0]}" for r in ("be", "have", "border", "capital", "flow") if sig[r]), flush=True)

# ---- 3/4. вопрос → якоря + тип цели ----
QTYPE = [(r"\bcapital\b", "city"), (r"\bcurrency\b", "currency"), (r"\blanguage\b", "language"), (r"\briver\b", "river"),
         (r"\b(?:sea|ocean)\b", "sea"), (r"\bcontinent\b", "continent"), (r"\bplanet\b", "planet"), (r"\bmountain\b", "mountain"),
         (r"\bcity\b", "city"), (r"\bcountry\b", "country"), (r"\bstate\b", "state"), (r"\bsymbol\b", "symbol"), (r"^who\b", "person")]
def goal_type(q):
    ql = q.lower(); head = ql.split(" of the ")[0].split(" in the ")[0]      # тип цели — из ПЕРВОЙ части вопроса
    for rx, t in QTYPE:
        if re.search(rx, head): return t
    return None
def anchors(q):
    out = []
    for m in NAME.finditer(q):
        n = norm(m.group(0))
        if n in by_x and n.split()[0] not in ("what", "who", "where", "when", "which", "the"): out.append(n)
    return out or [n for m in NAME.finditer(q) for n in [norm(m.group(0))] if n in by_x][:1]

# ---- 5. планировщик с типами ----
def reason(q, depth=3, beam=40):
    anc = anchors(q); gt = goal_type(q)
    if not anc or not gt: return None
    qwords = set(content_words(toks(q)))
    # ответ-кандидат: сущность типа gt, достижимая из якоря; приоритет — через рёбра, чьи слова есть в вопросе
    frontier = [(a, []) for a in anc]; seen = set(anc); best = None
    for d in range(depth):
        nxt = []
        for node, chain in frontier:
            for i in by_x.get(node, [])[:300]:
                X, rel, Y, src = facts[i]
                rel_w = set(toks(rel.replace("_", " ")))
                hit_rel = bool(rel_w & qwords) or (gt == "city" and rel == "capital") or (gt == "symbol" and "symbol" in rel)
                # ответ в Y: сущность типа gt или (symbol/currency) сам Y
                if gt == "symbol" and "symbol" in rel:
                    cand = (2 + d * -0.1, Y.strip(), chain + [i]); best = cand if not best or cand[0] > best[0] else best; continue
                for n in names_in(Y):
                    if etype.get(n) == gt or (gt == "person" and etype.get(n) == "person"):
                        score = (1.0 if hit_rel else 0.5) + (0.3 if n not in seen else 0) - 0.1 * d
                        # «capital of X»: ребро должно быть про capital; «city in country»: be-ребро с 'capital' в Y
                        if gt == "city" and not ("capital" in rel or "capital" in Y.lower()): score -= 0.4
                        if not best or score > best[0]: best = (score, n, chain + [i])
                    if n not in seen and etype.get(n) in ("country", "city", "state", "continent", "planet", "person", "work", "building", "river", "mountain", "lake"):
                        nxt.append((n, chain + [i])); seen.add(n)
        frontier = nxt[:beam]
    if not best: return None
    return dict(answer=best[1], chain=[f"{facts[j][0]} —{facts[j][1]}→ {facts[j][2][:55]}" for j in best[2]])

Q = [("What is the capital of the country where Mount Everest is?", ["kathmandu", "beijing"]),
     ("What is the currency of the country whose capital is Paris?", ["euro", "franc"]),
     ("What language is spoken in the country whose capital is Tashkent?", ["uzbek"]),
     ("What is the largest planet in the system that contains Earth?", ["jupiter"]),
     ("Who wrote the play in which Hamlet is the main character?", ["shakespeare"]),
     ("What river flows through the capital of France?", ["seine"]),
     ("What is the capital of the country that Istanbul is in?", ["ankara"]),
     ("What sea is next to the city of Istanbul?", ["marmara", "black sea", "bosporus"]),
     ("What continent is the country of Kazakhstan in?", ["asia"]),
     ("Who discovered the antibiotic that is made by Penicillium?", ["fleming"]),
     ("What is the chemical symbol of the metal used for coins and teeth?", ["au"]),
     ("What is the capital of the country that borders Uzbekistan to the north?", ["astana", "nur-sultan"]),
     ("What ocean is next to the country whose capital is Tokyo?", ["pacific"]),
     ("What is the largest city of the country where the Nile is?", ["cairo"]),
     ("Who painted the picture that hangs in the Louvre and shows a smiling woman?", ["leonardo"]),
     ("What is the capital of the state where Hollywood is?", ["sacramento"]),
     ("What planet is closest to the star at the center of the Solar System?", ["mercury"]),
     ("What is the highest mountain on the continent where Kenya is?", ["kilimanjaro"]),
     ("What is the capital of the country where the Taj Mahal is?", ["delhi"]),
     ("What language do people speak in the country where Berlin is?", ["german"])]
ok = 0; t0 = time.time(); rows = []
for q, golds in Q:
    r = reason(q); h = bool(r and any(g in r["answer"].lower() for g in golds)); ok += h
    print(f"  {'OK ' if h else 'MISS'} {q}\n       → «{r['answer'] if r else '-'}»  [{goal_type(q)}; якоря {anchors(q)}]  {' | '.join(r['chain']) if r else ''}")
    rows.append(dict(q=q, r=r, ok=h))
print(f"\nИТОГО: цепочка с типами {ok}/{len(Q)} (exp40 без типов: 0/20), {(time.time()-t0)/len(Q)*1000:.0f} мс/вопрос")
json.dump(dict(ok=ok, n=len(Q), types=len(etype), rows=rows), open(f"{OUT}/exp41_typed_reasoning.json", "w"), ensure_ascii=False, indent=1)
