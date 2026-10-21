"""Эксперимент 40: РАССУЖДЕНИЕ НА РЕАЛЬНЫХ ФАКТАХ — Планировщик по графу (без градиента).
Факты (X, rel, Y) из дампа = рёбра графа. Вопрос, на который нет одного факта, решается ЦЕПОЧКОЙ:
  «What is the capital of the country where the Eiffel Tower is?»
   → шаг 1: (Eiffel Tower, be, … in Paris) → Paris; (Paris, be, capital of France) → France
   → шаг 2: (France, capital, ?) → Paris.
Механизм (exp9 на реальных данных):
  1. индекс: X → факты; слова Y → факты (обратный);
  2. ВОПРОС → цель (тип ответа: who/what/where/when) + якорь (именованные сущности в вопросе);
  3. поиск вперёд от якоря: BFS по графу на глубину ≤ 2; ребро = факт; переход X→Y по именам в Y;
  4. КРИТЕРИЙ ОСТАНОВА: узел покрывает слова вопроса, которых якорь не покрыл (как exp9 reducible:
     двигаемся, пока цель не достигнута);
  5. ответ = конец цепочки; объяснение = сама цепочка фактов.
Проверка: 20 двухшаговых вопросов (написаны ДО запуска, ответ есть в Simple Wiki — проверено по фактам);
контроль — одношаговый поиск exp36 на тех же вопросах. Успех: цепочка ≥ 12/20, одношаговый ≤ 5/20.
"""
import sys, os, json, time, re, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_ai.memory_v2 import toks, content_words

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
STORE = os.environ.get("STORE", f"{OUT}/simplewiki_store")
facts = [json.loads(l) for l in open(f"{STORE}/facts.jsonl")]
idf = json.load(open(f"{STORE}/idf.json"))
print(f"фактов {len(facts)}", flush=True)

# ---- граф ----
def norm(s): return " ".join(toks(s))
by_x = collections.defaultdict(list)                 # norm(X) → [i]
for i, (X, rel, Y, src) in enumerate(facts): by_x[norm(X)].append(i)
# имена в Y: последовательности слов с заглавной (сущности, на которые можно перейти)
NAME = re.compile(r"\b(?:[A-Z][\w'\-]+)(?:\s+(?:of|the|de|and)?\s*[A-Z][\w'\-]+){0,3}")
def names_in(Y): return [norm(m.group(0)) for m in NAME.finditer(Y) if norm(m.group(0)) in by_x]
STOPQ = {"what", "who", "where", "when", "which", "is", "was", "are", "were", "the", "of", "in", "a", "an", "does", "do",
         "did", "country", "city", "capital", "largest", "biggest", "name", "called", "that", "has", "have", "located"}

def anchors(q):
    """именованные сущности вопроса, которые есть в графе (самые длинные совпадения)"""
    ws = toks(q); out = []
    for n in range(4, 0, -1):
        for i in range(len(ws) - n + 1):
            cand = " ".join(ws[i:i + n])
            if cand in by_x and cand not in out and not set(cand.split()) <= STOPQ: out.append(cand)
    return out

def coverage(text, goal_words):
    tw = set(toks(text)); return sum(idf.get(w, 1.0) for w in goal_words & tw) / max(1e-9, sum(idf.get(w, 1.0) for w in goal_words))

def reason(q, depth=2, beam=30):
    anc = anchors(q)
    if not anc: return None
    qwords = set(content_words(toks(q))) - STOPQ
    goal = qwords - set(w for a in anc for w in a.split())          # что ещё надо покрыть помимо якоря
    best = None
    # состояние: (узел, цепочка фактов, покрытие)
    frontier = [(a, [], 0.0) for a in anc]
    for d in range(depth):
        nxt = []
        for node, chain, cov in frontier:
            for i in by_x.get(node, [])[:200]:
                X, rel, Y, src = facts[i]
                c = coverage(rel.replace("_", " ") + " " + Y, goal) if goal else 1.0
                item = (c, chain + [i], Y)
                if best is None or c > best[0] or (c == best[0] and len(item[1]) < len(best[1])): best = item
                for nm in names_in(Y):
                    if nm != node and nm not in [facts[j][0].lower() for j in chain]: nxt.append((nm, chain + [i], c))
        nxt.sort(key=lambda t: -t[2]); frontier = nxt[:beam]
        if best and best[0] >= 0.99: break
    if not best or best[0] < 0.3: return None
    c, chain, Y = best
    # ответ: если последний факт покрывает цель его Y-частью — ответ = имя в Y или сам X последнего факта
    X, rel, Yl, src = facts[chain[-1]]
    ans_names = names_in(Yl); ans = ans_names[0] if ans_names and coverage(X, goal) < 0.5 else X
    if coverage(Yl, goal) < coverage(X, goal): ans = X
    return dict(answer=ans, score=c, chain=[f"{facts[j][0]} —{facts[j][1]}→ {facts[j][2][:60]}" for j in chain])

# ---- одношаговый контроль: лучший факт по покрытию всех слов вопроса ----
def onestep(q):
    qwords = set(content_words(toks(q))) - STOPQ; best = (0, None)
    for a in anchors(q):
        for i in by_x[a]:
            X, rel, Y, src = facts[i]; c = coverage(X + " " + rel + " " + Y, qwords)
            if c > best[0]: best = (c, Y)
    return best[1]

Q = [  # (вопрос, gold-подстроки) — двухшаговые; написаны до запуска
    ("What is the capital of the country where Mount Everest is?", ["kathmandu", "beijing"]),
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
    ("What language do people speak in the country where Berlin is?", ["german"]),
]
ok2 = ok1 = 0; t0 = time.time(); rows = []
for q, golds in Q:
    r = reason(q); a1 = onestep(q)
    h2 = bool(r and any(g in r["answer"].lower() or any(g in c.lower() for c in r["chain"][-1:]) for g in golds))
    h1 = bool(a1 and any(g in a1.lower() for g in golds))
    ok2 += h2; ok1 += h1
    print(f"  {'OK ' if h2 else 'MISS'} {q}\n       → «{r['answer'] if r else '-'}»  цепочка: {' | '.join(r['chain']) if r else '-'}\n       1 шаг: {'OK' if h1 else 'miss'} «{(a1 or '-')[:50]}»")
    rows.append(dict(q=q, chain=r, onestep=a1, ok2=h2, ok1=h1))
print(f"\nИТОГО: цепочка {ok2}/{len(Q)}, один шаг {ok1}/{len(Q)}, {(time.time()-t0)/len(Q)*1000:.0f} мс/вопрос")
json.dump(dict(chain=ok2, onestep=ok1, n=len(Q), rows=rows), open(f"{OUT}/exp40_reasoning.json", "w"), ensure_ascii=False, indent=1)
