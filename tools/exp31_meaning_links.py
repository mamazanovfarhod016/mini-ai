"""Эксперимент 31: СВЯЗЫВАНИЕ СМЫСЛОВ — смысл = Σ связей, на достаточных данных.
exp30: идея дала мусор, потому что у Tashkent 4 факта, у Samarkand 2. Здесь читаем статьи про 12 сущностей
(города и страны, 2 региона) — по 100–500 предложений каждая → десятки фактов на сущность.
Проверки (написаны ДО запуска):
  A. соседи по смыслу: для каждого города top-3 соседей — сколько из них города (а не «Boxing»)?
     для стран — страны? Успех: ≥ 2/3 правильного типа в среднем.
  B. аналогия Uzbekistan:Tashkent :: Kazakhstan:? → Astana в top-3; Turkey:Ankara :: Uzbekistan:? → Tashkent.
  C. контроль: мешок слов статьи (без связей) — те же проверки. Если связи не лучше мешка — идея не нужна.
Читатель из exp30 (субъект из контекста, связь = глагол) + вектор сущности = Σ bind(rel, IDF-мешок(Y)).
"""
import sys, os, json, re, collections, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_v2 import Memory, toks, content_words, C, bind, unit, D
from tools.exp24_wiki_crawler import fetch_article, clean_extract

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
CITIES = ["Tashkent", "Samarkand", "Bukhara", "Astana", "Almaty", "Istanbul", "Ankara", "Bishkek"]
COUNTRIES = ["Uzbekistan", "Kazakhstan", "Turkey", "Kyrgyzstan"]
CAPITAL = {"Uzbekistan": "Tashkent", "Kazakhstan": "Astana", "Turkey": "Ankara", "Kyrgyzstan": "Bishkek"}

mem = Memory(path=f"{OUT}/exp31_memory")
have = {m.get("src") for m in mem.L.smeta}
for t in CITIES + COUNTRIES:
    if t in have: continue
    try: title, text, _ = fetch_article(t)
    except Exception as e: print(f"  ! {t}: {e}"); time.sleep(10); continue
    n = mem.read_text(clean_extract(text), source=title); mem.save()
    print(f"  прочитано {title}: {n} предл."); time.sleep(1.5)
mem.save()
raw, meta = mem.L.stext, mem.L.smeta
print(f"всего {len(raw)} предложений")

# ---------- читатель (exp30) ----------
VERBS = ("is", "was", "are", "were", "has", "have", "had", "became", "borders", "produces", "consists", "includes",
         "contains", "remains", "lies", "hosts", "hosted", "won", "founded", "established", "built", "serves",
         "operates", "covers", "ranks", "declared", "gained", "joined", "signed", "adopted", "opened", "began",
         "uses", "speaks", "exports", "reached", "comprises", "features", "grew", "stands", "connects", "attracts")
GENERIC = {"it", "the country", "the city", "the republic", "the region", "the state", "the nation", "he", "she",
           "they", "the government", "this", "the capital", "the town", "the province"}
PREAMBLE = re.compile(r"^(?:In|By|During|After|Before|Since|According to|Following|At|On|Under|As of|Although|Despite|Today|Historically|Later|Currently)\b[^,]{0,60},\s*")
def lemma(v): return {"is": "be", "was": "be", "are": "be", "were": "be", "has": "have", "had": "have", "have": "have"}.get(v, re.sub(r"(ed|s)$", "", v))
RX = re.compile(r"^(?P<X>(?:[A-Z][\w'\-]*|the|It|it)(?:\s+(?:[A-Za-z][\w'\-]*)){0,3}?)\s+(?P<V>" + "|".join(VERBS) + r")\b\s*(?P<Y>[^,.;:()]{2,80})")

facts = []; last = None; cur = None
for si, s in enumerate(raw):
    src = meta[si].get("src", "")
    if src != cur: cur, last = src, src
    pre = PREAMBLE.match(s); body = s[pre.end():] if pre else s
    m = RX.match(body)
    if not m: continue
    X, V, Y = m.group("X").strip(), m.group("V"), m.group("Y").strip()
    if X.lower() in GENERIC: X = last
    elif X[0].isupper() and X.lower().split()[0] not in ("the",): last = X
    if X and content_words(toks(Y)): facts.append((X, lemma(V), Y))
per = collections.Counter(f[0] for f in facts)
print(f"фактов {len(facts)}; на сущность: " + ", ".join(f"{t}:{per[t]}" for t in CITIES + COUNTRIES))

# ---------- A/B: смысл = связи ----------
def yvec(Y):
    cw = content_words(toks(Y)) or toks(Y); return unit(sum(mem.sieve.idf(w) * C(w) for w in cw))
ent = collections.defaultdict(lambda: np.zeros(D))
for X, rel, Y in facts: ent[X] += bind(rel, yvec(Y))
# контроль: мешок слов всех предложений статьи (без связей)
bag = collections.defaultdict(lambda: np.zeros(D))
for si, s in enumerate(raw): bag[meta[si]["src"]] += mem.sieve.bag(toks(s))

def evaluate(vecs, label):
    names = [n for n in CITIES + COUNTRIES if n in vecs and np.linalg.norm(vecs[n]) > 0]
    M = np.stack([unit(vecs[n]) for n in names])
    good = tot = 0; print(f"--- {label} ---")
    for n in names:
        sc = M @ unit(vecs[n]); nb = [names[j] for j in np.argsort(-sc)[1:4]]
        kind = CITIES if n in CITIES else COUNTRIES
        g = sum(x in kind for x in nb); good += g; tot += 3
        print(f"  {n:11} ~ {nb}  ({g}/3 того же типа)")
    def analogy(a, b, c):
        v = unit(unit(vecs[b]) - unit(vecs[a]) + unit(vecs[c])); sc = M @ v
        return [names[j] for j in np.argsort(-sc) if names[j] not in (a, b, c)][:3]
    an = {}
    for a, c in (("Uzbekistan", "Kazakhstan"), ("Turkey", "Uzbekistan"), ("Kazakhstan", "Kyrgyzstan"), ("Uzbekistan", "Turkey")):
        if all(k in vecs for k in (a, CAPITAL[a], c)):
            r = analogy(a, CAPITAL[a], c); hit = CAPITAL[c] in r; an[f"{a}:{CAPITAL[a]}::{c}"] = (r, hit)
            print(f"  аналогия {a}:{CAPITAL[a]} :: {c}:? → {r}  {'OK' if hit else 'MISS'} (ждём {CAPITAL[c]})")
    print(f"  A: соседи того же типа {good}/{tot};  B: аналогии {sum(h for _, h in an.values())}/{len(an)}")
    return good / tot, sum(h for _, h in an.values()), len(an)

rA = evaluate(ent, "СМЫСЛ = Σ связей (bind(rel, Y))")
rB = evaluate(bag, "КОНТРОЛЬ: мешок слов статьи")
json.dump(dict(sentences=len(raw), facts=len(facts), per_entity=dict(per), links=rA, bag=rB),
          open(f"{OUT}/exp31_meaning_links.json", "w"), indent=1, ensure_ascii=False)
