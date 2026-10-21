"""Эксперимент 30: ПОНИМАНИЕ КАК У ЧИТАТЕЛЯ — субъект из контекста + связь из глагола (другая система).
exp29 упёрся: рамка требует «Субъект с заглавной в начале» → 71 факт из 1382 (5%). Википедия пишет
«It is…», «The country has…», «In 1991, the republic…». Другая система:
  1. СУБЪЕКТ ИЗ КОНТЕКСТА (как Coref в bAbI): It/The country/The city/The republic/местоимение → тема
     статьи (заголовок) или последняя названная сущность. Предложение может начинаться с оборота
     («In 1991, …», «According to X, …») — субъект ищем после запятой.
  2. СВЯЗЬ ИЗ ГЛАГОЛА: любой глагол после субъекта (is/has/was/borders/became/produces/…) → rel = лемма
     глагола (грубо: без -s/-ed). Не закрытый список из 25 оборотов, а открытый: связь = глагол.
     Y = всё до первой запятой/точки после глагола (аргумент).
  3. Дополнительно: «In YEAR, X verb Y» → факт (X, verb, Y) + (X, verb@YEAR) — время как связь.
Факт по-прежнему (X, rel, Y) + шаблон. Метрики: покрытие (фактов / предложений), тот же тест речи 30 вопросов,
соседи по смыслу (Σ bind(rel, Y)), аналогия «Bukhara:Uzbekistan = Istanbul:?» если есть данные.
"""
import sys, os, json, re, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_v2 import Memory, toks, content_words, C, bind, unit, D, STOP

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
mem = Memory(path=f"{OUT}/wiki_memory"); raw = mem.L.stext; meta = mem.L.smeta

VERBS = ("is", "was", "are", "were", "has", "have", "had", "became", "borders", "produces", "consists", "includes",
         "contains", "remains", "lies", "hosts", "hosted", "won", "lost", "founded", "established", "built", "serves",
         "operates", "operated", "covers", "covered", "accounts", "ranks", "ranked", "declared", "gained", "joined",
         "signed", "adopted", "introduced", "launched", "opened", "began", "ended", "uses", "used", "speaks", "spoke",
         "produced", "exports", "exported", "imports", "grew", "fell", "rose", "reached", "comprises", "features")
GENERIC = {"it", "the country", "the city", "the republic", "the region", "the state", "the nation", "he", "she",
           "they", "the airline", "the company", "the government", "this", "these", "the capital"}
PREAMBLE = re.compile(r"^(?:In|By|During|After|Before|Since|According to|Following|At|On|Under|As of|Although|Despite|Today|Historically|Later|Currently)\b[^,]{0,60},\s*")
YEAR = re.compile(r"\b(1\d{3}|20\d{2})\b")

def lemma(v):
    return {"is": "be", "was": "be", "are": "be", "were": "be", "has": "have", "had": "have", "have": "have"}.get(v, re.sub(r"(ed|s)$", "", v))

facts, templates = [], collections.defaultdict(list)
last_entity = None; cur_src = None
for si, s in enumerate(raw):
    src = meta[si].get("src", "")
    if src != cur_src: cur_src, last_entity = src, src
    pre = PREAMBLE.match(s); body = s[pre.end():] if pre else s
    year = YEAR.search(pre.group(0)) if pre else None
    m = re.match(r"^(?P<X>(?:[A-Z][\w'\-]*|the|of|and|It|it)(?:\s+(?:[A-Za-z][\w'\-]*)){0,3}?)\s+(?P<V>" + "|".join(VERBS) + r")\b\s*(?P<Y>[^,.;:()]{2,80})", body)
    if not m: continue
    X, V, Y = m.group("X").strip(), m.group("V"), m.group("Y").strip()
    xl = X.lower()
    if xl in GENERIC or xl.startswith(("it ", "the ")) and xl in GENERIC: X = last_entity; how = "ctx"
    elif xl.split()[0] in ("the", "a", "an") and len(xl.split()) <= 3 and xl not in GENERIC: how = "np"
    else: how = "name"
    if not X or not content_words(toks(Y)): continue
    if how == "name" and X[0].isupper(): last_entity = X
    rel = lemma(V)
    facts.append((X, rel, Y, si, how))
    tpl = s.replace(m.group("X"), "{X}", 1) if how != "ctx" else s
    tpl = tpl.replace(Y, "{Y}", 1)
    templates[rel].append((tpl if how != "ctx" else re.sub(r"^(?:[^,]{0,60},\s*)?(?:It|it|The country|The city|The republic|He|She|They)\b", "{X}", tpl), si))
    if year: facts.append((X, rel + "@year", year.group(1), si, how))

by_how = collections.Counter(f[4] for f in facts); by_rel = collections.Counter(f[1] for f in facts if "@" not in f[1])
print(f"знаний: {len(raw)} предл. → фактов {len(facts)} (exp29: 71). Субъект: {dict(by_how)}")
print("  связи: " + ", ".join(f"{r}:{n}" for r, n in by_rel.most_common(15)))

def ev(ws): return unit(sum(C(w) for w in ws))
FK = np.stack([unit(bind(ev(toks(X)), rel)) for X, rel, Y, _, _ in facts])
def recall_all(X, rel, thr=0.6):
    sc = FK @ unit(bind(ev(toks(X)), rel)); return [(facts[i], float(sc[i])) for i in np.argsort(-sc) if sc[i] > thr]

def say(X, rel, Y, src_si):
    """Шаблон из чужого предложения; хвост после {Y} (',… Kyrgyzstan to the northeast') обрезаем —
    он принадлежит чужому факту. Остаётся каркас «{X} глагол {Y}.»"""
    others = [t for t, si in templates[rel] if si != src_si and t.count("{X}") == 1 and t.count("{Y}") == 1]
    if not others: return f"{X} {rel} {Y}.", False
    t = others[0]; t = t[:t.index("{Y}") + 3] + "."
    return t.replace("{X}", X).replace("{Y}", Y), True

rng = np.random.default_rng(3); pool = [f for f in facts if "@" not in f[1]]
idx = rng.permutation(len(pool))[:30]; ok = foreign = 0
print("--- 30 вопросов «X rel ?» → ответ по чужому шаблону ---")
for n, i in enumerate(idx):
    X, rel, Y, si, how = pool[i]
    hits = recall_all(X, rel)
    if not hits: print(f"  MISS {X} {rel}?"); continue
    f = next((h[0] for h in hits if h[0][2] == Y), hits[0][0])      # тот же факт, если память его держит
    ans, fo = say(f[0], f[1], f[2], f[3]); foreign += fo
    hit = any(w in toks(ans) for w in content_words(toks(Y))); ok += hit
    if n < 10: print(f"  {X} {rel}? → {ans[:120]}  {'OK' if hit else 'MISS'}")
print(f"ИТОГО: содержит факт {ok}/30, чужой шаблон {foreign}/30")

print("--- «расскажи про X» ---")
for X in ("Tashkent", "Samarkand", "Uzbekistan", "Timur"):
    fs = [f for f in pool if f[0].lower() == X.lower()][:4]
    print(f"  {X} ({len([f for f in pool if f[0].lower()==X.lower()])} фактов): " + " ".join(say(*f[:4])[0] for f in fs)[:280])

ent = collections.defaultdict(lambda: np.zeros(D))
for X, rel, Y, _, _ in facts:
    cw = content_words(toks(Y)) or toks(Y)
    ent[X] += bind(rel, unit(sum(mem.sieve.idf(w) * C(w) for w in cw)))    # IDF: «country» < «cotton»
names = [n for n in ent if np.linalg.norm(ent[n]) > 0 and len(toks(n)) <= 2]; EM = np.stack([unit(ent[n]) for n in names])
print(f"--- смысл по связям ({len(names)} сущностей ≤2 слов) ---")
for X in ("Tashkent", "Samarkand", "Bukhara", "Uzbekistan", "Islam", "Timur", "Kazakhstan"):
    if X not in ent: continue
    sc = EM @ unit(ent[X]); print(f"  {X} ~ {[names[j] for j in np.argsort(-sc)[1:4]]}")
# аналогия через связи: Uzbekistan − Tashkent + Samarkand ≈ ?  (что для Самарканда то же, что Ташкент для Узбекистана)
def analogy(a, b, c):
    if not all(k in ent for k in (a, b, c)): return None
    v = unit(unit(ent[b]) - unit(ent[a]) + unit(ent[c])); sc = EM @ v
    return [names[j] for j in np.argsort(-sc) if names[j] not in (a, b, c)][:3]
print(f"  аналогия Uzbekistan:Tashkent :: Kazakhstan:? → {analogy('Uzbekistan', 'Tashkent', 'Kazakhstan')}")
json.dump(dict(facts=len(facts), by_how=dict(by_how), ok=ok, foreign=foreign), open(f"{OUT}/exp30_reader_frame.json", "w"), indent=1)
