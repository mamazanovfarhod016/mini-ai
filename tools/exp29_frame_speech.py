"""Эксперимент 29: ДРУГАЯ СИСТЕМА РЕЧИ — факт + шаблон (без n-грамм, без градиента).
exp26/27 (3-граммы) упёрлись: 3 слова контекста, факты врут, 13/20. Правило Фархода: не подходит —
строим другую. Новая система:
  1. БОГАТАЯ РАМКА: ~20 связок (is located in, is the capital of, was founded in, borders, is known as…)
     — закрытый класс служебных оборотов (как exp18). Предложение → (X, rel, Y) + ШАБЛОН: то же
     предложение, где X→{X}, Y→{Y}. Рамка даёт и факт, и образец, как такой факт говорят.
  2. ПАМЯТЬ ФАКТОВ: bind(X, rel) → Y (список, не забывает), ПАМЯТЬ ШАБЛОНОВ: rel → [шаблоны].
  3. РЕЧЬ: вопрос → (X, rel) → Y из памяти → шаблон того же rel из ДРУГОГО предложения → подстановка.
     «Расскажи про X» → все факты X → несколько предложений.
  4. СМЫСЛ = СВЯЗИ: вектор сущности = Σ bind(rel, Y). Сущности с похожими связями близки — без корпуса.
Метрики: фактов/связок (было 75 с одной «is»); ответы на 30 вопросов — содержит Y (exp27-метрика) и
доля шаблонов из ЧУЖОГО предложения (не копия); соседи по смыслу для 6 сущностей.
"""
import sys, os, json, re, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_v2 import Memory, toks, content_words, C, bind, unit, D, STOP

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
mem = Memory(path=f"{OUT}/wiki_memory"); raw = mem.L.stext
print(f"знаний: {len(raw)} предложений")

# ---- 1. связки: (regex, имя связи, вопрос-шаблон) ----
LINKERS = [
    (r"is located in", "located_in", "where is {X}"), (r"is situated in", "located_in", "where is {X}"),
    (r"is the capital of", "capital_of", "what is {X} the capital of"), (r"is the capital city of", "capital_of", "what is {X} the capital of"),
    (r"was founded in", "founded_in", "when was {X} founded"), (r"was established in", "founded_in", "when was {X} founded"),
    (r"was born in", "born_in", "where was {X} born"), (r"is part of", "part_of", "what is {X} part of"),
    (r"is known as", "known_as", "what is {X} known as"), (r"is also known as", "known_as", "what is {X} known as"),
    (r"is a member of", "member_of", "what is {X} a member of"), (r"borders", "borders", "what does {X} border"),
    (r"is bordered by", "borders", "what does {X} border"), (r"is named after", "named_after", "what is {X} named after"),
    (r"consists of", "consists_of", "what does {X} consist of"), (r"is home to", "home_to", "what is {X} home to"),
    (r"is one of the", "one_of", "what is {X} one of"), (r"became", "became", "what did {X} become"),
    (r"is the largest", "largest", "what is {X} the largest"), (r"is the second[- ]largest", "second_largest", "what is {X} the second largest"),
    (r"produces", "produces", "what does {X} produce"), (r"was built in", "built_in", "when was {X} built"),
    (r"was built by", "built_by", "who built {X}"), (r"was conquered by", "conquered_by", "who conquered {X}"),
    (r"is (?:a|an|the) ", "is", "what is {X}"), (r"was (?:a|an|the) ", "is", "what was {X}"),
]
LINK_RE = [(re.compile(r"^(?P<X>[A-Z][\w'\-]*(?:\s+(?:[A-Z][\w'\-]*|of|the|al|de))*?)\s+(?P<L>" + rx + r")\s*(?P<Y>[^,.;:()]{2,70})"), rel, q)
           for rx, rel, q in LINKERS]

facts, templates = [], collections.defaultdict(list)
for si, s in enumerate(raw):
    for cre, rel, q in LINK_RE:
        m = cre.match(s)
        if not m: continue
        X, Y = m.group("X").strip(), m.group("Y").strip()
        if len(toks(X)) > 4 or not content_words(toks(Y)): continue
        facts.append((X, rel, Y, si))
        templates[rel].append((s[:m.start("X")] + "{X}" + s[m.end("X"):m.start("Y")] + "{Y}" + s[m.end("Y"):], si))
        break
by_rel = collections.Counter(f[1] for f in facts)
print(f"фактов: {len(facts)} (было 75 с одной связкой), связок с фактами: {len(by_rel)}")
print("  " + ", ".join(f"{r}:{n}" for r, n in by_rel.most_common()))

# ---- 2. память фактов (векторная) ----
def ev(words): return unit(sum(C(w) for w in words))
FK = np.stack([unit(bind(ev(toks(X)), rel)) for X, rel, Y, _ in facts])
def recall(X, rel):
    sc = FK @ unit(bind(ev(toks(X)), rel)); i = int(np.argmax(sc))
    return (facts[i], float(sc[i])) if sc[i] > 0.6 else (None, 0.0)

# ---- 3. речь по шаблону ----
QRE = {rel: q for _, rel, q in LINKERS}
def say(X, rel, Y, src_si):
    others = [t for t, si in templates[rel] if si != src_si]
    t = others[0] if others else templates[rel][0][0]
    out = t.replace("{X}", X).replace("{Y}", Y)
    return out, bool(others)

rng = np.random.default_rng(3); idx = rng.permutation(len(facts))[:30]
ok = foreign = 0; print("--- 30 вопросов → ответ по шаблону ---")
for i in idx:
    X, rel, Y, si = facts[i]; q = QRE[rel].format(X=X) + "?"
    f, sc = recall(X, rel)
    if f is None: print(f"  MISS {q}"); continue
    ans, is_foreign = say(f[0], f[1], f[2], f[3]); foreign += is_foreign
    hit = any(w in toks(ans) for w in content_words(toks(Y))); ok += hit
    if i in idx[:12]: print(f"  {q}\n     → {ans[:130]}  {'[чужой шаблон]' if is_foreign else '[свой]'}")
print(f"ИТОГО: содержит факт {ok}/30, шаблон из чужого предложения {foreign}/30")

print("--- «расскажи про X» (несколько фактов → несколько предложений) ---")
for X in ("Tashkent", "Samarkand", "Uzbekistan"):
    fs = [f for f in facts if f[0].lower() == X.lower()][:4]
    print(f"  {X}: " + " ".join(say(*f)[0] for f in fs)[:300])

# ---- 4. смысл = связи: вектор сущности = Σ bind(rel, Y) ----
ent = collections.defaultdict(lambda: np.zeros(D))
for X, rel, Y, _ in facts: ent[X] += bind(rel, ev(content_words(toks(Y))))
names = [n for n in ent if np.linalg.norm(ent[n]) > 0]; EM = np.stack([unit(ent[n]) for n in names])
print(f"--- смысл по связям: {len(names)} сущностей; соседи ---")
for X in ("Tashkent", "Samarkand", "Bukhara", "Uzbekistan", "Islam", "Timur"):
    if X not in ent: continue
    sc = EM @ unit(ent[X]); top = [names[j] for j in np.argsort(-sc)[1:4]]
    print(f"  {X} ~ {top}")
json.dump(dict(facts=len(facts), rels=dict(by_rel), ok=ok, foreign=foreign), open(f"{OUT}/exp29_frame_speech.json", "w"), indent=1)
