"""exp54 — структурные коды сущностей: Σ (роль ⊛ значение) вместо мешка слов.

Гипотеза: если код сущности = bundle_i( code(rel_i) ⊛ code(Y_i) ), где code(Y) — мешок слов Y, то
(а) по коду БЕЗ факта «X is a <тип>» можно угадать тип (city/country/river/actor/chemical…) ближайшим
прототипом типа — обобщение «что это за вещь» на новых сущностях: top-1 ≥0.6 на ≥8 типах
(контроль: мешок слов всех фактов, случайный, частый тип);
(б) аналогия столица→страна (exp53: 11%) с такими кодами ≥30% top-1.
Все коды биполярные ±1, D=8192, binding — поэлементное умножение; никакого градиента.
"""
import json, os, re, time, collections, numpy as np
rng = np.random.default_rng(0)
SRC = os.path.expanduser("~/Developer/mini-ai-runs/memory-core/simplewiki_full/facts.jsonl")
OUT = os.path.expanduser("~/Developer/mini-ai-runs/learning-core")
STOP = set("a an the of in on at to for from by with and or as is was are were its his her their it this that also very most".split())
D = 8192
TYPES = ["city", "town", "village", "river", "country", "actor", "singer", "politician", "writer", "footballer", "band", "film", "movie", "album", "song", "chemical", "element", "lake", "island", "mountain", "company", "language", "species", "plant", "bird", "fish", "game", "book", "disease", "planet"]
def toks(s): return [w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 1]

class Sieve:
    def __init__(self): self.W = {}
    def w(self, t):
        if t not in self.W: self.W[t] = rng.choice([-1.0, 1.0], size=D).astype(np.float32)
        return self.W[t]
    def bag(self, words):
        v = np.zeros(D, np.float32)
        for t in words: v += self.w(t)
        return v
    def structured(self, facts, forbid_words=()):
        """Σ rel ⊛ sign(bag(Y)) — роль связывается со значением"""
        v = np.zeros(D, np.float32)
        for rel, Y in facts:
            ws = [t for t in toks(Y) if t not in forbid_words]
            if not ws: continue
            v += self.w("rel:" + rel) * np.sign(self.bag(ws) + 1e-9)
        return np.sign(v + 1e-9) if np.any(v) else None
    def flat(self, facts, forbid_words=()):
        ws = [t for rel, Y in facts for t in toks(rel + " " + Y) if t not in forbid_words]
        return np.sign(self.bag(ws) + 1e-9) if ws else None

def load(limit=1200000):
    facts = collections.defaultdict(list); n = 0
    with open(SRC) as f:
        for line in f:
            X, rel, Y, title = json.loads(line); n += 1
            if n > limit: break
            if X == title and "@" not in rel: facts[title].append((rel, Y))
    return facts

def type_of(facts):
    """тип = голова первого факта «be a/an/the <тип>»"""
    for rel, Y in facts:
        m = re.match(r"^(?:an?|the)\s+(?:[\w\-]+\s+){0,2}?([a-z]+)\b", Y.lower())
        if rel == "be" and m and m.group(1) in TYPES: return m.group(1), (rel, Y)
    return None, None

def task_type(facts, S):
    ents = []
    for e, fs in facts.items():
        t, f_be = type_of(fs)
        if t and len(fs) >= 4: ents.append((e, t, [f for f in fs if f != f_be]))
    cnt = collections.Counter(t for _, t, _ in ents); keep = {t for t, c in cnt.items() if c >= 40}
    ents = [x for x in ents if x[1] in keep]; rng.shuffle(ents)
    per = collections.defaultdict(list)
    for x in ents: per[x[1]].append(x)
    ents = [x for t in per for x in per[t][:400]]; rng.shuffle(ents)           # баланс: ≤400 на тип
    n_te = len(ents) // 5; te, tr = ents[:n_te], ents[n_te:]
    print(f"[тип по остальным фактам] сущностей {len(ents)}, типов {len(keep)}: {dict(collections.Counter(t for _, t, _ in ents))}")
    res = {}
    for name, enc in (("структурный Σ rel⊛Y", S.structured), ("мешок слов", S.flat)):
        P = {t: np.zeros(D, np.float32) for t in keep}
        for e, t, fs in tr:
            c = enc(fs)
            if c is not None: P[t] += c
        T = list(keep); M = np.stack([np.sign(P[t] + 1e-9) for t in T])
        ok = n = 0; conf = collections.Counter()
        for e, t, fs in te:
            c = enc(fs)
            if c is None: continue
            p = T[int(np.argmax(M @ c))]; ok += p == t; n += 1; conf[(t, p)] += 1
        res[name] = round(ok / n, 3); print(f"  {name}: top-1 {ok/n:.3f}  (n={n})")
        if name.startswith("структ"):
            wrong = [(k, v) for k, v in conf.most_common() if k[0] != k[1]][:5]
            print("   частые путаницы:", ", ".join(f"{a}→{b} {v}" for (a, b), v in wrong))
    maj = collections.Counter(t for _, t, _ in tr).most_common(1)[0][0]
    res["majority"] = round(sum(t == maj for _, t, _ in te) / len(te), 3); res["random"] = round(1 / len(keep), 3)
    print(f"  контроль: частый тип {res['majority']:.3f}, случайный {res['random']:.3f}")
    return res

def task_analogy(facts, S):
    pairs = {}
    for city, fs in facts.items():
        for rel, Y in fs:
            m = re.match(r"^the capital(?: city)?(?: and largest city)? of ([A-Z][\w\-]*(?: [A-Z][\w\-]*)?)\.?$", Y)
            if rel == "be" and m and m.group(1) in facts and city not in pairs: pairs[city] = m.group(1)
    items = list(pairs.items()); rng.shuffle(items); n_te = len(items) // 4; te, tr = items[:n_te], items[n_te:]
    enc = {}
    for a, b in items:
        ca = S.structured(facts[a], set(toks(b))); cb = S.structured(facts[b], set(toks(a)))
        if ca is not None and cb is not None: enc[a] = ca; enc[b] = cb
    tr = [(a, b) for a, b in tr if a in enc and b in enc]; te = [(a, b) for a, b in te if a in enc and b in enc]
    names = [a for a, _ in items if a in enc]; A = np.stack([enc[a] for a in names])
    R = np.sign(sum(enc[a] * enc[b] for a, b in tr) + 1e-9)
    top1 = sum(names[int(np.argmax(A @ (R * enc[b])))] == a for a, b in te); direct = sum(names[int(np.argmax(A @ enc[b]))] == a for a, b in te)
    r = dict(test=len(te), top1=round(top1 / len(te), 3), direct=round(direct / len(te), 3))
    print(f"[аналогия столица→страна, структурные коды] test {len(te)}: binding top-1 {r['top1']:.3f} | прямое сходство {r['direct']:.3f} | exp53 (мешок) 0.107")
    return r

if __name__ == "__main__":
    t0 = time.time(); facts = load(); S = Sieve()
    res = {"type": task_type(facts, S), "analogy": task_analogy(facts, S), "seconds": round(time.time() - t0, 1)}
    json.dump(res, open(os.path.join(OUT, "exp54_structured_codes.json"), "w"), indent=1); print("sec", res["seconds"])
