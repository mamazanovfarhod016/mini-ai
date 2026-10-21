"""exp53 — аналогия через binding: «Tashkent : Uzbekistan :: ? : Kazakhstan».

Гипотеза: если коды сущностей построены из их фактов (сито: bundle кодов слов), то отношение
«столица→страна» выучивается как ОДИН вектор R = bundle_i( code(city_i) ⊛ code(country_i) ) на train-парах,
и для новой страны ответ = cleanup( R ⊛ code(country) ) среди кодов городов — без хранения явной связи.
ЧЕСТНОСТЬ: из фактов города убираем упоминания его страны, из фактов страны — её столицы (иначе это поиск).
Контроли: случайный (1/N городов), «самый частый», и ПРЯМОЕ сходство code(country)·code(city) без R
(если оно столько же — binding ничего не добавил). Второй набор отношений: река→страна («flows through»).
Успех: top-1 ≥50% на отложенных странах.
"""
import json, os, re, time, collections, numpy as np
rng = np.random.default_rng(0)
SRC = os.path.expanduser("~/Developer/mini-ai-runs/memory-core/simplewiki_full/facts.jsonl")
OUT = os.path.expanduser("~/Developer/mini-ai-runs/learning-core")
STOP = set("a an the of in on at to for from by with and or as is was are were its his her their it this that also".split())
D = 8192
def toks(s): return [w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 1]

def load(limit=1200000):
    facts = collections.defaultdict(list); n = 0
    with open(SRC) as f:
        for line in f:
            X, rel, Y, title = json.loads(line); n += 1
            if n > limit: break
            if X == title: facts[title].append((rel, Y))
    return facts

def pairs_capital(facts):
    out = {}
    for city, fs in facts.items():
        for rel, Y in fs:
            m = re.match(r"^the capital(?: city)?(?: and largest city)? of ([A-Z][\w\-]*(?: [A-Z][\w\-]*)?)\.?$", Y)
            if rel == "be" and m and m.group(1) in facts and city not in out: out[city] = m.group(1)
    return out
def pairs_river(facts):
    out = {}
    for river, fs in facts.items():
        for rel, Y in fs:
            m = re.match(r"^a river (?:in|of|that flows through) ([A-Z][\w\-]*(?: [A-Z][\w\-]*)?)\.?$", Y)
            if rel == "be" and m and m.group(1) in facts and river not in out: out[river] = m.group(1)
    return out

class Sieve:
    def __init__(self): self.W = {}
    def w(self, t):
        if t not in self.W: self.W[t] = rng.choice([-1.0, 1.0], size=D).astype(np.float32)
        return self.W[t]
    def code(self, facts, ent, forbid):
        fb = set(toks(forbid)); v = np.zeros(D, np.float32); n = 0
        for rel, Y in facts[ent][:40]:
            for t in toks(rel + " " + Y):
                if t in fb or t in toks(ent): continue
                v += self.w(t); n += 1
        return np.sign(v + 1e-9) if n else None

def run(name, pairs, facts):
    items = [(a, b) for a, b in pairs.items()]; rng.shuffle(items)
    S = Sieve(); n_te = max(10, len(items) // 4); te, tr = items[:n_te], items[n_te:]
    enc = {}
    for a, b in items:
        ca, cb = S.code(facts, a, b), S.code(facts, b, a)
        if ca is not None and cb is not None: enc[a] = ca; enc[b] = cb
    tr = [(a, b) for a, b in tr if a in enc and b in enc]; te = [(a, b) for a, b in te if a in enc and b in enc]
    A = np.stack([enc[a] for a, _ in items if a in enc]); names = [a for a, _ in items if a in enc]   # все кандидаты-ответы
    R = np.sign(sum(enc[a] * enc[b] for a, b in tr) + 1e-9)                      # вектор отношения
    top1 = top5 = direct = 0; freq = collections.Counter(a for a, _ in tr).most_common(1)[0][0]
    ex = []
    for a, b in te:
        q = R * enc[b]; sc = A @ q; order = np.argsort(-sc)
        top1 += names[order[0]] == a; top5 += a in [names[i] for i in order[:5]]
        direct += names[int(np.argmax(A @ enc[b]))] == a
        if len(ex) < 4: ex.append(f"{b} → {names[order[0]]} (верно: {a})")
    n = len(te); res = dict(pairs=len(items), train=len(tr), test=n, top1=round(top1 / n, 3), top5=round(top5 / n, 3), direct_sim=round(direct / n, 3), random=round(1 / len(names), 4), majority=round(sum(a == freq for a, _ in te) / n, 3))
    print(f"[{name}] пар {len(items)}, train {len(tr)}, test {n}, кандидатов {len(names)}")
    print(f"  binding top-1 {res['top1']:.3f} | top-5 {res['top5']:.3f} || контроль: прямое сходство {res['direct_sim']:.3f} | случайный {res['random']:.4f} | частый {res['majority']:.3f}")
    for e in ex: print("   ", e)
    return res

if __name__ == "__main__":
    t0 = time.time(); facts = load(); print(f"сущностей {len(facts)}, {time.time()-t0:.0f}s")
    res = {"capital": run("столица→страна", pairs_capital(facts), facts), "river": run("река→страна", pairs_river(facts), facts)}
    res["seconds"] = round(time.time() - t0, 1); json.dump(res, open(os.path.join(OUT, "exp53_analogy.json"), "w"), indent=1); print("sec", res["seconds"])
