"""cot.py — рассуждение как у GPT с chain-of-thought, но без градиента и без «следующего слова».

Перенос идеи: рассуждение = ТЕКСТ шагов, который система пишет и сама ПЕРЕЧИТЫВАЕТ (CoT, Wei 2022),
несколько цепочек → голосование (self-consistency, Wang 2022), тупик → действие (ReAct: прочитать/спросить).
У нас: рабочая память (scratchpad) = список фактов-шагов. Каждый шаг — применение одного ОПЕРАТОРА вывода
к памяти + scratchpad; результат пишется в scratchpad как новый факт (X, rel, Y) и как строка мысли;
следующий шаг видит его как любой другой факт. Операторы (в архитектуре, не выучены):
  lookup     — факт о сущности из долговременной памяти (KB) → в scratchpad;
  chain      — X rel1 A, A rel2 B ⇒ X (rel1∘rel2) B  (композиция связей: born_in∘city_in = born in country);
  transitive — a R b, b R c ⇒ a R c для R ∈ {bigger, left, north…};
  inherit    — «Все T имеют P» + «x — T» ⇒ x имеет P;
  induce     — у большинства T свойство P, x — T ⇒ x вероятно P;
  negate     — «x не в L», «x в M (M≠L)» ⇒ «x в L?» = нет;
  read       — (действие) нет фактов о сущности → прочитать её статью → снова lookup.
Поиск: луч шириной B по цепочкам; цепочка заканчивается, когда scratchpad содержит факт, отвечающий на
вопрос (тип ответа совпал). Голосование: ответы всех завершённых цепочек взвешиваются суммой уверенностей.
Наружу отдаётся текст мыслей: «Step 1: … Step 2: … So: …» — это и есть «рассуждение вслух».
"""
import re, collections
from mini_ai.memory_v2 import toks, content_words
from mini_ai.think import NAME, norm, TW, PERSON, BAD_NODE, QWORD_TYPE, PLACE, ROLE_TYPE

COMPOSE = {("born_in_country", "country_in"): "born_in_continent", ("company_in_country", "country_in"): "company_in_continent", ("born_in", "city_in"): "born_in_country", ("company_in", "city_in"): "company_in_country", ("capital_of", "country_in"): "capital_in_continent",
           ("city_in", "country_in"): "city_in_continent", ("born_in", "capital_of"): "born_in_country", ("company_in", "capital_of"): "company_in_country"}
REL_PAT = [(r"\bcurrency of\s+$", "currency_of"), (r"\b(?:largest|biggest|most populous) city (?:of|in)\s+$", "largest_city_of"), (r"\bborn in\s+$", "born_in"), (r"\b(?:city|town|village) in\s+$", "city_in"), (r"\bcapital(?: city)?(?: and (?:the )?(?:largest|most populous|biggest) city)? of\s+$", "capital_of"),
           (r"\bcompany in\s+$", "company_in"), (r"\bcountry (?:located )?in(?: the)?\s+$", "country_in"), (r"\bcountry in (?:the )?$", "country_in")]
# «a country in the Vendar continent» / «in Central Asia» / «in Europe»: имя ПОСЛЕ «country in» — континент/регион
CONT_PAT = re.compile(r"\bcountry\b[^,.;]{0,40}?\b(?:in|of) (?:the )?((?:[A-Z][\w\-]*(?:,? and |,? )?){1,4})(?: continent)?\b")
KNOWN_CONT = {"europe", "asia", "africa", "america", "oceania", "antarctica", "australia", "vendar"}
def continent_of(phrase):
    """«Western and Central Europe» → Europe; «Central Asia» → Central Asia; «the European Union» / «the Treaty» → None (не континент)"""
    ws = phrase.replace(",", " ").split(); ws = [w for w in ws if w.lower() != "and"]
    if not ws: return None
    if ws[-1].lower() in KNOWN_CONT:
        if len(ws) >= 2 and ws[-2].lower() in ("north", "south", "central", "eastern", "western", "southeast", "southern", "northern") and ws[-1].lower() in ("america", "asia"): return " ".join(ws[-2:])
        return ws[-1]
    return None
REL_TYPE = {"born_in_country": "country", "company_in_country": "country", "born_in_continent": "continent", "born_in": "city", "city_in": "country", "capital_of": "country", "company_in": "city", "country_in": "continent", "born_in_country": "country", "company_in_country": "country", "capital_in_continent": "continent", "city_in_continent": "continent"}
ANSWER_REL = {"country": {"born_in_country", "company_in_country", "city_in", "capital_of", "largest_city_of"}, "continent": {"country_in", "capital_in_continent", "city_in_continent", "born_in_continent", "company_in_continent"}, "city": {"born_in", "company_in"}}

class Step:
    __slots__ = ("x", "rel", "y", "src", "conf", "text")
    def __init__(self, x, rel, y, src, conf, text): self.x, self.rel, self.y, self.src, self.conf, self.text = x, rel, y, src, conf, text

class CoT:
    def __init__(self, mind): self.mind = mind; self.kb = mind.kb
    # ---------- чтение факта KB → типизированные связи ----------
    def typed(self, i):
        Xw, rel, Yw, X, Y, src = self.kb.facts[i]; out = []
        mc = CONT_PAT.search(Y)
        if mc:
            c = continent_of(mc.group(1))
            if not c and re.search(r"\bcontinent\b", Y[mc.end():mc.end() + 12].lower()): c = mc.group(1).split()[-1]   # «in the Vendar continent»
            if c: out.append((X, "country_in", c, i))
        for m in NAME.finditer(Y):
            n = m.group(0); ctx = Y[:m.start()].lower()
            if norm(n) == norm(X) or norm(n).split()[0] in BAD_NODE: continue
            if mc and norm(n) in norm(mc.group(1)): continue
            for pat, r in REL_PAT:
                if re.search(pat, ctx): out.append((X, r, n, i))
        return out
    def goal(self, q):
        ql = q.lower()
        for pat, ty in QWORD_TYPE:
            if re.search(pat, ql): return ty
        return None
    def anchors(self, q):
        out = []
        for m in NAME.finditer(q):
            n = m.group(0)
            ws = [w for w in n.split() if w.lower() not in ("what", "who", "where", "which", "when", "in", "how", "is", "does", "was", "the")]
            if not ws: continue
            n = " ".join(ws)
            for k in range(len(ws), 0, -1):
                cand = " ".join(ws[:k])
                if norm(cand) in self.kb.by_x: out.append(cand); break
            else: out.append(n)                      # неизвестная сущность — кандидат на чтение
        return list(dict.fromkeys(out))
    # ---------- операторы ----------
    def op_lookup(self, ent, pad):
        new = []
        for i in self.kb.by_x.get(norm(ent), [])[:200]:
            for X, r, n, j in self.typed(i):
                if not any(s.rel == r and norm(s.y) == norm(n) and norm(s.x) == norm(X) for s in pad):
                    new.append(Step(X, r, n, f"memory#{j}", 1.0, f"I know: {X} —{r.replace('_', ' ')}→ {n}."))
        return new
    def op_chain(self, pad):
        new = []
        for a in pad:
            for b in pad:
                if norm(a.y) == norm(b.x) and (a.rel, b.rel) in COMPOSE:
                    r = COMPOSE[(a.rel, b.rel)]
                    if not any(s.rel == r and norm(s.x) == norm(a.x) for s in pad):
                        new.append(Step(a.x, r, b.y, "chain", a.conf * b.conf * 0.95, f"So {a.x} —{r.replace('_', ' ')}→ {b.y} (because {a.x} {a.rel.replace('_', ' ')} {a.y}, and {b.x} {b.rel.replace('_', ' ')} {b.y})."))
        return new
    def op_read(self, ent):
        if not self.mind or any(norm(k) == norm(ent) for k in self.kb.sources): return 0
        try: return self.mind._read_wiki(ent)
        except Exception: return 0
    def _dedup(self, pad):
        seen = {}; out = []
        for s in pad:
            k = (norm(s.x), s.rel, norm(s.y))
            if k not in seen: seen[k] = s; out.append(s)
            elif s.conf > seen[k].conf: seen[k].conf = s.conf
        return out
    # ---------- поиск с лучом + голосование ----------
    def direct(self, q, internet=True):
        """«What is the capital of X?» → факт с типизированной связью capital_of, у которой объект РОВНО X"""
        m = re.match(r"^(?:what|which) (?:is|was) the (capital|currency|largest city)(?: city)? of (.+?)\??$", q.strip(), re.I)
        if not m: return None
        role, X = m.group(1).lower(), m.group(2).strip(); rel = {"capital": "capital_of", "currency": "currency_of", "largest city": "largest_city_of"}[role]
        cands = []
        xw = set(norm(X).split()); role_w = {"capital": "capital", "currency": "currency", "largest city": "largest"}[role]
        idx = set(self.kb.by_word.get(role_w, [])) & set.intersection(*[set(self.kb.by_word.get(w, [])) for w in xw]) if xw else set()
        for i in idx:
            for A, r, n, j in self.typed(i):
                if r == rel and norm(n) == norm(X): cands.append((A, j))
        # обратная форма «Kazakhstan —capital→ Astana» (из «The capital is Astana» в статье о стране)
        for i in self.kb.by_x.get(norm(X), []):
            f = self.kb.facts[i]
            if f[1] == rel.replace("_of", "").replace("largest_city", "largest_city") and f[4][:1].isupper(): cands.append((f[4], i))
        if not cands and internet and self.op_read(X):
            return self.direct(q, internet=False)
        if not cands: return None
        votes = collections.Counter(norm(a) for a, _ in cands); best = votes.most_common(1)[0][0]
        A = next(a for a, _ in cands if norm(a) == best)
        j = next(j for a, j in cands if norm(a) == best); F = self.kb.facts[j]
        return dict(answer=A, confidence=round(min(1.0, 0.7 + 0.1 * votes[best]), 2), type=role,
                    thoughts=[f"I know: {A} —{rel.replace('_', ' ')}→ {X}."], support=[f"{F[3]} —{F[1]}→ {F[4]}"], votes=dict(votes.most_common(3)))

    def reason(self, q, beam=4, max_steps=4, internet=True):
        d = self.direct(q, internet)
        if d: return d
        ty = self.goal(q); anchors = self.anchors(q)
        if not ty or not anchors: return None
        want = ANSWER_REL.get(ty, set())
        def answers(pad):
            out = []
            for s in pad:
                if s.rel in want and norm(s.x) in {norm(a) for a in anchors}:
                    c = s.conf
                    M = getattr(self.mind, "meaning", None)
                    if M is not None and ty in ("continent", "country", "city"):
                        k, sc = M.kind(s.y.split()[-1])
                        if k == ty: c *= 1.2
                        elif k and sc > 0.3 and k in ("organization", "person", "number", "color", "month"): c *= 0.3   # «European Union» — организация, не континент
                    out.append((s.y, c, s))
            return out
        pad = []; thoughts = []
        for a in anchors:
            new = self.op_lookup(a, pad)
            if not new and internet and norm(a) not in {norm(k) for k in self.kb.sources}:
                n = self.op_read(a)
                if n: thoughts.append(f"I had nothing on {a}; read its article ({n} facts)."); new = self.op_lookup(a, pad)
            pad += new
        pad = self._dedup(pad); thoughts += [s.text for s in pad]
        for step in range(max_steps):
            ans = answers(pad)
            if ans: break
            # раскрыть концы известных связей (lookup по хвостам), потом композиция
            tails = {s.y for s in pad if REL_TYPE.get(s.rel) and not any(norm(t.x) == norm(s.y) for t in pad)}
            new = []
            for t in list(tails)[:beam * 2]:
                got = self.op_lookup(t, pad)
                if not got and internet and self.op_read(t): got = self.op_lookup(t, pad); thoughts.append(f"Read about {t}.")
                new += got
            new += self.op_chain(pad + new)
            if not new: break
            before = {(norm(s.x), s.rel, norm(s.y)) for s in pad}; pad = self._dedup(pad + new)
            thoughts += [s.text for s in pad if (norm(s.x), s.rel, norm(s.y)) not in before]
        ans = answers(pad)
        if not ans: return dict(answer=None, thoughts=thoughts, type=ty)
        votes = collections.Counter()
        for y, c, s in ans: votes[norm(y)] += c
        best = votes.most_common(1)[0][0]; surf = next(y for y, _, _ in ans if norm(y) == best)
        chosen = [s for y, _, s in ans if norm(y) == best]
        return dict(answer=surf, confidence=round(min(1.0, votes[best]), 2), type=ty, thoughts=thoughts, support=[s.text for s in chosen],
                    votes={k: round(v, 2) for k, v in votes.most_common(3)})
