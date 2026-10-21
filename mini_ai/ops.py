"""ops.py — операторы рассуждения над общей памятью фактов (без обучения): сравнение, да/нет, счёт, список, «почему».

Как у GPT-гибридов: ответ = результат ОПЕРАТОРА над извлечёнными фактами, с показом шагов. Операторы:
  compare  «Is X bigger/older/larger than Y?» / «Which is bigger, X or Y?» — величина берётся из фактов
           (population/area/year) оператором extract_quantity; ответ + оба числа; если одного нет — честно так.
  yesno    «Does X border Y?» / «Is X in Y?» — да, если есть факт с X и Y в нужной роли; «нет» только если
           есть исключающий факт (X в Z, Z≠Y для исключающих отношений: capital_of/continent); иначе «не знаю».
  count    «How many countries border X?» — список по оператору list → длина.
  list     «Which countries border X?» — все имена из фактов X с ключевым словом отношения.
  why      «Why is X the capital of Y?» — факт-ответ + поддерживающие факты про X (largest city/population) как причины.
  common   «What do X and Y have in common?» — пересечение типизированных связей/классов.
Величины: число + единица из текста факта («population of more than 39 million» → 39e6, «area of 448,978 square
kilometres» → 448978 km², «founded in 1147»/«in the year 1147» → 1147).
"""
import re, collections
from mini_ai.memory_v2 import toks, content_words
from mini_ai.think import NAME, norm, BAD_NODE

QUANT = {
    "population": (r"population (?:of|estimated at|is) (?:more than |over |around |about |approximately |roughly )?([\d][\d,\.]*)\s*(million|billion|thousand)?", "people"),
    "area": (r"area of (?:about |around |approximately )?([\d][\d,\.]*)\s*(million |thousand )?(?:square kilomet(?:re|er)s?|km2|km²|sq km)", "km²"),
    "year": (r"(?:founded|established|first mentioned|built|created|formed)\b[^.]{0,40}?\b(?:in|around|about)\s+(?:the year )?(\d{3,4})\b(?: ?BC)?", "year"),
}
ADJ = {"bigger": "population", "larger": "area", "more populous": "population", "smaller": "population", "older": "year", "younger": "year", "newer": "year", "higher population": "population"}
REL_WORDS = {"border": ["border", "bordered by", "borders", "neighbo"], "in": ["in", "located in", "part of"], "capital": ["capital"], "speak": ["language", "speak"], "river": ["river", "flows"]}
EXCLUSIVE = {"capital_of", "continent"}

def num(s, mult):
    v = float(s.replace(",", "")); m = (mult or "").strip().lower()
    return v * {"million": 1e6, "billion": 1e9, "thousand": 1e3}.get(m, 1)

class Ops:
    def __init__(self, kb): self.kb = kb
    def facts_of(self, ent):
        return [self.kb.facts[k] for k in self.kb.by_x.get(norm(ent), [])]
    def quantity(self, ent, q):
        pat, unit = QUANT[q]; best = None
        for f in self.facts_of(ent):
            m = re.search(pat, (f[1].replace("_", " ") + " " + f[4]), re.I)
            if m:
                v = num(m.group(1), m.group(2) if m.lastindex and m.lastindex >= 2 else None)
                if q == "year" and f[1] == "be" and "mentioned" not in f[4] and "found" not in f[4] and "built" not in f[4]: continue
                if best is None or (q == "year" and v < best[0]) or (q != "year" and v > best[0]): best = (v, f"{f[3]} —{f[1]}→ {f[4][:90]}")
        return best
    def names(self, ent, rel_words, want_kind=None):
        out = []; M = getattr(self, "meaning", None)
        for f in self.facts_of(ent):
            t = (f[1].replace("_", " ") + " " + f[4])
            if not any(w in t.lower() for w in rel_words): continue
            if rel_words is REL_WORDS["border"] and not re.search(r"\bborder", t.lower()[:60]): continue
            for m in NAME.finditer(f[4]):
                n = m.group(0)
                if norm(n) == norm(ent) or norm(n).split()[0] in BAD_NODE or len(n) < 3: continue
                if M is not None and want_kind:
                    k, sc = M.kind(n.split()[-1])
                    if k and k != want_kind and sc > 0.25 and k in ("month", "person", "number", "color"): continue   # «May», «Turkmen F» — не страны
                if norm(n) not in [norm(x) for x, _ in out]: out.append((n, f"{f[3]} —{f[1]}→ {f[4][:90]}"))
        return out
    def ent_pair(self, q):
        names = [m.group(0) for m in NAME.finditer(q) if norm(m.group(0)).split()[0] not in ("is", "which", "does", "do", "how", "what", "why", "are", "was")]
        return names
    # ---------- операторы ----------
    def compare(self, q):
        ql = q.lower().rstrip("?")
        m = re.match(r"^(?:is|was) (.+?) (bigger|larger|older|younger|newer|smaller|more populous) than (.+)$", ql) or re.match(r"^which is (bigger|larger|older|younger|newer|smaller|more populous),? (.+?) or (.+)$", ql)
        if not m: return None
        if ql.startswith("which"): adj, a, b = m.group(1), m.group(2), m.group(3); mode = "which"
        else: a, adj, b = m.group(1), m.group(2), m.group(3); mode = "yesno"
        a, b = a.strip().title(), b.strip().title(); qn = ADJ[adj]
        va, vb = self.quantity(a, qn), self.quantity(b, qn)
        steps = []
        if not va or not vb:
            miss = [x for x, v in ((a, va), (b, vb)) if not v]
            return dict(answer=None, text=f"I can't compare: I don't have the {qn} of {' and '.join(miss)}.", thoughts=[f"Need {qn} of {a} and {b}."] + [f"I know: {v[1]}" for v in (va, vb) if v], kind="unknown")
        steps += [f"I know: {va[1]}", f"I know: {vb[1]}"]
        fmt = (lambda v: f"{int(v):,}".replace(",", " ")) if qn != "year" else (lambda v: str(int(v)))
        if adj in ("older",): greater = va[0] < vb[0]
        elif adj in ("younger", "newer", "smaller"): greater = va[0] > vb[0] if adj != "smaller" else va[0] < vb[0]
        else: greater = va[0] > vb[0]
        winner = a if greater else b
        if mode == "which": ans = winner; text = f"Let me think. {a}: {fmt(va[0])} ({qn}). {b}: {fmt(vb[0])}. So {winner} is {adj}."
        else: ans = "yes" if greater else "no"; text = f"Let me think. {a}: {fmt(va[0])} ({qn}). {b}: {fmt(vb[0])}. So {'yes' if greater else 'no'}, {a} is {'' if greater else 'not '}{adj} than {b}."
        return dict(answer=ans, text=text, thoughts=steps + [f"So {winner} is {adj}."], kind="answer", confidence=0.9)
    def yesno(self, q):
        ql = q.lower().rstrip("?")
        m = re.match(r"^(?:does|do|did) (.+?) (border|neighbou?r) (.+)$", ql) or re.match(r"^(?:is|are) (.+?) (in|part of|located in) (.+)$", ql)
        if not m: return None
        a, rel, b = m.group(1).strip().title(), m.group(2), m.group(3).strip().title()
        words = REL_WORDS["border"] if rel.startswith(("border", "neighbo")) else REL_WORDS["in"]
        found = [(n, src) for n, src in self.names(a, words) if norm(n) == norm(b) or norm(b) in norm(n)]
        if found:
            verb = "borders" if rel.startswith(("border", "neighbo")) else "is in"
            return dict(answer="yes", text=f"Yes, {a} {verb} {b}.", thoughts=[f"I know: {found[0][1]}", "So yes."], kind="answer", confidence=0.95)
        others = self.names(a, words)
        if rel in ("in", "part of", "located in") and others:
            # исключающее: страна в одном континенте; «Is Uzbekistan in Europe?» при «country in Central Asia» → no
            cont = [n for n, _ in others if norm(n) != norm(b)]
            if cont: return dict(answer="no", text=f"No. {a} is in {cont[0]}, not {b}.", thoughts=[f"I know: {others[0][1]}", f"{b} ≠ {cont[0]} → no."], kind="answer", confidence=0.8)
        if others: return dict(answer="no", text=f"I don't think so. I know {a} {rel}s " + ", ".join(n for n, _ in others[:5]) + f" — {b} is not among them.", thoughts=[f"I know: {s}" for _, s in others[:5]] + [f"{b} not in list → probably no."], kind="answer", confidence=0.6)
        return dict(answer=None, text=f"I don't know whether {a} {rel}s {b} — I have no facts about that.", thoughts=[], kind="unknown")
    def listing(self, q):
        ql = q.lower().rstrip("?")
        m = re.match(r"^(?:which|what) (\w+) (border|neighbou?r)s? (.+)$", ql) or re.match(r"^how many (\w+) (border|neighbou?r)s? (.+)$", ql) or re.match(r"^(?:which|what) (\w+) (?:does|do) (.+?) (border|neighbou?r)$", ql)
        if not m: return None
        if ql.startswith("how many"): cls, ent, count = m.group(1), m.group(3), True
        elif m.lastindex == 3 and m.group(3) in ("border", "neighbor", "neighbour"): cls, ent, count = m.group(1), m.group(2), False
        else: cls, ent, count = m.group(1), m.group(3), False
        ent = ent.strip().title(); items = self.names(ent, REL_WORDS["border"], want_kind="country")
        if not items: return dict(answer=None, text=f"I don't know which {cls} border {ent} yet.", thoughts=[], kind="unknown")
        lst = ", ".join(n for n, _ in items); th = [f"I know: {s}" for _, s in items[:6]]
        if count: return dict(answer=str(len(items)), text=f"Let me count: {lst} — that's {len(items)}.", thoughts=th + [f"Count = {len(items)}."], kind="answer", confidence=0.8)
        return dict(answer=lst, text=f"{ent} borders {lst}.", thoughts=th, kind="answer", confidence=0.85)
    def why(self, q):
        ql = q.lower().rstrip("?")
        m = re.match(r"^why (?:is|was) (.+?) the (capital|largest city) of (.+)$", ql)
        if not m: return None
        a, role, b = m.group(1).strip().title(), m.group(2), m.group(3).strip().title()
        reasons = []
        for f in self.facts_of(a):
            if re.search(r"largest|most populous|population|centre|center|economic|cultural|political|silk road|located", f[4], re.I) and "capital" not in f[4].lower()[:20]:
                reasons.append(f"{f[3]} {('is' if f[1]=='be' else f[1].replace('_',' '))} {f[4][:100].rstrip('.')}")
            if len(reasons) >= 3: break
        if not reasons: return dict(answer=None, text=f"I know {a} is the {role} of {b}, but I don't have facts explaining why.", thoughts=[], kind="unknown")
        return dict(answer=reasons[0], text=f"I can't know the official reason, but here is what supports it: " + "; ".join(reasons) + ".", thoughts=[f"I know: {r}" for r in reasons], kind="answer", confidence=0.5)
    def common(self, q):
        ql = q.lower().rstrip("?")
        m = re.match(r"^what do (.+?) and (.+?) have in common$", ql)
        if not m: return None
        a, b = m.group(1).strip().title(), m.group(2).strip().title()
        def cls(e): return {w for f in self.facts_of(e) if f[1] == "be" for w in content_words(toks(f[4]))[:6]}
        def rel_names(e): return {norm(n) for n, _ in self.names(e, REL_WORDS["border"] + REL_WORDS["in"])}
        both_cls = cls(a) & cls(b) - {"one", "largest", "many", "former", "part", "country", "world"}; both_n = rel_names(a) & rel_names(b)
        parts = []
        if "country" in cls(a) and "country" in cls(b): parts.append("both are countries")
        if both_n: parts.append("both are linked to " + ", ".join(sorted(both_n)[:4]).title())
        if both_cls: parts.append("both are described as " + ", ".join(sorted(both_cls)[:4]))
        if not parts: return dict(answer=None, text=f"I can't find anything {a} and {b} share in my facts.", thoughts=[], kind="unknown")
        return dict(answer="; ".join(parts), text=f"{a} and {b}: " + "; ".join(parts) + ".", thoughts=[f"classes({a}) ∩ classes({b}) = {sorted(both_cls)[:5]}", f"links({a}) ∩ links({b}) = {sorted(both_n)[:5]}"], kind="answer", confidence=0.6)
    def run(self, q):
        for op in (self.compare, self.yesno, self.listing, self.why, self.common):
            r = op(q)
            if r: r["op"] = op.__name__; return r
        return None
