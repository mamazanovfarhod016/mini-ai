"""think.py — рассуждение цепочкой (chain-of-thought поверх памяти фактов), без градиента.

Что взято у современных сетей: ответ не «ближайший факт», а ЦЕПОЧКА шагов, каждый шаг проверяется,
тупик → откат (поиск в ширину с лучом); след рассуждения отдаётся наружу («подумала: …»).
Что наше: память фактов (ничего не забывает), типы сущностей из be-определений, схемы связей из данных.

Вопрос → цель: (тип ответа, якорь, роль). «In which country was Kelmar born?» → тип=country, якорь=Kelmar.
Поиск: от якоря по рёбрам фактов (X→имена в Y и обратно) до узла нужного типа; глубина ≤3; луч 30.
Тип узла: из be-факта («Zorvania is a country…») или из роли в факте («the capital of Z» → Z: country,
«is a city in Z» → Z: country/region) — роли выучены: ROLE_TYPE.
Оценка пути: произведение «уверенностей» рёбер: прямое ребро с именем = 1.0; через местоимение — 0.5.
"""
import re, collections
from mini_ai.memory_v2 import toks, content_words
from mini_ai.reason import TYPE_WORDS, PERSON, NAME, norm

QWORD_TYPE = [(r"\b(?:in |from )?(?:which|what) country\b", "country"), (r"\bwhich city\b|\bwhat city\b|\bin which city\b", "city"),
              (r"\bwhich continent\b|\bwhat continent\b", "continent"), (r"\bwhich river\b|\bwhat river\b", "river"),
              (r"\bwhich company\b|\bwhat company\b", "company"), (r"\bwhich language\b", "language"), (r"\bwhich currency\b|\bwhat currency\b", "currency"),
              (r"\bwho\b", "person"), (r"\bwhere\b", "place")]
PLACE = {"country", "city", "continent", "region", "state", "island", "sea", "lake", "mountain", "river", "place"}
# роль в Y → тип имени в этой позиции: «the capital of N» → N country; «a city in N» → N country|region
ROLE_TYPE = [(r"\bcapital of\s+$", "country"), (r"\b(?:city|town|village) in\s+$", "country"), (r"\bcurrency of\s+$", "country"),
             (r"\briver in\s+$", "country"), (r"\bcompany in\s+$", "city"), (r"\bborn in\s+$", "city"), (r"\bcontinent\b.*$", "continent")]
BAD_NODE = {"the", "it", "there", "this", "that", "they", "these", "those", "he", "she", "we", "you", "i", "a", "an", "in", "on", "by", "during", "after", "before", "however", "today", "also", "as", "its", "his", "her", "their", "many", "some", "most", "all"}
TW = dict(TYPE_WORDS); TW.update({"company": "company", "corporation": "company", "corp": "company", "firm": "company"})

class Thinker:
    def __init__(self, kb):
        self.kb = kb; self._built = -1; self.etype = {}; self.role_type = {}; self.adj = collections.defaultdict(list)
    def _build(self):
        """инкрементально: типы и граф по фактам, добавленным с прошлого раза"""
        F = self.kb.facts
        for i in range(self._built + 1, len(F)):
            Xw, rel, Yw, X, Y, src = F[i]; nx = norm(X)
            if rel == "be" and nx not in self.etype:
                for w in toks(Y)[:8]:
                    if w in TW: self.etype[nx] = TW[w]; break
                    if w in PERSON: self.etype[nx] = "person"; break
            if nx.split()[0] in BAD_NODE or not X[:1].isupper(): continue
            for m in NAME.finditer(Y):
                n = norm(m.group(0))
                if n == nx or len(n) < 3 or n.split()[0] in BAD_NODE: continue
                ctx = Y[:m.start()].lower()
                for pat, ty in ROLE_TYPE:
                    if re.search(pat, ctx) and n not in self.etype: self.role_type[n] = ty
                self.adj[nx].append((n, i, 1.0)); self.adj[n].append((nx, i, 0.9))
            if nx not in self.etype and X[:1].isupper() and "corp" in nx: self.etype[nx] = "company"
        self._built = len(F) - 1
    def anchors(self, q):
        qs = re.sub(r"^(?:in which|which|what|who|where|when|how)\s+\w*\s*", "", q)
        out = []
        for m in NAME.finditer(q):
            n = norm(m.group(0))
            if n.split()[0] in ("what", "who", "where", "which", "when", "in", "how"): continue
            if n in self.adj: out.append(n)
            else:
                for k in range(len(n.split()), 0, -1):                       # «Kelmar Belan born» → «kelmar belan»
                    cand = " ".join(n.split()[:k])
                    if cand in self.adj: out.append(cand); break
        return list(dict.fromkeys(out))
    def goal(self, q):
        ql = q.lower()
        for pat, ty in QWORD_TYPE:
            if re.search(pat, ql): return ty
        return None
    def fits(self, node, ty):
        t = self.etype.get(node) or self.role_type.get(node)
        if ty == "place": return t in PLACE
        return t == ty
    def think(self, q, depth=3, beam=30):
        self._build(); ty = self.goal(q); anchors = self.anchors(q)
        if not ty or not anchors: return None
        hint = set(content_words(toks(q))) - set(w for a in anchors for w in a.split())
        best = None
        frontier = [(1.0, a, []) for a in anchors]; seen = set(anchors)
        for d in range(depth):
            nxt = []
            for sc, node, path in frontier:
                for n, i, w in self.adj.get(node, []):
                    if n in seen and n not in anchors: continue
                    s = sc * w
                    rel, Y = self.kb.facts[i][1], self.kb.facts[i][4]
                    if hint & (set(toks(rel.replace("_", " "))) | set(toks(Y))): s *= 1.2
                    p = path + [i]
                    if self.fits(n, ty) and n not in anchors:
                        if not best or s > best[0]: best = (s, n, p)
                    else: nxt.append((s, n, p))
            if best and d >= 0 and best[0] >= 0.8: break
            dd = {}
            for s, n, p in nxt:
                if n not in dd or s > dd[n][0]: dd[n] = (s, n, p)
            frontier = sorted(dd.values(), key=lambda t: -t[0])[:beam]; seen |= {n for _, n, _ in frontier}
            if not frontier: break
        if not best: return None
        s, n, p = best; F = self.kb.facts
        steps = [f"{F[i][3]} —{F[i][1]}→ {F[i][4]}" for i in p]
        return dict(answer=F[p[-1]][3] if norm(F[p[-1]][3]) == n else self._surface(n, p), score=round(min(1.0, s), 2), steps=steps, type=ty)
    def _surface(self, n, path):
        for i in path:
            for m in NAME.finditer(self.kb.facts[i][4]):
                if norm(m.group(0)) == n: return m.group(0)
            if norm(self.kb.facts[i][3]) == n: return self.kb.facts[i][3]
        return n
