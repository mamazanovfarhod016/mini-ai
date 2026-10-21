"""deep.py — ГЛУБОКОЕ РАССУЖДЕНИЕ: много шагов по любым фактам, с откатом, без фиксированного списка связей.

cot.py умеет 2–4 шага, но только по географии (born_in∘capital_of…). Здесь — общий поиск в графе фактов:
  узел = сущность (X факта или имя внутри Y), ребро = факт (X —rel→ Y) в обе стороны.
  Старт: сущности вопроса. Цель: узел, который (а) подходит по ТИПУ ответа (what organ / which gas / who),
  (б) связан с остальными словами вопроса. Поиск — луч (beam) глубиной до 7 с откатом: цепочка, которая
  перестала приближаться к словам вопроса, отбрасывается; ветвь без выхода → возврат к предыдущему узлу.
  Оценка узла = (слова вопроса, собранные по пути) × (тип совпал) × (затухание по длине 0.85^шаг).
Мысли пишутся как текст: «Step 1: X is Y. Step 2: Y contains Z. … So: Z.» — это объяснение, не украшение:
каждая строка — реально использованный факт. Без градиента: граф, луч, счётчики.
Самопроверка: ответ не из слов вопроса; по типу — через определения в памяти («Z is an organ») или смысл (meaning.kind).
"""
import re, collections
from mini_ai.memory_v2 import toks, content_words, STOP

NAME = re.compile(r"\b[A-Z][\w\-]+(?: [A-Z][\w\-]+){0,3}\b")
BAD = {"the", "it", "this", "that", "these", "those", "they", "there", "figure", "table", "chapter", "section", "what", "which", "who", "where", "when", "how", "why", "in", "a", "an"}


class Deep:
    def __init__(self, mind):
        self.mind = mind; self.kb = mind.kb; self._built = -1; self.edges = collections.defaultdict(list)   # node → [(rel, other, fact_idx, direction)]

    # ---------- граф строится лениво поверх KB ----------
    def _build(self):
        F = self.kb.facts
        for i in range(self._built + 1, len(F)):
            Xw, rel, Yw, X, Y, src = F[i]
            xn = self._node(X)
            if not xn: continue
            ends = set()
            for m in NAME.finditer(Y):                       # имена внутри Y — узлы
                n = self._node(m.group(0))
                if n and n != xn: ends.add(n)
            # именные группы внутри Y: «a sugar called deoxyribose» → sugar, deoxyribose; «the energy currency of the cell» → energy currency, cell;
            # «made of nucleotides» → nucleotides. Режем по служебным словам и глаголам-связкам, каждая группа (1–3 слова) — узел.
            for grp in re.split(r"\b(?:called|known as|named|of|in|on|at|to|for|from|by|with|into|through|and|or|that|which|is|are|was|were|made|found|used|such as|like)\b|[,;:()]", Y.lower()):
                gw = [w for w in toks(grp) if w not in STOP and w not in BAD]
                if not gw: continue
                node = " ".join(gw[-3:])
                if node != xn and len(node) > 2: ends.add(node)
                if len(gw) > 1 and gw[-1] != xn and len(gw[-1]) > 3: ends.add(gw[-1])     # и голова группы отдельно («energy currency» → «currency»)
            for e in ends:
                self.edges[xn].append((rel, e, i, 1)); self.edges[e].append(("inv:" + rel, xn, i, -1))
            # определения внутри Y: «a sugar called deoxyribose» / «a protein known as hemoglobin» → deoxyribose —be→ sugar
            for md in re.finditer(r"\b(?:a|an|the)?\s*([a-z][\w\-]*(?:\s+[a-z][\w\-]*)?)\s+(?:called|known as|named|termed)\s+([A-Za-z][\w\-]*(?:\s+[A-Za-z][\w\-]*)?)", Y):
                t, n = self._node(md.group(1)), self._node(md.group(2))
                if t and n and t != n: self.edges[n].append(("be", t, i, 1)); self.edges[t].append(("inv:be", n, i, -1))
        self._built = len(F) - 1

    @staticmethod
    def _node(s):
        w = [t for t in toks(s) if t not in BAD]
        return " ".join(w[:3]) if w else None

    # ---------- цель: тип ответа ----------
    def _goal(self, q):
        ql = q.lower().rstrip("?")
        m = re.search(r"\b(?:what|which)\s+(?:type|kind|sort|group|class)\s+of\s+([a-z]+)", ql) or re.search(r"\b(?:what|which)\s+(?:broad\s+|main\s+|large\s+)?([a-z]+)\s+(?!of\b|to\b)([a-z]+)", ql)
        ty = m.group(1) if m else None
        if m and m.lastindex == 2 and m.group(2) in ("of", "to", "that", "which", "from"): ty = None
        if ty in ("is", "are", "do", "does", "did", "the", "a", "an", "type", "kind", "term", "name", "one", "two", "other", "else", "happens", "was", "were", "has", "have", "can", "will", "would", "makes", "made", "causes") or (ty and ty in self.mind.META): ty = None
        if re.match(r"^who\b", ql): ty = "person"
        if re.match(r"^where\b", ql): ty = "place"
        return ty

    def _is_type(self, node, ty):
        """узел подходит под тип? по определениям в памяти «node is a/an … ty» или по смыслу"""
        if not ty: return 0.5
        nw = set(toks(node)); st = lambda w: min(self.mind._forms(w), key=len)
        if st(ty) in {st(w) for w in nw}:
            return 0.0 if len(nw) == 1 else 0.9                     # сам тип («sugar») — не ответ; «simple sugar» — ответ
        # «node is a/an … ty» в графе (ребро be к типу) или в памяти
        for rel, other, fi, d in self.edges.get(node, []):
            if rel == "be" and st(ty) in {st(w) for w in toks(other)}: return 1.0
        for k in self.kb.by_x.get(node, [])[:30]:
            f = self.kb.facts[k]
            if f[1] == "be" and st(ty) in {st(w) for w in f[2]}: return 1.0
        M = getattr(self.mind, "meaning", None)
        if M is not None:
            sims = [M.similar(w, ty) or 0 for w in nw if len(w) > 2]
            if sims and max(sims) > 0.5: return 0.7
            if ty in ("person", "place"):
                k, sc = M.kind(node.split()[-1])
                if k == ty and sc > 0.3: return 0.8
        return 0.15

    # ---------- поиск ----------
    def reason(self, q, beam=6, max_depth=7):
        self._build()
        qall = set(content_words(toks(q))) - self.mind.META
        VERBISH = {"pump", "pumps", "carry", "carries", "take", "takes", "takes", "contain", "contains", "produce", "produces", "use", "uses", "make", "makes",
                   "release", "releases", "absorb", "absorbs", "transport", "transports", "store", "stores", "form", "forms", "cause", "causes", "around", "through", "into"}
        qw = {w for w in qall if not self.mind._soft(w) and w not in VERBISH}
        qverbs = {min(self.mind._forms(w), key=len) for w in qall if w in VERBISH or self.mind._soft(w)}     # carries/take/pumps → рёбра-подсказки
        if not qw: return None
        ty = self._goal(q)
        if ty: qw = {w for w in qw if w != ty and self.mind._stem(w) != self.mind._stem(ty)} or qw   # тип ответа — не слово для поиска
        # старты: сущности вопроса, которые есть в графе (самые длинные совпадения)
        starts = []
        for m in NAME.finditer(q):
            n = self._node(m.group(0))
            if n and n in self.edges: starts.append(n)
        for w in qw:
            if w in self.edges and w not in starts: starts.append(w)
        for a in list(starts):
            ws = a.split()
            for k in range(len(ws), 0, -1):
                c = " ".join(ws[:k])
                if c in self.edges and c not in starts: starts.append(c)
        starts = starts[:4]
        if not starts: return None
        # луч: состояние = (score, node, path=[(fact_idx, text)], covered=set(words))
        beams = [(1.0, s, [], {w for w in qw if w in set(toks(s))}) for s in starts]
        best = None; seen = set(); thoughts = []
        for depth in range(1, max_depth + 1):
            nxt = []
            for score, node, path, cov in beams:
                for rel, other, fi, d in self.edges.get(node, [])[:40]:
                    if (node, other, fi) in seen or other in {p[2] for p in path}: continue
                    seen.add((node, other, fi))
                    f = self.kb.facts[fi]; fw = f[0] | f[2]
                    ncov = cov | (qw & fw) | ({w for w in qw if w in set(toks(other))})
                    gain = len(ncov) - len(cov)
                    rb = rel.replace("inv:", "").replace("_", " ").split()
                    vb = 1.25 if rb and min(self.mind._forms(rb[0]), key=len) in qverbs else 1.0            # связь названа тем же глаголом, что в вопросе
                    s = score * 0.85 * (1.0 + 0.4 * gain) * vb
                    if depth >= 3 and gain == 0 and len(ncov) < len(qw) * 0.5: continue     # откат: ветвь не приближается к вопросу
                    text = f"{f[3]} {('is' if f[1] == 'be' else f[1].replace('_', ' '))} {f[4][:70]}"
                    npath = path + [(fi, text, other)]
                    # кандидат-ответ: узел не из слов вопроса, покрытие вопроса ≥ 60 %, тип подходит
                    ow = set(toks(other)) - STOP
                    if ow and not (ow <= qw) and len(ncov) >= max(1, 0.6 * len(qw)):
                        tsc = self._is_type(other, ty)
                        cand = s * (len(ncov) / len(qw)) * (0.6 + 0.6 * tsc)
                        if (tsc >= 0.5 or (ty is None)) and (not best or cand > best[0]): best = (cand, other, npath, ncov)
                    nxt.append((s, other, npath, ncov))
            if not nxt: break
            nxt.sort(key=lambda b: -b[0]); beams = nxt[:beam]
            if best and best[0] > beams[0][0] * 1.2 and depth >= 2: break          # ответ уверенный — дальше не ищем
        if not best: return None
        cand, ans, path, cov = best
        steps = [f"Step {i+1}: {t}." for i, (fi, t, o) in enumerate(path)]
        return dict(answer=ans, confidence=round(min(1.0, cand), 2), steps=len(path), thoughts=steps + [f"So: {ans}."],
                    facts=[f"{self.kb.facts[fi][3]} —{self.kb.facts[fi][1]}→ {self.kb.facts[fi][4][:80]}" for fi, _, _ in path], covered=sorted(cov), type=ty)
