"""fewshot.py — рассуждение по показанным примерам (как GPT-3 few-shot), без градиента.

Правило вывода не вшито: пользователь показывает 1–3 примера «вопрос → ответ» на уже известных фактах,
ядро находит в графе связей путь от якоря вопроса к ответу — это схема; схема привязана к форме вопроса
без имён («who is the grand-teacher of _»). Новый вопрос той же формы → тот же путь от нового якоря.
Связи — любые, прочитанные общим читателем из текста (teacher_of, wrote, parent_of…), не список в коде.
Граф строится инкрементально поверх KB (факты (X, rel, Y) + имена внутри Y с ролью «the R of»).
exp60: 3 новых отношения, 1 пример на каждое → 90/90 на новых сущностях; cot.py без примеров — 0/30.
"""
import re, collections
from mini_ai.think import NAME, norm

ROLE_OF = re.compile(r"^(?:the |a |an )?(?P<R>[a-z][\w\- ]{1,30}?) (?:of|from|in) (?P<Y>[A-Z][\w\-]+(?: [A-Z][\w\-]+){0,3})")

class FewShot:
    def __init__(self, kb): self.kb = kb; self.edges = collections.defaultdict(list); self._built = -1; self.rules = {}; self.examples = collections.defaultdict(list)
    def _build(self):
        F = self.kb.facts
        for i in range(self._built + 1, len(F)):
            Xw, rel, Yw, X, Y, src = F[i]
            nx = norm(X).split()
            if not nx or not X[:1].isupper() or nx[0] in ("the", "it", "there", "this", "he", "she", "they"): continue
            m = ROLE_OF.match(Y)
            if rel == "be" and m:                                   # «X is the teacher of Y» → X —teacher→ Y
                r = m.group("R").strip().replace(" ", "_"); y = m.group("Y")
                self._add(X, r, y, i); continue
            if rel == "be":                                          # «X is a writer from Y» (role in ROLE_OF too) / «X is born in Y»
                mb = re.match(r"^(?:born|located|situated) in (?P<Y>[A-Z][\w\-]+(?: [A-Z][\w\-]+){0,2})", Y)
                if mb: self._add(X, "born_in" if Y.startswith("born") else "located_in", mb.group("Y"), i); continue
            for mn in NAME.finditer(Y):                              # «X wrote Y», «X founded Y» — глагол = связь
                n = mn.group(0)
                if norm(n) == norm(X) or norm(n).split()[0] in ("the", "it"): continue
                if rel not in ("be", "have") and Y.strip().startswith(n): self._add(X, rel, n, i); break
        self._built = len(F) - 1
    def _add(self, x, r, y, i):
        self.edges[norm(x)].append((r, norm(y), i)); self.edges[norm(y)].append(("inv:" + r, norm(x), i))
    def names(self, q):
        qs = re.sub(r"^(?:Who|What|Which|Where|In|How)\b\s*", "", q.strip().rstrip("?"))
        out = []
        for n in re.findall(r"\b[A-Z][\w\-]+(?: [A-Z][\w\-]+)*", qs):
            ws = n.split()
            for k in range(len(ws), 0, -1):
                if norm(" ".join(ws[:k])) in self.edges: out.append(" ".join(ws[:k])); break
        return out
    def qkey(self, q):
        qs = q.strip().rstrip("?")
        return re.sub(r"\b[A-Z][\w\-]+(?: [A-Z][\w\-]+)*", "_", re.sub(r"^(?:Who|What|Which|Where|In|How)\b", lambda m: m.group(0).lower(), qs)).lower().strip()
    def paths(self, a, b, max_len=3):
        out = []; stack = [(a, [], {a})]
        while stack:
            node, path, seen = stack.pop()
            if node == b and path: out.append(path); continue
            if len(path) >= max_len: continue
            for rel, nxt, _ in self.edges.get(node, []):
                if nxt not in seen: stack.append((nxt, path + [rel], seen | {nxt}))
        return out
    def show(self, q, answer):
        """пример: вопрос + правильный ответ → схема (пересечение с прежними примерами той же формы)"""
        self._build(); a = self.names(q)
        if not a or norm(answer) not in self.edges: return None
        cands = {tuple(p) for p in self.paths(norm(a[-1]), norm(answer))}
        if not cands: return None
        k = self.qkey(q); self.examples[k].append((q, answer))
        self.rules[k] = (self.rules[k] & cands) if k in self.rules and self.rules[k] & cands else cands
        return sorted(self.rules[k], key=len)
    def answer(self, q):
        self._build(); k = self.qkey(q); a = self.names(q)
        if k not in self.rules or not a: return None
        votes = collections.Counter(); trace = {}
        for schema in sorted(self.rules[k], key=len):
            cur = {norm(a[-1])}; steps = []
            for rel in schema:
                nxt = set()
                for n in cur:
                    for r, y, i in self.edges.get(n, []):
                        if r == rel: nxt.add(y); steps.append(i)
                cur = nxt
                if not cur: break
            for y in cur: votes[y] += 1.0 / len(schema); trace.setdefault(y, steps)
        if not votes: return None
        best = votes.most_common(1)[0][0]; F = self.kb.facts
        surf = next((F[i][3] for i in trace[best] if norm(F[i][3]) == best), None) or next((m.group(0) for i in trace[best] for m in NAME.finditer(F[i][4]) if norm(m.group(0)) == best), best)
        schema = sorted(self.rules[k], key=len)[0]
        return dict(answer=surf, schema="→".join(schema), facts=[f"{F[i][3]} —{F[i][1]}→ {F[i][4][:80]}" for i in dict.fromkeys(trace[best])][:4],
                    example=self.examples[k][0], confidence=round(min(1.0, votes[best]), 2))
