"""Рассуждение (exp42): планировщик с РАЗБОРОМ ВОПРОСА и ТИПАМИ СВЯЗЕЙ. Без градиента.
Три части, все из данных:
  1. Типы сущностей (из be-определений) — exp41.
  2. СХЕМЫ СВЯЗЕЙ — выучены из фактов: для пары типов (A → B) какие слова в rel/Y ведут из A в B.
     Например city→country: «capital of», «city in», «located in»; country→sea: «border», «coast», «next to».
     Считается как Counter по всем фактам, где X типа A и имя типа B в Y. Это таблица exp9 GRAMMAR, но
     выученная на 1.16 млн фактов.
  3. РАЗБОР ВОПРОСА на цепочку типов: «What is the capital of the country where Mount Everest is?» →
     [Everest] → (?country) → (?city, роль capital). Правила разбора — по служебным словам (where/whose/that/
     of the), как exp18: закрытый класс, ~10 шаблонов.
  4. ПОИСК: на каждом шаге из узла типа A в тип B — только по рёбрам, чья «сила» (A→B, слово) ≥ порога;
     beam по сумме сил. Ответ = узел последнего шага; объяснение — цепочка.
Отдельно: прямые 1-шаговые вопросы (без вложенности) → тот же механизм с 1 шагом.
"""
import re, json, collections
from mini_ai.memory_v2 import toks, content_words

TYPE_WORDS = {"city": "city", "town": "city", "capital": "city", "village": "city", "municipality": "city",
              "country": "country", "republic": "country", "kingdom": "country", "nation": "country", "state": "state",
              "river": "river", "mountain": "mountain", "lake": "lake", "sea": "sea", "ocean": "sea", "gulf": "sea", "island": "island",
              "continent": "continent", "planet": "planet", "star": "star", "language": "language", "currency": "currency",
              "element": "element", "metal": "element", "antibiotic": "drug", "drug": "drug", "play": "work", "novel": "work",
              "painting": "work", "film": "work", "book": "work", "song": "work", "museum": "building", "tower": "building",
              "mausoleum": "building", "palace": "building", "district": "region", "region": "region", "province": "region",
              "county": "region", "desert": "region", "volcano": "mountain", "peak": "mountain"}
PERSON = {"scientist", "writer", "actor", "actress", "singer", "painter", "poet", "king", "queen", "emperor", "ruler", "president",
          "politician", "philosopher", "composer", "player", "footballer", "author", "physicist", "mathematician", "inventor",
          "general", "leader", "musician", "prince", "princess", "doctor", "biologist", "chemist", "artist", "architect",
          "explorer", "engineer", "astronomer", "playwright", "dramatist", "novelist", "bacteriologist", "pharmacologist",
          "conqueror", "sultan", "khan", "pharaoh", "statesman", "economist", "pianist", "director", "producer"}
NAME = re.compile(r"\b(?:[A-Z][\w'\-]+)(?:\s+(?:of|the|de|and)?\s*[A-Z][\w'\-]+){0,3}")
QTYPE = [("capital", "city"), ("currency", "currency"), ("language", "language"), ("river", "river"), ("sea", "sea"), ("ocean", "sea"),
         ("continent", "continent"), ("planet", "planet"), ("mountain", "mountain"), ("city", "city"), ("country", "country"),
         ("state", "state"), ("lake", "lake"), ("island", "island"), ("element", "element"), ("metal", "element")]
NEXT = {"country", "city", "state", "continent", "planet", "person", "work", "building", "river", "mountain", "lake", "region", "island", "sea", "star"}


def norm(s): return " ".join(toks(s))


class Reasoner:
    def __init__(self, facts):
        self.facts = list(facts); self.by_x = collections.defaultdict(list); self.etype = {}
        for X, rel, Y, src in facts:                                    # 1. типы сущностей
            if rel != "be": continue
            nx = norm(X)
            if nx in self.etype: continue
            for w in toks(Y)[:8]:
                if w in TYPE_WORDS: self.etype[nx] = TYPE_WORDS[w]; break
                if w in PERSON: self.etype[nx] = "person"; break
        # 1b. ОБРАТНЫЕ рёбра (exp9: «чей друг Y» = чтение по значению). «Tokyo be the capital of Japan» →
        #     (Japan, inv:capital_of, Tokyo). Для каждого имени N в Y добавляем факт (N, inv:<rel>, X + контекст).
        #     Контекст = слова Y до имени (например «the capital of») — чтобы схема видела «capital».
        n0 = len(self.facts)
        for i in range(n0):
            X, rel, Y, src = self.facts[i]
            for m in NAME.finditer(Y):
                n = norm(m.group(0))
                if n == norm(X) or len(n) < 3: continue
                ctx = Y[:m.start()].strip()
                self.facts.append((m.group(0), "inv:" + rel, (ctx + " " + X).strip(), src))
        for i, (X, rel, Y, src) in enumerate(self.facts): self.by_x[norm(X)].append(i)
        self.n_inverse = len(self.facts) - n0
        # 2. схемы связей: (type_x, type_y) → Counter(слово из rel+Y) — по ВСЕМ рёбрам, включая обратные
        self.scheme = collections.defaultdict(collections.Counter); self.scheme_n = collections.Counter()
        for X, rel, Y, src in self.facts:
            tx = self.etype.get(norm(X))
            if not tx: continue
            for n in self.names_in(Y)[:2]:
                ty = self.etype.get(n)
                if not ty or ty == tx and tx == "city": continue
                self.scheme_n[(tx, ty)] += 1
                for w in set(content_words(toks(rel.replace("_", " ") + " " + Y))) - set(n.split()):
                    self.scheme[(tx, ty)][w] += 1

    def names_in(self, Y):
        out = []
        for m in NAME.finditer(Y):
            n = norm(m.group(0))
            if n in self.by_x: out.append(n)
            else:                                                       # «East Africa» → «africa», «Western Pacific» → «pacific»
                tail = n.split()[-1]
                if len(n.split()) > 1 and tail in self.by_x and self.etype.get(tail) in ("continent", "sea", "country"): out.append(tail)
        return out

    def edge_strength(self, tx, ty, rel, Y, n):
        """насколько ребро (слова rel+Y) типично для перехода tx→ty — из статистики, [0..1]"""
        c = self.scheme.get((tx, ty)); N = self.scheme_n.get((tx, ty), 0)
        if not c or N < 5: return 0.0
        ws = set(content_words(toks(rel.replace("_", " ") + " " + Y))) - set(n.split())
        return max((c[w] / N for w in ws), default=0.0)

    # 3. разбор вопроса → [якорь, (тип1, слова-подсказки), (тип2, слова-подсказки)...]
    def parse(self, q):
        ql = q.lower().rstrip("?")
        anchors = [n for n in (norm(m.group(0)) for m in NAME.finditer(q)) if n in self.by_x and n.split()[0] not in ("what", "who", "where", "when", "which")]
        if not anchors: return None
        # вложенные обороты: «... of the TYPE (where|whose capital is|that|in which) ...» → внутренний шаг
        steps = []
        m = re.search(r"\b(?:the|a)\s+(\w+)\s+(?:where|that|which|whose|in which|on which)\b(.*)$", ql)
        inner_type = TYPE_WORDS.get(m.group(1)) if m else None
        inner_hint = set(content_words(toks(m.group(2)))) if m else set()
        head = ql[:m.start()] if m else ql
        hw = set(content_words(toks(head)))
        goal = next((t for w, t in QTYPE if re.search(rf"\b{w}\b", head)), "person" if ql.startswith("who") else None)
        goal_hint = {w for w, t in QTYPE if re.search(rf"\b{w}\b", head)} | (hw - {"largest", "biggest", "highest"})
        if inner_type: steps.append((inner_type, inner_hint))
        if goal: steps.append((goal, goal_hint))
        return anchors, steps

    # 4. поиск по шагам
    def reason(self, q, beam=25, verbose=False):
        p = self.parse(q)
        if not p: return None
        anchors, steps = p
        if not steps: return None
        frontier = [(0.0, a, []) for a in anchors]
        for (ty, hint) in steps:
            nxt = []
            for score, node, chain in frontier:
                tx = self.etype.get(node, "?")
                for i in self.by_x.get(node, [])[:400]:
                    X, rel, Y, src = self.facts[i]
                    for n in self.names_in(Y):
                        if self.etype.get(n) != ty or n == node: continue
                        s = self.edge_strength(tx, ty, rel, Y, n)
                        hit = len(hint & (set(toks(rel.replace("_", " "))) | set(toks(Y)))) / max(1, len(hint)) if hint else 0
                        # «capital»: ребро должно содержать capital (вопрос про столицу, не любой город)
                        if ty == "city" and "capital" in hint and "capital" not in rel and "capital" not in Y.lower(): continue
                        nxt.append((score + s + 1.5 * hit, n, chain + [i]))
            # дедуп по узлу: лучший score
            best = {}
            for s, n, c in nxt:
                if n not in best or s > best[n][0]: best[n] = (s, n, c)
            frontier = sorted(best.values(), key=lambda t: -t[0])[:beam]
            if not frontier: return None
        s, n, chain = frontier[0]
        return dict(answer=n, score=round(s, 2), chain=[f"{self.facts[j][0]} —{self.facts[j][1]}→ {self.facts[j][2][:55]}" for j in chain],
                    alternatives=[f[1] for f in frontier[1:4]])
