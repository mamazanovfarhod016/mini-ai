"""exp60 — few-shot рассуждение (как GPT-3), без градиента: правило вывода ПОКАЗЫВАЮТ примерами, она его обобщает.

Сейчас типы связей и их композиции (born_in∘capital_of → born_in_country) заданы в коде cot.py — ~15 штук.
Гипотеза: правило композиции можно ИНДУЦИРОВАТЬ из 2–3 показанных примеров: пользователь даёт факты и
один-два ответа-образца («Ann is the teacher of Bob. Bob is the teacher of Carl. → Ann is the teacher of the
teacher of Carl» или просто «Q: Who is the grand-teacher of Carl? A: Ann»). Из образца извлекается СХЕМА:
путь по связям от якоря к ответу (teacher_of → teacher_of) и имя нового отношения. Дальше эта схема применяется
к новым сущностям без дополнительных примеров.
Механизм: связи читаются из текста общим читателем «X is the <rel> of Y» / «X <verb> Y» (reader) — любые, не из
списка; граф сущностей; из примера (вопрос, ответ) ищем кратчайший путь якорь→ответ по связям — это и есть схема;
имя отношения берётся из вопроса («grand-teacher», «in which country»). Применение: тот же путь от нового якоря.
Числа (до запуска): 3 новых отношения (grand-teacher, author's country, grandparent), по 2 примера на каждое,
тест — 30 новых вопросов на отношение, вымышленные имена. Успех ≥0.8 при контроле cot.py = 0 (связей нет в списке).
Также: 1 пример вместо 2 — сколько даст (one-shot).
"""
import os, sys, re, random, json, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.expanduser("~/Developer/mini-ai-runs/learning-core")
rng = random.Random(0)
SYL = ["zor", "va", "ni", "kel", "mar", "ta", "shi", "lun", "dra", "vek", "o", "ri", "sun", "bel", "ka", "mo", "re", "th", "an", "ul"]
_used = set()
def name(k=2):
    while True:
        n = "".join(rng.choice(SYL) for _ in range(k)).capitalize()
        if n not in _used: _used.add(n); return n

# ---------- общий читатель связей: любые «X is the R of Y», «X R-ed Y», «X is a T in Y» ----------
REL_RX = [re.compile(r"^(?P<X>[A-Z][\w\-]+(?: [A-Z][\w\-]+)?) (?:is|was) the (?P<R>[a-z][\w\- ]{1,25}?) of (?P<Y>[A-Z][\w\-]+(?: [A-Z][\w\-]+)?)\.?$"),
          re.compile(r"^(?P<X>[A-Z][\w\-]+(?: [A-Z][\w\-]+)?) (?P<R>wrote|founded|teaches|taught|owns|leads|married|discovered|built) (?P<Y>[A-Z][\w\-]+(?: [A-Z][\w\-]+)?)\.?$"),
          re.compile(r"^(?P<X>[A-Z][\w\-]+(?: [A-Z][\w\-]+)?) (?:is|was) (?:a|an) (?P<R>[a-z][\w\-]+) (?:in|of|from) (?P<Y>[A-Z][\w\-]+(?: [A-Z][\w\-]+)?)\.?$"),
          re.compile(r"^(?P<X>[A-Z][\w\-]+(?: [A-Z][\w\-]+)?) (?:is|was) born in (?P<Y>[A-Z][\w\-]+(?: [A-Z][\w\-]+)?)\.?$")]

class Graph:
    def __init__(self): self.edges = collections.defaultdict(list)   # X → [(rel, Y)]
    def read(self, text):
        n = 0
        for s in re.split(r"(?<=[.!?])\s+", text.strip()):
            for rx in REL_RX:
                m = rx.match(s.strip())
                if m:
                    d = m.groupdict(); rel = d.get("R", "born_in").lower().replace(" ", "_")
                    if rel.endswith("s") and rel in ("teaches", "owns", "leads"): rel = rel[:-1]
                    self.edges[d["X"]].append((rel, d["Y"])); self.edges[d["Y"]].append(("inv:" + rel, d["X"])); n += 1; break
        return n
    def paths(self, a, b, max_len=3):
        """все пути связей a→b длиной ≤ max_len (без обратных петель)"""
        out = []; stack = [(a, [], {a})]
        while stack:
            node, path, seen = stack.pop()
            if node == b and path: out.append(path); continue
            if len(path) >= max_len: continue
            for rel, nxt in self.edges.get(node, []):
                if nxt not in seen: stack.append((nxt, path + [rel], seen | {nxt}))
        return out
    def follow(self, a, schema):
        """применить схему (список связей) от якоря a → множество концов"""
        cur = {a}
        for rel in schema:
            nxt = set()
            for n in cur:
                for r, y in self.edges.get(n, []):
                    if r == rel: nxt.add(y)
            cur = nxt
            if not cur: break
        return cur

class FewShot:
    """правила вывода из примеров: (шаблон вопроса → схема пути)"""
    def __init__(self, g): self.g = g; self.rules = {}
    def _names(self, q):
        """имена в вопросе (многословные тоже), без вопросительных слов; якорь — последнее"""
        qs = re.sub(r"^(?:Who|What|Which|Where|In|How)\b\s*", "", q.strip().rstrip("?"))
        return [n for n in re.findall(r"\b[A-Z][\w\-]+(?: [A-Z][\w\-]+)*", qs) if n in self.g.edges]
    def qkey(self, q):
        """ключ вопроса = вопрос без имён (имена → _): «who is the grand-teacher of _»"""
        return re.sub(r"\b[A-Z][\w\-]+(?: [A-Z][\w\-]+)*", "_", re.sub(r"^(?:Who|What|Which|Where|In|How)\b", lambda m: m.group(0).lower(), q.strip().rstrip("?"))).lower()
    def anchors(self, q): return re.findall(r"\b[A-Z][\w\-]+\b", q.strip().rstrip("?"))[0:1] if re.findall(r"\b[A-Z][\w\-]+\b", q) else []
    def show(self, q, answer):
        """один пример: вопрос + правильный ответ → кандидаты схем; пересечение с прежними примерами того же вопроса"""
        a = self._names(q)
        if not a: return None
        cands = {tuple(p) for p in self.g.paths(a[-1], answer)}
        k = self.qkey(q)
        self.rules[k] = (self.rules[k] & cands) if k in self.rules and self.rules[k] & cands else cands
        return self.rules[k]
    def answer(self, q):
        k = self.qkey(q); a = self._names(q)
        if k not in self.rules or not a: return None, []
        votes = collections.Counter(); used = []
        for schema in sorted(self.rules[k], key=len):                       # короткие схемы — надёжнее
            for y in self.g.follow(a[-1], schema): votes[y] += 1.0 / len(schema); used.append(schema)
        if not votes: return None, []
        best = votes.most_common(1)[0][0]
        return best, [f"schema {'→'.join(s)}" for s in sorted(self.rules[k], key=len)[:2]]

def world(n=40):
    facts, qa = [], collections.defaultdict(list)
    for _ in range(n):
        A, B, C = name(), name(), name(); book, country, city = name() + " Chronicles", name(3), name()
        P, Q, R = name(), name(), name()
        facts += [f"{A} is the teacher of {B}.", f"{B} is the teacher of {C}.",                      # grand-teacher
                  f"{P} wrote {book}.", f"{P} is a writer from {country}.",                           # author's country
                  f"{Q} is the parent of {R}.", f"{R} is the parent of {name()}."]
        qa["grand"].append((f"Who is the grand-teacher of {C}?", A))
        qa["author_country"].append((f"Which country is the author of {book} from?", country))
        grandchild = facts[-1].split(" is the parent of ")[1].rstrip(".")
        qa["grandparent"].append((f"Who is the grandparent of {grandchild}?", Q))
    return facts, qa

if __name__ == "__main__":
    facts, qa = world(); g = Graph(); n = g.read(" ".join(facts)); print(f"фактов прочитано {n}, сущностей {len(g.edges)}")
    res = {}
    for shots in (1, 2, 3):
        fs = FewShot(g); ok = tot = 0; per = {}
        for rel, pairs in qa.items():
            train, test = pairs[:shots], pairs[shots:shots + 30]
            for q, a in train: fs.show(q, a)
            hit = sum(fs.answer(q)[0] == a for q, a in test); per[rel] = round(hit / len(test), 2); ok += hit; tot += len(test)
        res[f"shots{shots}"] = dict(acc=round(ok / tot, 3), per=per); print(f"{shots}-shot: {ok}/{tot} = {ok/tot:.2f}  по отношениям {per}")
    fs = FewShot(g)
    for rel, pairs in qa.items():
        for q, a in pairs[:2]: fs.show(q, a)
    for rel, pairs in qa.items():
        q, a = pairs[5]; ans, how = fs.answer(q); print(f"  {q} -> {ans} (верно {a}) | {how}")
    # контроль: cot.py не знает этих связей
    try:
        from mini_ai.experimental import ExperimentalMind
        m = ExperimentalMind(school=False, wiki=False, state_dir="/tmp/exp60"); m.learn(" ".join(facts)); c = 0
        for rel, pairs in qa.items():
            for q, a in pairs[2:12]: c += a.lower() in ((m.ask(q, internet=False).get("text") or "").lower())
        res["control_cot"] = round(c / 30, 3); print(f"контроль cot.py (связи не в списке): {c}/30")
    except Exception as e: print("контроль пропущен:", e)
    json.dump(res, open(os.path.join(OUT, "exp60_fewshot_reasoning.json"), "w"), indent=1); print(res)
