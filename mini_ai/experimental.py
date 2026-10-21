"""Mini AI — ЭКСПЕРИМЕНТАЛЬНАЯ МОДЕЛЬ «память без градиента» (21.10.2026). Один вход для всего:
  ask(q)        — ответ по фактам: рамка (1 шаг) → 2 шага через мост → отказ «не знаю». Показывает факты-источники.
  tell(x)       — рассказать про сущность своими словами (speak.py) из её фактов.
  learn(text)   — прочитать текст: предложения → факты (reader.py) → память. Сразу доступно для ask.
  act(request)  — действие инструментами (agent.py): приёмы, генофонд, проверка.
  code(task, fname, tests) — написать функцию из образцов (coder.py), проверить запуском.
  remember/history — память разговора (последние реплики → контекст для «it/он/это»).
Память: факты (список), словарь IDF, образцы кода, приёмы — всё на диске, не забывается.
Источники знаний при старте: школьный учебник (SciQ train, 50k предл.) + Simple Wikipedia (опционально, 1.16M фактов).
"""
import os, re, json, time, collections
import numpy as np
from mini_ai.memory_v2 import toks, content_words
from mini_ai.reader import Reader
from mini_ai.question import parse_question, answer_by_frame
from mini_ai.speak import compose

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.expanduser("~/Developer/mini-ai-runs/memory-core")


class KnowledgeBase:
    """факты + индексы. Факт = (Xw:set, rel, Yw:set, X:str, Y:str, src:str)"""
    def __init__(self):
        self.facts = []; self.by_word = collections.defaultdict(list); self.by_x = collections.defaultdict(list)
        self.df = collections.Counter(); self.n_docs = 0; self.reader = Reader(); self.sources = collections.Counter()

    def idf(self, w): return float(np.log((1 + self.n_docs) / (1 + self.df.get(w, 0))) + 1.0)

    def add_fact(self, X, rel, Y, src):
        Xw, Yw = set(content_words(toks(X))), set(content_words(toks(Y)))
        if not (Xw or Yw): return
        k = len(self.facts); self.facts.append((Xw, rel, Yw, X, Y, src))
        for w in Xw | Yw: self.by_word[w].append(k)
        self.by_x[" ".join(toks(X))].append(k); self.sources[src] += 1

    def read_text(self, text, src="user"):
        n = 0
        for s in re.split(r"(?<=[.!?])\s+(?=[A-ZА-Я0-9\"])", re.sub(r"\s+", " ", text)):
            s = s.strip()
            if not (15 < len(s) < 400): continue
            ws = set(content_words(toks(s))); self.n_docs += 1
            for w in ws: self.df[w] += 1
            for X, rel, Y in self.reader.read(s, src)[0]:
                if "@" not in rel: self.add_fact(X, rel, Y, src); n += 1
        return n

    def load_facts_jsonl(self, path, limit=None):
        n = 0
        with open(path) as f:
            for line in f:
                X, rel, Y, src = json.loads(line)[:4]
                self.add_fact(X, rel, Y, src); n += 1
                if limit and n >= limit: break
        self.n_docs += n
        for Xw, rel, Yw, X, Y, src in self.facts[-n:]:
            for w in Xw | Yw: self.df[w] += 1
        return n


class ExperimentalMind:
    def __init__(self, school=True, wiki=False, wiki_limit=300000, state_dir=None):
        self.kb = KnowledgeBase(); self.history = []; self.state_dir = state_dir or os.path.join(RUNS, "experimental_state")
        os.makedirs(self.state_dir, exist_ok=True)
        t0 = time.time()
        if school:
            try:
                import pandas as pd
                df = pd.read_parquet(os.path.join(ROOT, "data/school/sciq/data/train-00000-of-00001.parquet"))
                for sup in df["support"].dropna(): self.kb.read_text(str(sup), "учебник")
            except Exception as e: print("учебник не загружен:", e)
        if wiki and os.path.exists(os.path.join(RUNS, "simplewiki_full/facts.jsonl")):
            self.kb.load_facts_jsonl(os.path.join(RUNS, "simplewiki_full/facts.jsonl"), wiki_limit)
        learned = os.path.join(self.state_dir, "learned.txt")
        if os.path.exists(learned): self.kb.read_text(open(learned).read(), "выучено")
        self.boot = dict(facts=len(self.kb.facts), seconds=round(time.time() - t0, 1), sources=dict(self.kb.sources.most_common(5)))
        self._agent = None; self._coder = None

    # ---------- понимание + рассуждение ----------
    def _resolve(self, q):
        """местоимения в вопросе → последняя сущность из истории"""
        if self.history and re.search(r"\b(it|this|that|he|she|they|оно|это|он|она)\b", q.lower()):
            last = self.history[-1].get("entity")
            if last: q = re.sub(r"\b(it|this|that|he|she|they)\b", last, q, count=1, flags=re.I)
        return q

    def _one_step(self, fr):
        known = set(fr["Y"] if fr["slot"] == "X" else fr["X"])
        if not known: return None
        cand = set.intersection(*[set(self.kb.by_word.get(w, [])) for w in known])
        if not cand: return None
        res = answer_by_frame(fr, [self.kb.facts[k] for k in cand], {w: self.kb.idf(w) for w in known})
        return res[0] if res else None

    def _two_step(self, fr, max_f1=60, max_t=8):
        known = set(fr["Y"] if fr["slot"] == "X" else fr["X"])
        if len(known) < 2: return None
        idf = {w: self.kb.idf(w) for w in known}; kidf = sum(idf.values())
        cnt = collections.Counter()
        for w in known:
            for k in self.kb.by_word.get(w, []): cnt[k] += idf[w]
        best = None
        for k1, c in cnt.most_common(max_f1):
            c1 = c / kidf
            if not (0.4 <= c1 < 0.99): continue
            F1 = self.kb.facts[k1]; w1 = F1[0] | F1[2]; rest = known - w1
            # мост: редкий термин F1, который ОПРЕДЕЛЁН в памяти (есть факт с X = термин) — не любое слово
            bridges = [t for t in w1 - known if self.kb.idf(t) > 2.5 and t in self.kb.by_x]
            for t in sorted(bridges, key=lambda t: -self.kb.idf(t))[:max_t]:
                for k2 in self.kb.by_word.get(t, []):
                    if k2 == k1: continue
                    F2 = self.kb.facts[k2]; w2 = F2[0] | F2[2]
                    if not rest <= w2: continue
                    for F in (F2, F1):
                        for ans, aw in ((F[3], F[0]), (F[4], F[2])):
                            if aw and not (aw & known) and ans.lower() not in ("it", "this", "they", "these"):
                                s = c1 + sum(idf[w] for w in rest & w2) / kidf
                                if not best or s > best[0]: best = (s, ans, t, F1, F2)
        return best

    def ask(self, q):
        q = self._resolve(q); fr = parse_question(q)
        out = dict(question=q, frame=fr)
        r = self._one_step(fr)
        if r:
            s, ans, fact = r; out.update(answer=ans, how="1 шаг", facts=[f"{fact[0]} —{fact[1]}→ {fact[2]}"], confidence=round(min(1.0, s / 1.8), 2))
        else:
            r2 = self._two_step(fr)
            if r2:
                s, ans, t, F1, F2 = r2
                out.update(answer=ans, how=f"2 шага через «{t}»", facts=[f"{F1[3]} —{F1[1]}→ {F1[4]}", f"{F2[3]} —{F2[1]}→ {F2[4]}"], confidence=round(min(1.0, s / 2.0), 2))
            else: out.update(answer=None, how="не знаю", facts=[], confidence=0.0)
        ent = out["answer"] if out.get("answer") else (" ".join(fr["X"][:2]) if fr["X"] else None)
        self.history.append(dict(q=q, a=out["answer"], entity=ent))
        return out

    # ---------- речь своими словами ----------
    def tell(self, X):
        key = " ".join(toks(X)); ks = self.kb.by_x.get(key, [])
        if not ks:                                                    # частичное совпадение по первому слову
            ks = [k for k in self.kb.by_word.get(toks(X)[0], []) if self.kb.facts[k][0] and toks(X)[0] in self.kb.facts[k][0]][:20] if toks(X) else []
        fs = [(self.kb.facts[k][3], self.kb.facts[k][1], self.kb.facts[k][4]) for k in ks]
        if not fs: return dict(text=None, facts=0, how="не знаю")
        self.history.append(dict(q=f"tell {X}", a=None, entity=X))
        return dict(text=compose(X, fs), facts=len(fs), how="из фактов")

    # ---------- обучение ----------
    def learn(self, text, src="выучено"):
        n = self.kb.read_text(text, src)
        with open(os.path.join(self.state_dir, "learned.txt"), "a") as f: f.write(text.strip() + "\n")
        return dict(new_facts=n, total=len(self.kb.facts))

    # ---------- действия и код ----------
    def act(self, request, check=None):
        if self._agent is None:
            from mini_ai.agent import Agent; self._agent = Agent(os.path.join(self.state_dir, "agent"))
        if check is None: check = lambda out: bool(out and str(out).strip() and "rror" not in str(out)[:40])   # без проверки: непустой вывод без ошибки
        out, how = self._agent.run(request, check); self._agent.save()
        return dict(output=out, how=how)

    def code(self, task, fname, tests):
        if self._coder is None:
            from mini_ai.coder import CodeMemory, Coder
            cm = CodeMemory(os.path.join(self.state_dir, "code_memory.json"))
            tb = os.path.join(ROOT, "data/code/textbook.py")
            if os.path.exists(tb) and not cm.items: cm.learn_from_code(open(tb).read(), "textbook")
            self._coder = Coder(cm)
        r = self._coder.solve(task, fname, tests); self._coder.mem.save(); return r
