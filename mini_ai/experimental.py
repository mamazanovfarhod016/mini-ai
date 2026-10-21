"""Mini AI — ЭКСПЕРИМЕНТАЛЬНАЯ МОДЕЛЬ «память без градиента» (21.10.2026). Один вход для всего:
  ask(q)        — ответ по фактам: рамка (1 шаг) → 2 шага через мост → отказ «не знаю». Показывает факты-источники.
  tell(x)       — рассказать про сущность своими словами (speak.py) из её фактов.
  learn(text)   — прочитать текст: предложения → факты (reader.py) → память. Сразу доступно для ask.
  act(request)  — действие инструментами (agent.py): приёмы, генофонд, проверка.
  code(task, fname, tests) — написать функцию из образцов (coder.py), проверить запуском.
  remember/history — память разговора (последние реплики → контекст для «it/он/это»).
  exp40 (диалог, mini_ai/dialog.py): переспрос вместо «не знаю», нить разговора (эллипсис/местоимения/yes-no),
  разные формы ответа, инициатива. ask() → text (реплика целиком), answer (коротко), kind (answer|clarify|unknown).
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
            if len(s) < 15: continue
            if len(s) > 400: s = s[:400]                          # очень длинное — читаем начало (там главное)
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
    def __init__(self, school=True, wiki=False, wiki_limit=300000, state_dir=None, dialog=True, seed=None):
        self.kb = KnowledgeBase(); self.history = []; self.use_dialog = dialog; self.seed = seed; self.state_dir = state_dir or os.path.join(RUNS, "experimental_state")
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
        if os.path.exists(learned): self._load_learned(open(learned).read())
        self.boot = dict(facts=len(self.kb.facts), seconds=round(time.time() - t0, 1), sources=dict(self.kb.sources.most_common(5)))
        self._agent = None; self._coder = None
        from mini_ai.think import Thinker
        self.thinker = Thinker(self.kb); self.dialog = None; self.new_session()

    def new_session(self):
        """новая нить разговора (сервер держит по одной на sid)"""
        from mini_ai.dialog import Dialog
        self.history = []; self.dialog = Dialog(self.kb, seed=self.seed); return self.dialog

    def _load_learned(self, text):
        """learned.txt: блоки «# wiki: Title» читаются с темой статьи (иначе заголовок склеивался с первым
        предложением и «It is bordered by…» теряло субъект); остальное — как «выучено»."""
        for i, chunk in enumerate(re.split(r"^# wiki: (.+)$", text, flags=re.M)):
            if i % 2 == 1: title = chunk.strip(); continue
            if i == 0: src = "выучено"
            else: src = title
            if chunk.strip(): self.kb.read_text(chunk, src)

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

    def ask(self, q, internet=True):
        if not self.use_dialog:
            out = self._ask_core(self._resolve(q), internet); out["kind"] = "answer" if out.get("answer") else "unknown"; return out
        return self._ask_dialog(q, internet)

    def _ask_core(self, q, internet=True, fr=None):
        fr = fr or parse_question(q)
        out = dict(question=q, frame=fr, read=[])
        if internet and fr.get("atype") == "place" and fr["X"]:
            # «Where is X» — лучший ответ в первом предложении статьи о самом X; читаем её один раз
            title = " ".join(w.capitalize() for w in fr["X"])
            if title not in self.kb.sources:
                n = self._read_wiki(title)
                if n: out["read"].append(f"{title}: {n} фактов")
        # РАССУЖДЕНИЕ ЦЕПОЧКОЙ (think.py): вопрос с типом ответа («in which country», «who», «where») →
        # путь по фактам до узла нужного типа; показываем шаги. Одношаговый ответ допускается только если тип совпал.
        th = self.thinker.think(q) if re.search(r"\b(?:which|what) (?:country|city|continent|river|company|language|currency)\b", q.lower()) else None
        typed_q = bool(re.search(r"\b(?:which|what) (?:country|city|continent|river|company|language|currency)\b", q.lower()))
        if th:
            out.update(answer=th["answer"], how=f"{len(th['steps'])} шага: " + " → ".join(th["steps"]), facts=th["steps"], confidence=th["score"])
            out["text"] = self._phrase(out, fr); out["kind"] = "answer"
            self.history.append(dict(q=q, a=th["answer"], entity=th["answer"])); return out
        if typed_q and not th and self.thinker.anchors(q):
            # тип ответа задан, цепочки нет → честно: знаю сущность, но связи до нужного типа не нашла (не подменять поиском по словам)
            a = self.thinker.anchors(q)[0]
            out.update(answer=None, how="нет цепочки", facts=[], confidence=0.0, kind="unknown",
                       text=f"I know {a.title()}, but I can't trace it to a {self.thinker.goal(q)} from what I've read. Tell me the link and I'll remember.")
            return out
        r = self._one_step(fr)
        if not r:
            r2 = self._two_step(fr)
            if r2:
                s, ans, t, F1, F2 = r2
                out.update(answer=ans, how=f"2 шага через «{t}»", facts=[f"{F1[3]} —{F1[1]}→ {F1[4]}", f"{F2[3]} —{F2[1]}→ {F2[4]}"], confidence=round(min(1.0, s / 2.0), 2))
                r = None
            elif internet:
                # НЕ ЗНАЮ → ИДУ В ИНТЕРНЕТ: ищу статьи Википедии по словам вопроса, читаю, пробую снова
                for title in self._wiki_titles(fr):
                    n = self._read_wiki(title)
                    if n: out["read"].append(f"{title} (+{n} фактов)")
                    r = self._one_step(fr)
                    if r: break
                if not r and out["read"]:
                    r2 = self._two_step(fr)
                    if r2:
                        s, ans, t, F1, F2 = r2
                        out.update(answer=ans, how=f"прочитала в интернете, 2 шага через «{t}»", facts=[f"{F1[3]} —{F1[1]}→ {F1[4]}", f"{F2[3]} —{F2[1]}→ {F2[4]}"], confidence=round(min(1.0, s / 2.0), 2))
        if r:
            s, ans, fact = r
            how = "1 шаг" if not out["read"] else "прочитала в интернете, 1 шаг"
            out.update(answer=ans, how=how, facts=[f"{fact[0]} —{fact[1]}→ {fact[2]}"], confidence=round(min(1.0, s / 1.8), 2))
        if "answer" not in out: out.update(answer=None, how="не знаю" + (" (в интернете тоже не нашла)" if out["read"] else ""), facts=[], confidence=0.0)
        out["text"] = self._phrase(out, fr)
        ent = out["answer"] if out.get("answer") else (" ".join(fr["X"][:2]) if fr["X"] else None)
        if not self.use_dialog: self.history.append(dict(q=q, a=out["answer"], entity=ent))
        return out

    # ---------- exp40: слой диалога ----------
    def _ask_dialog(self, q, internet=True):
        from mini_ai.dialog import _is_question, entity_phrase, strip_greeting, YES, NO
        from mini_ai.speak import small_talk
        d = self.dialog; conv = d.conv; q0 = q.strip(); notes = []
        pend = conv.pending()
        # 0. реплика-собеседник (hi / thanks / how are you / who are you) — без поиска фактов
        st = small_talk(q0, conv.rng, len(self.kb.facts))
        if st and not (pend and YES.match(q0)):
            conv.push(q=q0, frame=None, entity=None, answer=None, kind="chat", pending=None, key=None)
            return dict(question=q0, frame=None, read=[], answer=None, kind="chat", how="small talk: " + st[0], facts=[], confidence=1.0, text=st[1], note=None)
        q0 = strip_greeting(q0)
        # 1. реплика после переспроса: yes/no/выбор варианта/утверждение-обучение
        if pend:
            fact, note = d.answer_pending(q0, pend)
            if fact:
                X, rel, Y, src = fact
                return self._reply_fact(q0, X, rel, Y, src, 1.0, how="из переспроса: " + note, facts=[f"{X} —{rel}→ {Y}"], frame=None)
            if note in ("отказ от вариантов", "класс сущности", "подтверждение, но вариантов нет"):
                name = pend.get("name") or "it"
                text = (f"Then tell me one sentence about {name} and I'll remember it." if note != "класс сущности"
                        else f"Okay. Tell me one sentence about {name} and I'll remember it.")
                conv.push(q=q0, frame=None, entity=pend.get("entity"), answer=None, kind="clarify", pending=dict(pend, options=[]), key=None)
                return dict(question=q0, frame=None, read=[], answer=None, kind="clarify", how=note, facts=[], confidence=0.0, text=text, note=None)
        if YES.match(q0) or NO.match(q0) or len(toks(q0)) == 0:
            conv.push(q=q0, frame=None, entity=None, answer=None, kind="chat", pending=None, key=None)
            return dict(question=q0, frame=None, read=[], answer=None, kind="chat", how="реплика без вопроса", facts=[], confidence=1.0, text="Okay. What would you like to ask?", note=None)
        # 2. утверждение (не вопрос) → выучить и сразу ответить тем, что поняла
        if not _is_question(q0) and len(q0.split()) >= 3:
            facts = [f for f in self.kb.reader.read(q0, "выучено")[0] if "@" not in f[1]]
            if facts:
                self.learn(q0)
                X, rel, Y = facts[0]
                r = self._reply_fact(q0, X, rel, Y, "выучено", 1.0, how=f"выучила {len(facts)} факт(а)", facts=[f"{X} —{rel}→ {Y}"], frame=None, learned=True)
                r["learned"] = len(facts); return r
        # 2b. одно имя без вопроса («Tashkent?») → что именно интересует
        if d.bare_entity(q0):
            c = d.clarify(q0, None)
            if c["kind"] == "clarify":
                conv.push(q=q0, frame=None, entity=c["pending"]["name"], answer=None, kind="clarify", pending=c["pending"], key=None)
                return dict(question=q0, frame=None, read=[], answer=None, kind="clarify", how="имя без вопроса → переспрос", facts=[], confidence=0.0, text=c["text"], note=None)
        # 3. нить: эллипсис и местоимения — кандидаты по очереди, берётся первый с ответом
        fr = parse_question(q0); out = None
        for q1, fr1, note in d.resolve(q0, fr):
            if fr1 is None and q1 != q0: fr1 = parse_question(q1)
            if fr1 is None: fr1 = fr
            out = self._ask_core(q1, internet, fr=fr1)
            if out.get("answer"): break
        if note: notes.append(note)
        if out.get("answer") and out["how"].startswith("2 шага") and out["confidence"] < 0.6:
            g = d.guess(q1, out)
            if g:
                conv.push(q=q1, frame=fr1, entity=g["pending"]["name"], answer=None, kind="clarify", pending=g["pending"], key=None)
                return dict(question=q1, frame=fr1, read=out["read"], answer=None, kind="clarify", how="слабая цепочка → догадка-переспрос", facts=out["facts"], confidence=out["confidence"], text=g["text"], note=", ".join(notes) or None)
            out["answer"] = None                                            # мост через случайное слово — не ответ
        if out.get("answer"):
            X, rel, Y, src = self._fact_of(out)
            r = self._reply_fact(q1, X, rel, Y, src, out["confidence"], how=out["how"], facts=out["facts"], frame=fr1, read=out["read"], two_step=out if len(out["facts"]) > 1 else None)
            r["question"] = q1; r["note"] = ", ".join(notes) or None; return r
        # 4. не знаю → переспрос из того, что понято
        c = d.clarify(q1, fr1)
        ents = d.entities_in(q1)
        conv.push(q=q1, frame=fr1, entity=(c["pending"] or {}).get("name"), answer=None, kind=c["kind"], pending=c["pending"], key=None)
        return dict(question=q1, frame=fr1, read=out["read"], answer=None, kind=c["kind"], how="не знаю → " + ("переспрос" if c["kind"] == "clarify" else "уточнение"),
                    facts=[], confidence=0.0, text=c["text"], note=", ".join(notes) or None)

    def _fact_of(self, out):
        m = re.match(r"^(.*?) —(.*?)→ (.*)$", out["facts"][-1] if len(out["facts"]) > 1 else out["facts"][0])
        X, rel, Y = m.groups() if m else (out["answer"], "be", "")
        src = None
        for k in self.kb.by_x.get(" ".join(toks(X)), []):
            f = self.kb.facts[k]
            if f[1] == rel and f[4] == Y: src = f[5]; break
        return X, rel, Y, src

    def _reply_fact(self, q, X, rel, Y, src, confidence, how, facts, frame, read=None, two_step=None, learned=False):
        """факт → реплика: форма по контексту (повтор → другая + «As I said»), инициатива — ещё факт/встречный вопрос"""
        d = self.dialog; conv = d.conv
        key = f"{X.lower()}|{rel}|{Y.lower()}"; qkey = " ".join(toks(q))
        pron, etype = d.persona(X)
        tense = "past" if (etype == "person" and rel not in ("be", "have")) else "present"
        if two_step: text, style = two_step["text"], "two-step"
        else:
            if learned: text, style = "Got it. " + __import__("mini_ai.speak", fromlist=["phrase"]).phrase(X, rel, Y, "direct", tense), "learned"
            else: text, style = d.say(X, rel, Y, src, key, qkey, confidence, tense)
        answer = Y if (frame and frame.get("slot") == "Y") or (frame is None and conv.last and conv.last.get("pending") and conv.last["pending"].get("want")) else X
        if frame and frame.get("slot") == "X": answer = X
        elif frame and frame.get("slot") == "Y": answer = Y
        elif learned: answer = X
        ini = None if learned else d.initiative(X, rel, Y, set(toks(answer)))
        ini_pend = None
        if ini: text = text + " " + ini[0]; ini_pend = ini[1]; ini = ini[0]
        ent = X if X.lower() not in ("it", "this", "they") else conv.last_entity()
        from mini_ai.speak import split_prop
        sp = split_prop(Y) if rel == "be" else None
        topic = sp[1] if sp and frame and frame.get("slot") == "X" else None            # «capital of Kazakhstan» → тема Kazakhstan
        conv.push(q=q, frame=frame, entity=ent, topic=topic, answer=answer, kind="answer", style=style, pending=ini_pend, key=key, qkey=qkey)
        self.history.append(dict(q=q, a=answer, entity=ent))
        return dict(question=q, frame=frame, read=read or [], answer=answer, kind="answer", how=how, facts=facts, confidence=confidence,
                    text=text, style=style, initiative=ini, note=None)

    def _phrase(self, out, fr):
        """ответ СВОИМИ СЛОВАМИ: из факта (X, rel, Y) собирается предложение по грамматике связи (speak.py),
        а не копируется исходный текст. Для 2 шагов — два предложения: мост и вывод."""
        from mini_ai.speak import verb, clean_y
        if not out.get("answer"): return None
        facts = []
        for f in out["facts"]:
            m = re.match(r"^(.*?) —(.*?)→ (.*)$", f)
            if m: facts.append(m.groups())
        sents = []
        for X, rel, Y in facts:
            y = clean_y(Y) or Y.strip(" ,.")
            if rel.startswith("be_called"): s = f"{X} is called {y}"
            elif "_" in rel and not rel.startswith("be_"): s = f"The {rel.replace('_', ' ')} of {X} is {y}"
            else:
                v = verb(rel, "present") or rel
                last = X.split()[-1].lower()
                plural = (last.endswith("s") and not last.endswith(("ss", "us", "is")) and (len(X.split()) > 1 or X[:1].islower())) or last in ("people", "children", "men", "women")
                if plural: v = {"is": "are", "has": "have"}.get(v, rel)
                s = f"{X} {v} {y}"
            sents.append(s[0].upper() + s[1:].rstrip(".") + ".")
        ans = out["answer"]
        if len(sents) == 1: return sents[0]
        return " ".join(sents[:2]) + f" So the answer is {ans}."

    # ---------- интернет: Википедия по словам вопроса ----------
    def _wiki_titles(self, fr, k=2):
        """кандидаты-статьи: имена с заглавной из вопроса → поиск Википедии по содержательным словам"""
        try:
            from tools.exp24_wiki_crawler import api
        except Exception: return []
        words = (fr["Y"] or []) + (fr["X"] or [])
        query = " ".join(w for w in words if w not in ("capital", "largest", "biggest", "called", "name", "type", "kind"))[:80]
        if not query.strip(): return []
        try:
            d = api(dict(action="query", list="search", srsearch=query, srlimit=k))
            return [h["title"] for h in d["query"]["search"]]
        except Exception: return []

    def _read_wiki(self, title):
        if title in self.kb.sources: return 0
        try:
            from tools.exp24_wiki_crawler import fetch_article, clean_extract
            real, text, _ = fetch_article(title)
            if real in self.kb.sources: return 0
            n = self.kb.read_text(clean_extract(text)[:20000], real)
            with open(os.path.join(self.state_dir, "learned.txt"), "a") as f: f.write(f"\n# wiki: {real}\n" + clean_extract(text)[:20000] + "\n")
            return n
        except Exception: return 0

    # ---------- речь своими словами ----------
    def tell(self, X):
        key = " ".join(toks(X)); ks = self.kb.by_x.get(key, [])
        if not ks:                                                    # частичное совпадение по первому слову
            ks = [k for k in self.kb.by_word.get(toks(X)[0], []) if self.kb.facts[k][0] and toks(X)[0] in self.kb.facts[k][0]][:20] if toks(X) else []
        fs = [(self.kb.facts[k][3], self.kb.facts[k][1], self.kb.facts[k][4]) for k in ks]
        read = []
        if len(fs) < 3:                                              # мало знаю → читаю статью в интернете
            n = self._read_wiki(X)
            if n: read.append(f"{X} (+{n} фактов)"); ks = self.kb.by_x.get(key, []); fs = [(self.kb.facts[k][3], self.kb.facts[k][1], self.kb.facts[k][4]) for k in ks]
        if not fs: return dict(text=None, facts=0, how="не знаю", read=read)
        self.history.append(dict(q=f"tell {X}", a=None, entity=X))
        return dict(text=compose(X, fs), facts=len(fs), how="из фактов" + (" (прочитала в интернете)" if read else ""), read=read)

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
