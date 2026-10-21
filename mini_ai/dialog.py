"""Слой ДИАЛОГА (exp40, 21.10.2026) поверх памяти без градиента. Без LLM, без эмбеддеров — правила + факты памяти.
Четыре способности:
  1. ПЕРЕСПРОС вместо «не знаю»: вопрос не разобрался / факт не найден → уточняющий вопрос из того, что понято
     (сущность, её свойства из РЕАЛЬНЫХ фактов): «Do you mean the capital of Uzbekistan, or its largest city?»
     Сущность неизвестна совсем → «I don't know X yet. Is it a person, a place or a thing? Or tell me one sentence…»
  2. НИТЬ РАЗГОВОРА: последние N ходов (рамка + сущности). Местоимения (it/he/she/there), эллипсис
     («And Germany?» → прошлая рамка с новой сущностью; «And its population?» → прошлая сущность с новым свойством),
     «yes/no» как ответ на переспрос (не вопрос!), утверждение в ответ на переспрос → выучить и сразу ответить.
  3. РАЗНООБРАЗИЕ РЕЧИ: один факт → разные формы (speak.phrase): прямая, инверсия, «As far as I know», «I read in
     the textbook that». Повторный вопрос → другая форма + «As I said»; низкая уверенность → оговорка.
  4. ИНИЦИАТИВА: после ответа — ещё один факт про ту же сущность («By the way, …») или встречный вопрос
     («Do you want to know its population?») — только из фактов в памяти, не чаще чем раз в 2 хода.
Состояние разговора (Conversation) — список ходов: dict(q, frame, entity, answer, kind, style, pending).
"""
import re, collections, random
from mini_ai.memory_v2 import toks, content_words
from mini_ai.reader import lemma
from mini_ai.speak import phrase, split_prop, STYLES, entity_type, clean_y, small_talk

GREET_PREFIX = re.compile(r"^(?:hi|hello|hey|привет|please|пожалуйста)[,!\s]+(?=\S)", re.I)


def strip_greeting(q):
    """«Hi, what is the capital…» → «what is the capital…»"""
    return GREET_PREFIX.sub("", q.strip()) if len(q.split()) > 2 else q.strip()

QW = ("what", "which", "who", "where", "when", "how", "why")
YES = re.compile(r"^(?:yes|yeah|yep|sure|ok|okay|да|ага|the first|first|former|right|correct)[\s.!]*$", re.I)
NO = re.compile(r"^(?:no|nope|нет|neither|none|not that|wrong)[\s.!]*$", re.I)
PRON = re.compile(r"\b(it|its|he|his|she|her|they|their|there|this|that|оно|он|она|это)\b", re.I)
ELLIPSIS = re.compile(r"^(?:and|what about|how about|or)\s+(?:the\s+|its\s+|his\s+|her\s+|their\s+)?(.+?)[?!.]*$", re.I)
PROP_WORDS = {"population", "capital", "height", "area", "length", "size", "currency", "language", "climate", "depth",
              "width", "age", "name", "largest city", "biggest city", "main city", "founder", "president", "symbol", "density",
              "mass", "temperature", "color", "colour", "speed"}
GENERIC_X = {"it", "this", "that", "they", "these", "those", "he", "she", "what", "which", "who", "there", "here"}
CLASS_WORDS = {"person": "a person", "city": "a city", "town": "a town", "capital": "a city", "country": "a country", "republic": "a country",
               "river": "a river", "mountain": "a mountain", "planet": "a planet", "element": "a chemical element",
               "scientist": "a scientist", "physicist": "a scientist", "poet": "a poet", "writer": "a writer", "king": "a ruler",
               "organelle": "a part of the cell", "organ": "an organ", "animal": "an animal", "plant": "a plant", "gas": "a gas"}


def _is_question(q):
    ql = q.strip().lower()
    return ql.endswith("?") or ql.split(" ")[0] in QW or ql.split(" ")[0] in ("is", "are", "was", "were", "do", "does", "did", "can")


def _cap(s): return s[0].upper() + s[1:] if s else s


def entity_phrase(words, kb):
    """имя сущности так, как она записана в памяти (регистр из факта), иначе — слова как есть"""
    key = " ".join(words)
    ks = kb.by_x.get(key)
    if ks: return kb.facts[ks[0]][3]
    return " ".join(w.capitalize() if len(w) > 2 and not w.isdigit() else w for w in words)


class Conversation:
    """нить разговора: последние N ходов + ожидающий переспрос"""
    def __init__(self, n=8, seed=None):
        self.turns = collections.deque(maxlen=n); self.rng = random.Random(seed)
        self.last_initiative = -3; self.i = 0

    def push(self, **t):
        t["i"] = self.i; self.i += 1; self.turns.append(t); return t

    @property
    def last(self): return self.turns[-1] if self.turns else None

    def last_entity(self):
        for t in reversed(self.turns):
            if t.get("entity"): return t["entity"]
        return None

    def last_frame(self):
        for t in reversed(self.turns):
            if t.get("frame") and (t["frame"].get("Y") or t["frame"].get("X")): return t["frame"]
        return None

    def pending(self):
        t = self.last
        return t.get("pending") if t else None

    def topic_entities(self):
        """сущности последних ходов по порядку: ответ последнего хода, затем тема его рамки («of Kazakhstan»), затем раньше"""
        out = []
        for t in reversed(self.turns):
            for e in (t.get("entity"), t.get("topic")):
                if e and e not in out: out.append(e)
        return out

    def times_asked(self, qkey):
        """сколько раз этот же ВОПРОС уже задавали (не факт: «Where is it?» после «longest river» — другой вопрос)"""
        return sum(1 for t in self.turns if t.get("qkey") == qkey and t.get("kind") == "answer")

    def last_style(self, key):
        for t in reversed(self.turns):
            if t.get("key") == key and t.get("style"): return t["style"]
        return None


class Dialog:
    def __init__(self, kb, conv=None, seed=None):
        self.kb = kb; self.conv = conv or Conversation(seed=seed)

    # ---------- знания про сущность ----------
    def facts_of(self, name_words, limit=12):
        """факты, где X = сущность (по ключу by_x), иначе — факты, где все слова сущности есть в X"""
        key = " ".join(name_words); ks = list(self.kb.by_x.get(key, []))
        if not ks and name_words:
            cand = set.intersection(*[set(self.kb.by_word.get(w, [])) for w in name_words]) if name_words else set()
            ks = [k for k in cand if set(name_words) <= self.kb.facts[k][0] and len(self.kb.facts[k][0]) <= len(name_words) + 1][:limit]
        out = []
        for k in ks[:limit]:
            Xw, rel, Yw, X, Y, src = self.kb.facts[k]
            if X.lower() in GENERIC_X: continue
            out.append((X, rel, Y, src))
        return out

    def mentioned(self, name_words, limit=12):
        """факты, где сущность стоит в Y («Tashkent be the capital of Uzbekistan» для Uzbekistan)"""
        if not name_words: return []
        cand = set.intersection(*[set(self.kb.by_word.get(w, [])) for w in name_words])
        out = []
        for k in sorted(cand)[:200]:
            Xw, rel, Yw, X, Y, src = self.kb.facts[k]
            if set(name_words) <= Yw and X.lower() not in GENERIC_X and X[:1].isupper(): out.append((X, rel, Y, src))
            if len(out) >= limit: break
        return out

    def props_of(self, name_words):
        """свойства сущности, о которых есть факты: [(prop, fact)] — «capital of Uzbekistan», «population»"""
        props = []
        for X, rel, Y, src in self.mentioned(name_words):
            sp = split_prop(Y)
            if rel == "be" and sp and content_words(toks(sp[1])) == list(name_words): props.append((sp[0], (X, rel, Y, src)))
        for X, rel, Y, src in self.facts_of(name_words):
            sp = split_prop(Y)
            if rel == "have" and sp: props.append((sp[0], (X, rel, Y, src)))
            elif rel == "be" and sp and sp[0] in PROP_WORDS | {"largest city", "longest river", "highest mountain"}: props.append((f"{sp[0]} of {sp[1]}", (X, rel, Y, src)))
            elif rel == "be" and clean_y(Y):
                yl = Y.lower().strip()
                if re.match(r"^(?:famous|known|notable|responsible) for\b", yl): props.append((f"what it is {yl.split()[0]} for", (X, rel, Y, src)))
                elif re.match(r"^(?:a|an|the)\s+(?:\w+\s){0,2}\w+\s+(?:in|on|near)\s+[A-Z]", Y.strip()): props.append(("where it is", (X, rel, Y, src)))
                else: props.append(("what it is", (X, rel, Y, src)))
            elif rel not in ("be", "have") and clean_y(Y):
                from mini_ai.speak import verb
                props.append((f"what it {verb(rel, 'present') or rel}", (X, rel, Y, src)))
        seen = set(); out = []
        for p, f in props:
            if p not in seen: seen.add(p); out.append((p, f))
        return out

    def entities_in(self, q):
        """имена с заглавной в вопросе (не первое вопросное слово) → список слов-ключей"""
        words = re.findall(r"[A-Za-z][\w'\-]*", q)
        ents = []; cur = []
        for i, w in enumerate(words):
            if w[:1].isupper() and not (i == 0 and w.lower() in QW + ("and", "or", "is", "are", "do", "does", "did", "was", "were", "the", "its", "tell", "yes", "no")):
                cur.append(w.lower())
            else:
                if cur: ents.append(cur); cur = []
        if cur: ents.append(cur)
        return [e for e in ents if any(len(w) > 1 for w in e)]

    def known_entity(self, words):
        return bool(self.facts_of(words) or self.mentioned(words))

    # ---------- 2. нить: подстановка рамки / сущности ----------
    def resolve(self, q, fr):
        """→ список кандидатов [(q', fr', note)] — эллипсис и местоимения через историю; первый, что даст ответ, и берётся.
        «And its capital?» после «largest city of Kazakhstan → Almaty»: сначала Almaty, потом тема рамки — Kazakhstan."""
        ql = q.strip().rstrip("?!. ").lower()
        last_fr = self.conv.last_frame(); last_ent = self.conv.last_entity(); ents = self.conv.topic_entities()
        m = ELLIPSIS.match(q.strip()) or (re.match(r"^(?:its|his|her|their)\s+(.+?)[?!.]*$", q.strip(), re.I))
        if m and last_fr:
            rw = content_words(toks(m.group(1)))
            if rw and all(w in PROP_WORDS or w in ("the", "a") for w in rw) and ents:
                return [(f"What is the {' '.join(rw)} of {e}?", None, "эллипсис: свойство") for e in ents[:3]]
        if last_ent and PRON.search(q) and re.search(r"\b(its|his|her|their)\b", q, re.I):
            out = []
            for e in ents[:3]:
                nq = re.sub(r"\b(its|his|her|their)\b", e + "'s", q, count=1, flags=re.I)
                nq = re.sub(r"^(what|which) (is|was|are|were) " + re.escape(e) + r"'s (.+?)[?]*$", rf"what is the \3 of {e}?", nq, flags=re.I)
                out.append((nq, None, "местоимение (притяжательное)"))
            return out
        return [self._resolve_one(q, fr, ql, last_fr, last_ent)]

    def _resolve_one(self, q, fr, ql, last_fr, last_ent):
        m = ELLIPSIS.match(q.strip()) or (re.match(r"^(?:its|his|her|their)\s+(.+?)[?!.]*$", q.strip(), re.I))
        if m and last_fr:
            rest = m.group(1).strip(); rw = content_words(toks(rest))
            known = list(last_fr["Y"] if last_fr["slot"] == "X" else last_fr["X"])
            prop_old = [w for w in known if w in PROP_WORDS]
            if rw and prop_old:
                # «What about Kazakhstan?» → прошлое свойство, новая сущность
                nq = f"What is the {' '.join(prop_old)} of {entity_phrase(rw, self.kb)}?"
                return nq, None, "эллипсис: сущность"
            prev_q = next((t["q"] for t in reversed(self.conv.turns) if t.get("q") and t.get("frame")), None)
            if rw and prev_q:
                # «What about South America?» после «What is the longest river in Africa?» → подставить имя в прошлый вопрос
                old = self.entities_in(prev_q)
                if old:
                    pat = r"\b" + r"\s+".join(re.escape(w) for w in old[-1]) + r"\b"
                    nq = re.sub(pat, entity_phrase(rw, self.kb), prev_q.rstrip("?"), count=1, flags=re.I) + "?"
                    if nq.lower() != prev_q.lower(): return nq, None, "эллипсис: сущность в прошлом вопросе"
            if rw and last_fr["slot"] == "X" and last_fr["Y"]:
                # «And Mars?» после «What does it have?» → та же связь с новой сущностью
                old_ent = [w for w in known if w not in PROP_WORDS]
                nfr = dict(last_fr); nfr["Y"] = [w for w in known if w not in old_ent] + rw
                return q, nfr, "эллипсис: та же рамка"
        if last_ent and PRON.search(q):
            ent = last_ent
            if re.search(r"\bthere\b", q, re.I) and re.match(r"^how many people live there", ql):
                return f"What is the population of {ent}?", None, "местоимение there"
            if re.match(r"^how many people live in (it|there)", ql): return f"What is the population of {ent}?", None, "местоимение it"
            # местоимение — только как самостоятельное подлежащее/дополнение: «What is it?», «Where does he live?».
            # «socket THAT houses», «so hard» (that=союз), «when IT sends» в длинном школьном вопросе — не местоимение.
            qw = toks(q)
            relative = re.search(r"\b(?:[a-z]+)\s+(?:that|which)\s+[a-z]+", q, re.I) and re.search(r"\b(?:that|which)\b", q, re.I)
            if len(qw) <= 9 and not relative:
                nq = re.sub(r"\b(it|he|she|they|this|that|there)\b", ent, q, count=1, flags=re.I)
                if nq != q: return nq, None, "местоимение"
        return q, fr, None

    # ---------- 1. переспрос ----------
    def bare_entity(self, q):
        """«Tashkent?» / «Samarkand» — одно имя без вопроса → спросить, что именно интересует (если фактов ≥ 2)"""
        qs = q.strip().rstrip("?!. ")
        if not qs or len(qs.split()) > 3 or qs.split()[0].lower() in QW or not all(w[:1].isupper() for w in qs.split()): return None
        e = [w.lower() for w in toks(qs)]
        return e if e and len(self.props_of(e)) >= 2 and len(self.facts_of(e)) >= 2 else None

    def clarify(self, q, fr):
        """вопрос без ответа → dict(kind, text, pending) построенный из того, что понято"""
        ents = self.entities_in(q)
        want = set(content_words(toks(q))) & PROP_WORDS
        for e in ents:
            props = self.props_of(e)
            if props:
                name = entity_phrase(e, self.kb)
                etype = entity_type([(X, r, Y) for X, r, Y, s in self.facts_of(e)])[1] if self.facts_of(e) else "thing"
                options = [p for p, f in props][:3]
                if want:                                                     # знаю сущность, но не это свойство
                    wanted = " ".join(sorted(want))
                    opts = [o for o in options if not o.startswith(("what it", "where it"))][:2]
                    text = f"I don't know the {wanted} of {name} yet. I do know "
                    text += (", ".join(f"its {o}" for o in opts) if opts else "what it is") + f" — do you mean that? Or tell me the {wanted} in one sentence and I'll remember."
                    pend_opts = [(p, f) for p, f in props if p != "what it is"][:2] + [(p, f) for p, f in props if p == "what it is"]
                    return dict(kind="clarify", text=text, pending=dict(entity=" ".join(e), name=name, options=pend_opts[:3], want=sorted(want)))
                else:
                    text = f"I know {name}" + (f" is {CLASS_WORDS.get(etype, 'something I have facts about')}" if etype != "thing" else "") + " — what do you want to know about it: " + ", ".join(options) + "?"
                return dict(kind="clarify", text=text, pending=dict(entity=" ".join(e), name=name, options=[(p, f) for p, f in props][:3], want=sorted(want)))
        name = entity_phrase(ents[0], self.kb) if ents else None
        if name:
            return dict(kind="unknown", text=f"I don't know {name} yet. Is it a person, a place or a thing? Or tell me one sentence about it and I'll remember.",
                        pending=dict(entity=" ".join(ents[0]), name=name, options=[], want=sorted(want)))
        cw = content_words(toks(q))
        if cw:
            return dict(kind="unknown", text=f"I don't have a fact about {' '.join(cw)} yet. Tell me one sentence about it and I'll remember.",
                        pending=dict(entity=" ".join(cw[:3]), name=" ".join(cw[:3]), options=[], want=sorted(want)))
        return dict(kind="unknown", text="I didn't understand the question. Ask about one thing, like «What is the capital of Uzbekistan?»", pending=None)

    def guess(self, q, out):
        """слабая 2-шаговая цепочка (уверенность < 0.6) → не ответ, а ДОГАДКА-переспрос, и только если ответ — сущность с фактами
        и цепочка касается сущности вопроса; иначе цепочка выбрасывается (мост через случайное слово)"""
        ans = out.get("answer") or ""
        aw = " ".join(toks(ans))
        if not aw or aw not in self.kb.by_x or len(toks(ans)) > 3: return None
        qents = {w for e in self.entities_in(q) for w in e}
        facts = []
        for f in out["facts"]:
            m = re.match(r"^(.*?) —(.*?)→ (.*)$", f)
            if m: facts.append(m.groups())
        if qents and not any(qents & (set(toks(X)) | set(toks(Y))) for X, rel, Y in facts): return None
        about = [(X, rel, Y, None) for X, rel, Y in facts if " ".join(toks(X)) == aw] or [(X, rel, Y, None) for X, rel, Y in facts]
        from mini_ai.speak import sentence
        why = " and ".join(sentence(X, rel, Y) for X, rel, Y, _ in about[:2])
        text = f"I'm not sure. Do you mean {ans}? I know that {why}."
        return dict(kind="clarify", text=text, pending=dict(entity=aw, name=ans, options=[(p, f) for p, f in [(ans, about[0])]], want=[]))

    def answer_pending(self, q, pend):
        """реплика после переспроса: yes → первый вариант; «its population» → выбранный вариант; no → попросить предложение"""
        ql = q.strip().lower()
        if YES.match(ql):
            if pend.get("options"): return pend["options"][0][1], "подтверждение переспроса"
            return None, "подтверждение, но вариантов нет"
        if NO.match(ql):
            return None, "отказ от вариантов"
        rw = set(content_words(toks(ql)))
        # полный новый вопрос с другой сущностью («What is the capital of Germany?») — это не выбор варианта
        if _is_question(ql) and len(ql.split()) >= 4 and not (rw & set(content_words(toks(pend.get("name") or "")))): return None, None
        for p, f in pend.get("options", []):
            if rw & set(content_words(toks(p))) - {"capital", "city", "country", "population", "largest", "known"}: return f, "выбор варианта переспроса"
            if rw <= set(content_words(toks(p))): return f, "выбор варианта переспроса"
        if rw <= {"person", "place", "thing", "city", "country"} and rw: return None, "класс сущности"
        return None, None

    def persona(self, X):
        """(pron, etype) сущности по её определению в памяти: Isaac Newton → he/person"""
        fs = [(x, r, y) for x, r, y, s in self.facts_of(content_words(toks(X)))]
        return entity_type(fs) if fs else ("it", "thing")

    # ---------- 3. речь: выбор формы ----------
    def choose_style(self, key, qkey, confidence):
        prev = self.conv.last_style(key); n = self.conv.times_asked(qkey)
        if confidence < 0.6: return "unsure", ""
        styles = [s for s in STYLES if s != prev] or list(STYLES)         # тот же факт — другая форма, чем в прошлый раз
        return self.conv.rng.choice(styles), ("As I said, " if n else "")

    def say(self, X, rel, Y, src, key, qkey, confidence, tense="present"):
        style, prefix = self.choose_style(key, qkey, confidence)
        return phrase(X, rel, Y, style, tense, src, prefix), style

    # ---------- 4. инициатива ----------
    def initiative(self, X, rel, Y, answer_words):
        """ещё один факт про ту же сущность (не тот же) или встречный вопрос — только если есть что сказать"""
        if self.conv.i - self.conv.last_initiative < 2: return None
        ent = content_words(toks(X))
        others = [(x, r, y, s) for x, r, y, s in self.facts_of(ent) if (r, clean_y(y)) != (rel, clean_y(Y)) and clean_y(y)]
        others = [f for f in others if not (set(content_words(toks(f[2]))) <= set(content_words(toks(Y))))]
        # «be + причастие» (conquered by…, thought to…) и длинные/обрывочные Y — не для «кстати»
        others = [f for f in others if not (f[1] == "be" and re.match(r"^(?:\w+ed|thought|known|made|born|found|used|called|considered|conquered|located)\b", f[2].strip(), re.I))
                  and len(f[2].split()) <= 14 and not re.search(r"\b(?:figure|table|chapter|below|above|shown)\b", f[2], re.I)]
        if not others: return None
        x, r, y, s = self.conv.rng.choice(others[:5])
        self.conv.last_initiative = self.conv.i
        pron, etype = self.persona(X); poss = {"he": "his", "she": "her"}.get(pron, "its")
        if etype == "person" and pron == "he": pron, poss = X, X + "'s"          # пол из фактов не известен → имя, не «he»
        sp = split_prop(y)
        if r == "have" and sp and self.conv.rng.random() < 0.5:
            return f"Do you want to know {poss} {sp[0]}?", dict(entity=" ".join(ent), name=X, options=[(sp[0], (x, r, y, s))], want=[sp[0]])
        yy = clean_y(y)
        from mini_ai.speak import verb
        past = etype == "person" and r not in ("be", "have")
        if r == "be": t = f"By the way, {pron} {'was' if (etype == 'person') else 'is'} also {yy}."
        elif r == "have": t = f"By the way, {pron} also {'had' if etype == 'person' else 'has'} {yy}."
        else: t = f"By the way, {pron} also {verb(r, 'past' if past else 'present') or r} {yy}."
        return t, None
