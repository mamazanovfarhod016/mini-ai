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
from mini_ai.speak import phrase, split_prop, STYLES, entity_type, clean_y

QW = ("what", "which", "who", "where", "when", "how", "why")
YES = re.compile(r"^(?:yes|yeah|yep|sure|ok|okay|да|ага|the first|first|former|right|correct)[\s.!]*$", re.I)
NO = re.compile(r"^(?:no|nope|нет|neither|none|not that|wrong)[\s.!]*$", re.I)
PRON = re.compile(r"\b(it|its|he|his|she|her|they|their|there|this|that|оно|он|она|это)\b", re.I)
ELLIPSIS = re.compile(r"^(?:and|what about|how about|or)\s+(?:the\s+|its\s+|his\s+|her\s+|their\s+)?(.+?)[?!.]*$", re.I)
PROP_WORDS = {"population", "capital", "height", "area", "length", "size", "currency", "language", "climate", "depth",
              "width", "age", "name", "largest city", "biggest city", "main city", "founder", "president", "symbol", "density",
              "mass", "temperature", "color", "colour", "speed"}
GENERIC_X = {"it", "this", "that", "they", "these", "those", "he", "she", "what", "which", "who", "there", "here"}
CLASS_WORDS = {"city": "a city", "town": "a town", "capital": "a city", "country": "a country", "republic": "a country",
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
        return t.get("pending") if t and t.get("kind") in ("clarify", "unknown") else None

    def times_asked(self, key):
        return sum(1 for t in self.turns if t.get("key") == key and t.get("kind") == "answer")

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
            elif rel == "be" and clean_y(Y): props.append(("what it is", (X, rel, Y, src)))
            elif rel not in ("be", "have") and clean_y(Y): props.append((f"what it {rel}s" if rel[-1] not in "s" else f"what it {rel}", (X, rel, Y, src)))
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
        """→ (q', fr', note) — эллипсис и местоимения через историю. fr может быть None (не разобрался)."""
        ql = q.strip().rstrip("?!. ").lower()
        last_fr = self.conv.last_frame(); last_ent = self.conv.last_entity()
        m = ELLIPSIS.match(q.strip()) or (re.match(r"^(?:its|his|her|their)\s+(.+?)[?!.]*$", q.strip(), re.I))
        if m and last_fr:
            rest = m.group(1).strip(); rw = content_words(toks(rest))
            known = list(last_fr["Y"] if last_fr["slot"] == "X" else last_fr["X"])
            prop_old = [w for w in known if w in PROP_WORDS]
            if rw and all(w in PROP_WORDS or w in ("the", "a") for w in rw) and last_ent:
                # «And its population?» → прошлая сущность, новое свойство
                nq = f"What is the {' '.join(rw)} of {last_ent}?"
                return nq, None, "эллипсис: свойство"
            if rw and prop_old:
                # «What about Kazakhstan?» → прошлое свойство, новая сущность
                nq = f"What is the {' '.join(prop_old)} of {entity_phrase(rw, self.kb)}?"
                return nq, None, "эллипсис: сущность"
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
            nq = re.sub(r"\b(its|his|her|their)\b", ent + "'s", q, count=1, flags=re.I) if re.search(r"\b(its|his|her|their)\b", q, re.I) else q
            if nq != q:
                nq = re.sub(r"^(what|which) (is|was|are|were) " + re.escape(ent) + r"'s (.+?)[?]*$", rf"what is the \3 of {ent}?", nq, flags=re.I)
                return nq, None, "местоимение (притяжательное)"
            nq = re.sub(r"\b(it|he|she|they|this|that|there)\b", ent, q, count=1, flags=re.I)
            if nq != q: return nq, None, "местоимение"
        return q, fr, None

    # ---------- 1. переспрос ----------
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
                    text = f"I don't know the {wanted} of {name} yet. I do know "
                    text += (", ".join(f"its {o}" for o in options[:2]) if options else "a little about it") + f" — do you mean that? Or tell me the {wanted} in one sentence and I'll remember."
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

    def answer_pending(self, q, pend):
        """реплика после переспроса: yes → первый вариант; «its population» → выбранный вариант; no → попросить предложение"""
        ql = q.strip().lower()
        if YES.match(ql):
            if pend.get("options"): return pend["options"][0][1], "подтверждение переспроса"
            return None, "подтверждение, но вариантов нет"
        if NO.match(ql):
            return None, "отказ от вариантов"
        rw = set(content_words(toks(ql)))
        for p, f in pend.get("options", []):
            if rw & set(content_words(toks(p))): return f, "выбор варианта переспроса"
        if rw <= {"person", "place", "thing", "city", "country"} and rw: return None, "класс сущности"
        return None, None

    # ---------- 3. речь: выбор формы ----------
    def choose_style(self, key, confidence):
        prev = self.conv.last_style(key); n = self.conv.times_asked(key)
        if confidence < 0.6: return "unsure", ""
        if n:                                                          # повторный вопрос — другая форма + «as I said»
            styles = [s for s in STYLES if s != prev] or list(STYLES)
            return self.conv.rng.choice(styles), "As I said, "
        return self.conv.rng.choice(STYLES), ""

    def say(self, X, rel, Y, src, key, confidence, tense="present"):
        style, prefix = self.choose_style(key, confidence)
        return phrase(X, rel, Y, style, tense, src, prefix), style

    # ---------- 4. инициатива ----------
    def initiative(self, X, rel, Y, answer_words):
        """ещё один факт про ту же сущность (не тот же) или встречный вопрос — только если есть что сказать"""
        if self.conv.i - self.conv.last_initiative < 2: return None
        ent = content_words(toks(X))
        others = [(x, r, y, s) for x, r, y, s in self.facts_of(ent) if (r, clean_y(y)) != (rel, clean_y(Y)) and clean_y(y)]
        others = [f for f in others if not (set(content_words(toks(f[2]))) <= set(content_words(toks(Y))))]
        if not others: return None
        x, r, y, s = self.conv.rng.choice(others[:5])
        self.conv.last_initiative = self.conv.i
        sp = split_prop(y)
        if r == "have" and sp and self.conv.rng.random() < 0.5:
            return f"Do you want to know its {sp[0]}?"
        yy = clean_y(y)
        if r == "be": return f"By the way, it is also {yy}."
        if r == "have": return f"By the way, it also has {yy}."
        from mini_ai.speak import verb
        return f"By the way, it also {verb(r, 'present') or r} {yy}."
