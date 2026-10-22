"""reform.py — ПЕРЕЧИТЫВАНИЕ ВОПРОСА, выученное из исправлений (без градиента).

Счётчики доверия/смелости лишь переставляют кандидатов; они не меняют того, КАК читается вопрос. Здесь из каждого
исправления «no, it's X» (и подтверждения «yes») выводится, как вопрос НАДО БЫЛО прочитать, — и правило переносится
на НОВЫЕ вопросы той же формы:
  (а) УТОЧНИТЕЛИ: кусок вопроса, которого в ответном предложении не было («within an incomplete digestive system»,
      «such as the breakdown of…», «what do YOU CALL…»). Запоминается не слово, а ФОРМА куска: вводное служебное слово
      + длина («within|3», «such|2», «do|2»). У новой формы вопроса кусок той же формы, целиком отсутствующий в предложении,
      считается уточнением, а не обязательным словом → предложение проходит порог покрытия.
  (б) СЛОВО-ТИП ОТВЕТА: слово сразу после what/which («what PART of…», «what SEQUENCE is…», «what KIND of feedback»),
      которого в предложении нет, — говорит, ЧЕМ должен быть ответ, а не что искать. Счётчик по слову.
  (в) ЗАМЕНЫ: слово вопроса ↔ слово предложения по близости смысла (make↔produce, holds↔held) — расширение поиска.
Всё читаемо (reform.json), всё объяснимо: «я прочитала „within …“ как уточнение, потому что в 4 из 5 исправлений такой
кусок в предложении отсутствовал».
"""
import os, re, json, collections
from mini_ai.memory_v2 import toks, content_words, STOP

QW = {"what", "which", "who", "where", "when", "how", "why"}
DET = set("a an the its their this these those some most all many each every other another such own".split())
# вводные слова, с которых начинается кусок-уточнение (предлоги, союзы, местоимения); остальные служебные — в STOP
INTRO = set("in on at of for from by with within without during before after while whether because despite through across along "
            "between among into onto under over such like via per than when where if unless until since although though except "
            "including regarding about around behind below above beneath beside toward towards upon out up down off you we they it "
            "do does did is are was were can could should would will may might has have had be been being also not".split())
VERB_TAIL = ("ed", "ing")
PREP = set("in on at of for from by with within without during before after through across along between among into onto under over "
           "via per than toward towards upon about around behind below above beneath beside except including regarding".split())
COP = set("is are was were be been being do does did can could should would will may might has have had not also".split())
CONJ = set("and or but that which while whether because despite such like if unless until since although though when where as".split())
PRON = set("you we they it".split())
def intro_class(intro):
    """класс вводного слова — чтобы опыт переносился с «within» на «during»: prep / cop / conj / pron / qw / start (^) / mid (_)"""
    if intro in PREP: return "prep"
    if intro in COP: return "cop"
    if intro in CONJ: return "conj"
    if intro in PRON: return "pron"
    if intro in QW: return "qw"
    return intro if intro in ("^", "_") else "other"


class Reform:
    def __init__(self, mind, state_dir=None):
        self.mind = mind; self.path = os.path.join(state_dir, "reform.json") if state_dir else None
        # КОНТРАСТ: кусок формы S отсутствовал в ПРАВИЛЬНОМ предложении (уточнитель, [0]) vs отсутствовал в лучшем НЕПРАВИЛЬНОМ
        # предложении ([1] — его отсутствие и делало предложение неправильным). Только так форма куска отличает «можно пропустить»
        # («within …», «such as …») от «нельзя» (термин-подлежащее).
        self.phr = collections.defaultdict(lambda: [0, 0])       # «intro|len», «intro» → [absent-in-right, absent-in-wrong]
        self.typ = collections.defaultdict(lambda: [0, 0])       # слово после what/which → [absent-in-right, absent-in-wrong]
        self.sub = collections.defaultdict(collections.Counter)  # слово вопроса → Counter(слово предложения)
        self.n = 0; self.log = []
        self.on = dict(phr=True, typ=True, sub=True)
        self.load()

    # ---------- хранение ----------
    def load(self):
        if not self.path or not os.path.exists(self.path): return
        try: d = json.load(open(self.path))
        except Exception: return
        for k, v in d.get("phr", {}).items(): self.phr[k] = v
        for k, v in d.get("typ", {}).items(): self.typ[k] = v
        for k, v in d.get("sub", {}).items(): self.sub[k] = collections.Counter(v)
        self.n = d.get("n", 0); self.log = d.get("log", [])[-40:]

    def save(self):
        if not self.path: return
        json.dump(dict(phr=dict(self.phr), typ=dict(self.typ), sub={k: dict(v) for k, v in self.sub.items()}, n=self.n, log=self.log[-40:]),
                  open(self.path, "w"), ensure_ascii=False, indent=1)

    # ---------- разбор вопроса на куски ----------
    def _forms(self, w): return self.mind._forms(w)

    def runs(self, q, present):
        """куски вопроса: максимальные цепочки содержательных слов одной судьбы (все есть в предложении / всех нет),
        служебные слова внутри не рвут цепочку. → [(intro, words, absent)]; intro = ближайшее служебное слово перед куском"""
        qt = toks(q); out = []; run = []; intro = "^"; last_fn = "^"; state = None
        for i, w in enumerate(qt):
            fn = w in STOP or w in INTRO or w in DET or w in QW
            if fn:
                if w not in DET:                            # служебное слово (не артикль) начинает новый кусок: «within …», «such as …», «by …»
                    if run: out.append((intro, run, state)); run = []
                    last_fn = w
                continue
            ab = not present(w)
            if run and ab != state:
                out.append((intro, run, state)); run = []
            if not run: intro = last_fn if last_fn != "" else "_"; state = ab
            run.append(w); last_fn = ""                     # после содержательного слова вводное обнуляется («_» = шёл кусок)
        if run: out.append((intro, run, state))
        return out

    @staticmethod
    def _keys(intro, n):
        c = intro_class(intro); n = min(n, 3)
        return (f"{intro}|{n}", intro, f"{c}|{n}", c, "*")

    def typeword(self, q):
        """слово-тип сразу после what/which: «what PART of», «what SEQUENCE is», «what kind of FEEDBACK» (не «what is the X of»)"""
        ql = q.lower().rstrip("?")
        m = re.search(r"\b(?:what|which)\s+(?:type|kind|sort|class|category)s?\s+of\s+([a-z]+)", ql)
        if m: return [m.group(1)]
        m = re.search(r"\b(?:what|which)\s+(?:[a-z]+\s+)?([a-z]+)\s+(?:of|is|are|was|were|do|does|did|in|that|has|have|can|will|would|for)\b", ql)
        if m and all(x not in STOP and x not in INTRO and x not in QW for x in m.group(0).split()[1:-1]) and len(m.group(1)) > 2: return [m.group(1)]
        return []

    # ---------- учёба ----------
    def _sentence(self, q, right):
        """→ (правильное предложение: с ответом, покрывает больше всего слов вопроса; лучшее НЕПРАВИЛЬНОЕ: без ответа, с наибольшим покрытием)"""
        kb = self.mind.kb; rl = right.lower().strip(); qw = [w for w in content_words(toks(q)) if w not in self.mind.META]
        if not rl or not qw: return None, None
        def pack(sent):
            sw = set(content_words(toks(sent))); sf = set().union(*[self._forms(x) for x in sw]) if sw else set()
            return (sum(1 for w in qw if self._forms(w) & sf), sent, sw, sf)
        best = None
        for si in range(len(kb.sents) - 1, max(-1, len(kb.sents) - 40000), -1):
            sent = kb.sents[si][0]
            if rl not in sent.lower(): continue
            p = pack(sent)
            if not best or p[0] > best[0]: best = p
        if not best or best[0] == 0: return None, None
        cnt = collections.Counter()
        for w in qw:
            for si in kb.sent_by_word.get(w, [])[:400]: cnt[si] += kb.idf(w)
        wrong = None
        for si, _ in cnt.most_common(12):
            sent = kb.sents[si][0]
            if rl in sent.lower(): continue
            p = pack(sent)
            if not wrong or p[0] > wrong[0]: wrong = p
        return best, wrong

    def learn(self, q, right, confirmed=False):
        """исправление (или подтверждение): выровнять вопрос с правильным предложением (и с лучшим неправильным) →
        формы уточнителей (контраст), слово-тип, замены"""
        b, wrong = self._sentence(q, right)
        if not b: return []
        cov, sent, sw, sf = b; rw = set(content_words(toks(right))); grew = []
        present = lambda w: bool(self._forms(w) & sf) or w in self.mind.META
        rs = self.runs(q, present)
        for intro, words, absent in rs:
            if not absent: continue
            for k in self._keys(intro, len(words)): self.phr[k][0] += 1
            grew.append(f"«{intro} {' '.join(words)}» was a qualifier")
        if wrong and wrong[0] > 0:
            wf = wrong[3]
            for intro, words, absent in self.runs(q, lambda w: bool(self._forms(w) & wf) or w in self.mind.META):
                if absent:
                    for k in self._keys(intro, len(words)): self.phr[k][1] += 1
        for t in self.typeword(q):
            if not present(t): self.typ[t][0] += 1; grew.append(f"«{t}» names the answer type")
            elif wrong and not (self._forms(t) & wrong[3]): self.typ[t][1] += 1
        # замены: отсутствующее слово вопроса ↔ слово предложения, которого нет в вопросе (не ответ), по смыслу
        M = getattr(self.mind, "meaning", None)
        q_abs = [w for _, ws, ab in rs if ab for w in ws if len(w) > 3]
        qf = set().union(*[self._forms(w) for w in content_words(toks(q))]) if q_abs else set()
        s_only = [w for w in sw if len(w) > 3 and not (self._forms(w) & qf) and w not in rw]
        if M is not None and q_abs and s_only and len(s_only) <= 8:
            for a in q_abs:
                best = max(((M.similar(a, b_) or 0, b_) for b_ in s_only), default=(0, None))
                if best[0] > 0.3: self.sub[a][best[1]] += 1; grew.append(f"{a}↔{best[1]}")
        self.n += 1; self.log.append(dict(q=q[:80], right=right[:40], grew=grew[:6], ok=confirmed)); self.save()
        return grew

    # ---------- применение ----------
    def _rate(self, intro, n, min_n=2):
        """кусок формы (intro, n) — уточнитель? Сравниваем P(правильное | кусок отсутствует) у формы с общей: форма, у которой
        отсутствие куска реже делает предложение неправильным, чем в среднем, — уточнитель. Перенос: слово → класс слова → общее."""
        A, B = self.phr.get("*", (0, 0))
        if A + B < 4: return False
        g = (A + 1) / (A + B + 2)
        for k in self._keys(intro, n)[:-1]:
            a, b = self.phr.get(k, (0, 0))
            if a + b >= min_n: return a >= 1 and (a + 1) / (a + b + 2) >= g * self.K
        return False
    K = 1.2

    def qualifiers(self, q, absent_words):
        """слова вопроса, которые на ЭТОМ предложении можно не требовать: они образуют целиком отсутствующий кусок
        формы, выученной как уточнение. absent_words: множество слов вопроса, которых нет в предложении."""
        if not self.on.get("phr", True) or not self.phr: return set()
        out = set()
        for intro, words, absent in self.runs(q, lambda w: w not in absent_words):
            if absent and self._rate(intro, len(words)): out |= set(words)
        return out

    def type_words(self, q):
        if not self.on.get("typ", True): return set()
        out = set()
        for t in self.typeword(q):
            x, need = self.typ.get(t, (0, 0))
            if x >= 2 and x >= need: out.add(t)
        return out

    def subs(self, w, min_n=2):
        if not self.on.get("sub", True): return []
        return [b for b, n in self.sub.get(w, {}).items() if n >= min_n]

    def report(self):
        A, B = self.phr.get("*", (0, 0)); g = (A + 1) / (A + B + 2)
        qual = sorted(((k, v) for k, v in self.phr.items() if "|" not in k and k != "*" and v[0] + v[1] >= 2 and (v[0] + 1) / (v[0] + v[1] + 2) >= g * self.K), key=lambda kv: -kv[1][0])[:8]
        typ = [t for t, (x, n) in self.typ.items() if x >= 2 and x >= n]
        return (f"reading rules: {len(qual)} qualifier shapes (" + ", ".join(f"«{k}…» {v[0]}/{v[0]+v[1]}" for k, v in qual) + f"); answer-type words: {', '.join(typ[:8])}; "
                f"substitutions: {sum(len(v) for v in self.sub.values())}; from {self.n} corrections")
