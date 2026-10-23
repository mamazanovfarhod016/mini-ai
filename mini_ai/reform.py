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
        for m in os.environ.get("MINI_AI_REFORM_OFF", "").split(","):          # абляция: MINI_AI_REFORM_OFF=typ,sub
            if m.strip() in self.on: self.on[m.strip()] = False
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

    CL_RX = re.compile(r"\s*(?:[,;:—]|\s-\s|\(|\))\s*")
    def clauses(self, q, present, frac=0.6):
        """ПРИДАТОЧНЫЕ: части вопроса между запятыми без вопросного слова («…, used to locate underwater objects, …»,
        «Because sound waves must move through a medium, …», «…, causing death of brain cells») — если ≥ frac их слов отсутствует
        в предложении, вся часть — уточнитель. → [(intro, absent_words, absent)]"""
        parts = [p for p in self.CL_RX.split(q.strip().rstrip("?")) if p and p.strip()]
        if len(parts) < 2: return []
        out = []
        for p in parts:
            t = toks(p)
            if any(w in QW for w in t): continue
            cw = [w for w in content_words(t) if w not in DET and w not in INTRO and w not in self.mind.META]
            if not cw: continue
            ab = [w for w in cw if not present(w)]
            out.append((t[0] if t[0] in INTRO or t[0] in STOP else "clause", ab, len(ab) >= frac * len(cw)))
        return out

    @staticmethod
    def _keys(intro, n):
        c = intro_class(intro); n = min(n, 3)
        return (f"{intro}|{n}", intro, f"{c}|{n}", c, "*")
    @staticmethod
    def _ckeys(intro):
        return (f"clause|{intro}", "clause", "*")

    VERBISH = set("occurs serves utilizes contains leads controls carries uses has makes forms causes produces results refers means describes allows "
                  "helps provides includes involves requires holds takes gives undergo undergoes transports secretes regulates reflects separates "
                  "surrounds covers lacks consists exists lives feeds eats moves does do can will would should may might".split())
    def typeword(self, q):
        """слово-тип: голова именной группы сразу после what/which (до первого служебного слова или глагола): «what PART of»,
        «what SEQUENCE is», «what kind of muscle CELLS undergo», «what flying MAMMAL», «what medical EMERGENCY occurs»"""
        ql = q.lower().rstrip("?")
        m = re.search(r"\b(?:what|which)\s+(?:(?:type|kind|sort|class|category|group)s?\s+of\s+)?((?:[a-z'\-]+\s*){1,4})", ql)
        if not m: return []
        bad = STOP | INTRO | QW | {"many", "much", "type", "kind", "sort", "term", "name"}
        grp = []
        for w in m.group(1).split()[:3]:
            if w in bad or w in self.VERBISH or w.endswith(("ed", "ing")): break
            grp.append(w)
        return [grp[-1]] if grp and len(grp[-1]) >= 3 else []

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
            wpresent = lambda w: bool(self._forms(w) & wf) or w in self.mind.META
            for intro, words, absent in self.runs(q, wpresent):
                if absent:
                    for k in self._keys(intro, len(words)): self.phr[k][1] += 1
            for intro, ab, absent in self.clauses(q, wpresent):
                if absent:
                    for k in self._ckeys(intro): self.phr[k][1] += 1
        for intro, ab, absent in self.clauses(q, present):
            if absent:
                for k in self._ckeys(intro): self.phr[k][0] += 1
                grew.append(f"clause «{intro} {' '.join(ab[:4])}…» was a qualifier")
        for t in self.typeword(q):
            # по слову и по ФОРМЕ («*»: любое слово сразу после what/which) — опыт переносится на новые слова-типы
            if not present(t): self.typ[t][0] += 1; self.typ["*"][0] += 1; grew.append(f"«{t}» names the answer type")
            else: self.typ[t][1] += 1; self.typ["*"][1] += 1
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
        return self._rate_keys(self._keys(intro, n)[:-1], min_n)
    def _rate_keys(self, keys, min_n=2):
        A, B = self.phr.get("*", (0, 0))
        if A + B < 4: return False
        g = (A + 1) / (A + B + 2)
        for k in keys:
            a, b = self.phr.get(k, (0, 0))
            if a + b >= min_n: return a >= 1 and (a + 1) / (a + b + 2) >= g * self.K
        return False
    K = 1.2

    JUNK_RX = re.compile(r"^(?:'s|'t|don|doesn|isn|aren|won|can|ll|re|ve|[a-z]{1,2})$")
    def qualifiers(self, q, absent_words):
        """слова вопроса, которые на ЭТОМ предложении можно не требовать: они образуют целиком отсутствующий кусок
        формы, выученной как уточнение. absent_words: множество слов вопроса, которых нет в предложении."""
        out = {w for w in absent_words if self.JUNK_RX.match(w) or w in INTRO}      # обломки токенов («'s», «don») и служебные — не термины
        if not self.on.get("phr", True) or not self.phr: return out
        present = lambda w: w not in absent_words
        for intro, words, absent in self.runs(q, present):
            if absent and self._rate(intro, len(words)): out |= set(words)
        for intro, ab, absent in self.clauses(q, present):
            if absent and self._rate_keys(self._ckeys(intro)[:-1]): out |= set(ab)
        return out

    def type_words(self, q):
        if not self.on.get("typ", True): return set()
        out = set(); X, N = self.typ.get("*", (0, 0))
        general = X >= 4 and X >= 1.5 * N                 # форма «what <слово>» вообще: слово-тип обычно отсутствует в предложении
        for t in self.typeword(q):
            x, need = self.typ.get(t, (0, 0))
            if (x >= 2 and x >= need) or (general and x >= need): out.add(t)
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
