"""Читатель (exp30/31/36): предложение → факты (X, rel, Y) + шаблон. Без градиента.
  субъект из контекста: It / Its / The country / The city → тема статьи или последняя названная сущность;
  связь = лемма глагола (открытый класс); Y = аргумент до запятой (числа «1,5 million» не режутся);
  «In YEAR, …» → факт (X, rel@year, YEAR); «X, Y and Z are …» → факт на каждого;
  «Its symbol is Au» → (тема, symbol, Au) — притяжательное: свойство как связь.
"""
import re, collections

VERBS = ("is", "was", "are", "were", "has", "have", "had", "became", "borders", "produces", "consists", "includes",
         "contains", "remains", "lies", "hosts", "hosted", "won", "founded", "established", "built", "serves",
         "operates", "covers", "ranks", "declared", "gained", "joined", "signed", "adopted", "opened", "began",
         "uses", "speaks", "exports", "reached", "comprises", "features", "grew", "stands", "connects", "attracts",
         "lost", "launched", "introduced", "ended", "produced", "operated", "covered", "ranked", "used", "spoke",
         "wrote", "painted", "discovered", "invented", "founded", "created", "composed", "directed", "designed",
         "developed", "led", "ruled", "conquered", "defeated", "killed", "married", "lives", "lived", "died",
         "means", "comes", "came", "makes", "made", "gives", "gave", "takes", "took", "eats", "eat", "grows", "flows")
GENERIC = {"it", "the country", "the city", "the republic", "the region", "the state", "the nation", "he", "she",
           "they", "the government", "this", "the capital", "the town", "the province", "the airline", "the company",
           "the river", "the island", "the planet", "the species", "the band", "the film", "the book", "the game"}
TOPIC_WORDS = {"it", "the city", "the country", "the town", "the river", "the island", "the planet", "the species",
               "the band", "the film", "the book", "the game", "the company"}
PREAMBLE = re.compile(r"^(?:In|By|During|After|Before|Since|According to|Following|At|On|Under|As of|Although|Despite|Today|Historically|Later|Currently|For example|However|Also)\b[^,]{0,60},\s*")
YEAR = re.compile(r"\b(1\d{3}|20\d{2})\b")
_V = "|".join(VERBS)
RX = re.compile(r"^(?P<X>(?:[A-Z][\w'\-]*|the|It|it)(?:\s+(?:[A-Za-z][\w'\-]*)){0,3}?)\s+(?P<V>" + _V + r")\b\s*(?P<Y>(?:[^,.;:()]|,(?=\d)|\.(?=\d)|,(?=\s+(?!and\b|but\b|or\b|which\b|while\b|whereas\b|so\b|yet\b|as\b|because\b|when\b|where\b|who\b|that\b|the\b|a\b|an\b|it\b|he\b|she\b|they\b)[a-z]+\s+[a-z]+)|,(?=\s+(?:and\s+)?[A-Z][\w\-]*\s+(?:to|in|on)\s+the\s+(?:north|south|east|west|northeast|northwest|southeast|southwest))){2,220})")
# Y: запятая допустима только перед цифрой (1,500), точка — внутри числа (3.14)
POSS = re.compile(r"^(?:Its|Their|His|Her)\s+(?P<P>[a-z][\w\s\-]{1,30}?)\s+(?P<V>is|was|are|were)\s+(?P<Y>[^,.;:()]{1,80})")
PLURAL = re.compile(r"^(?P<L>[A-Z][\w'\-]*(?:,\s*[A-Z][\w'\-]*)*(?:,?\s+and\s+[A-Z][\w'\-]*))\s+(?P<V>are|were|have|had)\b\s*(?P<Y>[^,.;:()]{2,80})")
_LEM = {"is": "be", "was": "be", "are": "be", "were": "be", "has": "have", "had": "have", "have": "have",
        "became": "become", "made": "make", "makes": "make", "uses": "use", "used": "use", "means": "mean",
        "wrote": "write", "began": "begin", "died": "die", "won": "win", "came": "come", "comes": "come",
        "took": "take", "takes": "take", "includes": "include", "gave": "give", "gives": "give", "lived": "live",
        "lives": "live", "married": "marry", "led": "lead", "created": "create", "eats": "eat", "grew": "grow",
        "grows": "grow", "joined": "join", "ended": "end", "lost": "lose", "contains": "contain", "built": "build",
        "developed": "develop", "consists": "consist", "borders": "border", "killed": "kill", "discovered": "discover",
        "reached": "reach", "signed": "sign", "ruled": "rule", "covers": "cover", "covered": "cover", "features": "feature",
        "produces": "produce", "produced": "produce", "hosts": "host", "hosted": "host", "opened": "open", "remains": "remain",
        "flows": "flow", "lies": "lie", "introduced": "introduce", "founded": "found", "defeated": "defeat",
        "invented": "invent", "spoke": "speak", "speaks": "speak", "gained": "gain", "declared": "declare",
        "launched": "launch", "stands": "stand", "established": "establish", "painted": "paint", "composed": "compose",
        "directed": "direct", "designed": "design", "conquered": "conquer", "operates": "operate", "operated": "operate",
        "serves": "serve", "ranks": "rank", "ranked": "rank", "adopted": "adopt", "exports": "export", "comprises": "comprise",
        "connects": "connect", "attracts": "attract"}
# прошедшее время для речи (speak.py): лемма → форма
PAST_FORM = {"be": "was", "have": "had", "become": "became", "make": "made", "write": "wrote", "begin": "began",
             "die": "died", "win": "won", "come": "came", "take": "took", "give": "gave", "grow": "grew", "lose": "lost",
             "build": "built", "lead": "led", "speak": "spoke", "found": "founded", "eat": "ate", "flow": "flowed",
             "lie": "lay", "stand": "stood", "mean": "meant", "use": "used", "include": "included", "live": "lived",
             "marry": "married", "create": "created", "produce": "produced", "rule": "ruled", "host": "hosted"}


def lemma(v):
    v = v.lower()
    if v in _LEM: return _LEM[v]
    if v.endswith("ies") and len(v) > 4: return v[:-3] + "y"
    if v.endswith(("ses", "shes", "ches", "xes")): return v[:-2]
    if v.endswith("s") and not v.endswith("ss") and len(v) > 3: return v[:-1]
    return v


# ---- общий читатель: любой субъект (именная группа) + глагол из ОТКРЫТОГО списка + аргумент ----
GEN_VERBS = (r"is called|are called|is known as|are known as|is|are|was|were|has|have|had|can be|contains?|includes?|consists? of|"
             r"produces?|causes?|forms?|makes?|means?|occurs?|results? in|leads? to|comes? from|gets?|uses?|used|refers? to|"
             r"describes?|begins?|ends?|enters?|exhibits?|reproduces?|releases?|absorbs?|reflects?|transmits?|carries|carry|"
             r"requires?|needs?|allows?|helps?|prevents?|affects?|determines?|controls?|regulates?|provides?|stores?|moves?|"
             r"flows?|travels?|increases?|decreases?|changes?|depends? on|becomes?|became|remains?|represents?|measures?|"
             r"shows?|indicates?|involves?|surrounds?|connects?|separates?|divides?|protects?|supports?|covers?|lines?|"
             r"attracts?|repels?|binds?|breaks?|splits?|combines?|joins?|converts?|transfers?|transports?|pumps?|"
             r"grows?|develops?|evolves?|lives?|feeds? on|eats?|kills?|infects?|spreads?|occurs? (?:in|at|when|during)|"
             r"takes? place|is found|are found|is located|are located|is made|are made|is formed|are formed|is caused|are caused|"
             r"is composed|is used|are used|is produced|are produced|is stored|are stored|is released|are released")
GEN_RX = re.compile(r"^(?P<X>(?:[A-Za-z][\w'\-]*)(?:\s+(?:[A-Za-z][\w'\-]*)){0,7}?)\s+(?P<V>" + GEN_VERBS + r")\b\s+(?P<Y>.{3,120})$", re.I)
CALLED_RX = re.compile(r"^(?P<X>.{3,90}?)\s+(?:is|are|was|were)\s+(?:called|known as|referred to as|named)\s+(?P<Y>[^,;:]{2,60})", re.I)
BAD_SUBJ_START = {"for", "in", "on", "at", "by", "with", "if", "when", "as", "because", "although", "while", "after", "before",
                  "here", "there", "you", "we", "i", "figure", "consider", "note", "however", "thus", "so", "then", "also",
                  "some", "many", "most", "all", "each", "every", "one", "two", "several", "both", "either", "neither", "how", "what", "why", "which", "who", "where"}


class Reader:
    """Держит контекст (последняя сущность) между предложениями одной статьи."""
    def __init__(self): self.last = None; self.src = None

    def read(self, s, src=""):
        """→ список фактов [(X, rel, Y)], шаблон (или None). Факты с rel@year — время."""
        if src != self.src: self.src, self.last = src, (src if src and src[:1].isupper() else None)
        s = re.sub(r"\s*\([^()]*\)", "", s).strip()           # «Paris (nicknamed …) is» → «Paris is»
        # «Russia, or the Russian Federation, is» / «Uzbekistan, officially the Republic of Uzbekistan, formerly known as Uzbekia, is» → «Russia is»
        for _ in range(3): s = re.sub(r",\s*(?:or|also known as|also called|formerly|formely|officially|known as|sometimes called)\b[^,]{2,60}(?=,)", "", s)
        s = re.sub(r"^([A-Z][\w\s\-']{0,40}?),\s+(is|are|was|were)\b", r"\1 \2", s)
        pre = PREAMBLE.match(s); body = s[pre.end():] if pre else s
        year = YEAR.search(pre.group(0)) if pre else None
        # тема: заголовок статьи, если это имя (Википедия); для учебника/пользователя — последняя названная сущность
        topic = self.src if (self.src and self.src[:1].isupper()) else self.last
        # 1) притяжательное: «Its chemical symbol is Au» → (тема, chemical symbol, Au).
        #    Только если предыдущее предложение было про тему (self.last == тема) — иначе «Its rings» уйдут
        #    к Сатурну из статьи Water. Родство (father/mother) — не свойство, а связь с человеком: пропускаем.
        m = POSS.match(body)
        if m and topic and (self.last == self.src or self.last == topic or self.src and self.src[:1].isupper()):
            prop = m.group("P").strip()
            if prop.split()[-1] not in ("father", "mother", "parents", "wife", "husband", "son", "daughter", "brother", "sister"):
                Y = m.group("Y").strip()
                facts = [(topic, prop.replace(" ", "_"), Y)]
                # «Its capital and largest city is Tashkent» → ещё и (Tashkent, be, the capital of <тема>) — чтобы ответ знал страну
                if Y[:1].isupper() and len(Y.split()) <= 3: facts.append((Y, "be", f"the {prop} of {topic}"))
                return facts, None
            return [], None
        # 2) множественный субъект: «Tashkent, Samarkand and Bukhara are cities» → факт на каждого
        m = PLURAL.match(body)
        if m:
            names = [n.strip() for n in re.split(r",\s*|\s+and\s+", m.group("L")) if n.strip()]
            rel = lemma(m.group("V")); Y = m.group("Y").strip()
            self.last = names[-1]
            return [(n, rel, Y) for n in names], None
        # 0) «X … is called Y» / «is known as Y» — главный глагол в конце, даже если внутри X есть другой глагол
        #    («The electrode at which oxidation occurs is called the anode» → X = всё до «is called»)
        m = CALLED_RX.match(body)
        if m:
            X, Y = m.group("X").strip(), m.group("Y").strip().rstrip(".")
            if X.lower().split()[0] not in BAD_SUBJ_START and Y: return [(X, "be_called", Y)], None
        m = RX.match(body)
        if not m:
            return self._read_general(body, topic)
        X, V, Y = m.group("X").strip(), m.group("V"), m.group("Y").strip().rstrip(".")
        # «The capital is Astana» / «Its capital is Astana» в статье о стране → (Astana, be, the capital of <тема>) + (тема, capital, Astana)
        mp = re.match(r"^(?:the|its)\s+(capital|currency|largest city|official language|president|population)$", X.lower())
        if mp and topic and V.lower() in ("is", "was") and Y[:1].isupper():
            Yc = re.split(r",\s+(?:and|while|but)\s+|\s+and\s+", Y)[0].strip()
            return [(Yc, "be", f"the {mp.group(1)} of {topic}"), (topic, mp.group(1).replace(" ", "_"), Yc)], None
        if X.lower() in GENERIC:
            X, how = (topic if X.lower() in TOPIC_WORDS and self.src else self.last), "ctx"
        else:
            how = "name" if X[0].isupper() and X.lower().split()[0] != "the" else "np"
            if how == "name": self.last = X
        if not X or not Y: return [], None
        rel = lemma(V); facts = [(X, rel, Y)]
        if year: facts.append((X, rel + "@year", year.group(1)))
        tpl = body[:m.start("X")] + "{X}" + body[m.end("X"):m.start("Y")] + "{Y}."
        return facts, tpl

    # ---- общий читатель (учебник): субъект = именная группа любой длины до первого глагола; клаузы ----
    def _read_general(self, body, topic):
        facts = []
        # делим на клаузы: «; », «: », «, and », «, which », «, while »
        clauses = re.split(r";\s+|:\s+|,\s+(?:and|which|while|but|whereas)\s+", body)
        for cl in clauses:
            cl = cl.strip().rstrip(".")
            if re.search(r"\b(?:do|does|did|is|are|was|were|can|cannot|will)\s+not\b|\bcannot\b|\bn't\b", cl.lower()): continue   # отрицания — отдельная рамка
            m = GEN_RX.match(cl)
            if not m: continue
            X, V, Y = m.group("X").strip(), m.group("V").lower(), m.group("Y").strip()
            xw = X.lower().split()
            if not X or not Y or len(xw) > 8 or xw[0] in BAD_SUBJ_START or len(Y.split()) < 1: continue
            if xw[0] in ("it", "they", "this", "these", "that", "those") and len(xw) == 1:
                if not topic: continue
                X = topic
            elif X[:1].isupper() and xw[0] not in ("the", "a", "an"): self.last = X
            neg = bool(re.match(r"(?:do|does|did|is|are|was|were)\s+not\b", cl[m.start("V"):].lower()))
            if neg: continue                                             # отрицания — отдельная рамка, пока пропускаем
            rel = lemma(V.split()[-1]) if " " in V else lemma(V)
            if V.startswith(("is", "are", "was", "were")) and " " in V: rel = "be_" + V.split()[-1]   # «is called» → be_called
            facts.append((X, rel, Y))
        return facts, None


def read_all(sentences, sources):
    r = Reader(); facts, templates = [], collections.defaultdict(list)
    for si, (s, src) in enumerate(zip(sentences, sources)):
        fs, tpl = r.read(s, src)
        for f in fs: facts.append(f + (si,))
        if tpl and fs: templates[fs[0][1]].append((tpl, si))
    return facts, templates
