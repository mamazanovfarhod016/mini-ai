"""converse.py — СОБЕСЕДНИК, КОТОРЫЙ УЧИТСЯ ИЗ РАЗГОВОРА (без параметров, без градиента, без чужих моделей). 23.10.2026.

Отвергнуто Фарходом: говорилка-трансформер, Qwen/llama. Здесь всё — память + смысл + конструкции + решение уравнения:

  1. СЛУЧАИ РЕЧИ. Каждый обмен «реплика собеседника → ответ» — случай с кодом смысла (meaning.py), скелетом реплики
     (вопрос / о себе / прошлое / будущее / эмоция) и конструкцией ответа с дырками ([NAME], [U:слово], [N:число]).
     Новый вход → ближайшие случаи → конструкция → заполнение словами ИЗ ЭТОГО разговора. Ответ разный, не копия.
  2. ПАМЯТЬ О СОБЕСЕДНИКЕ. «меня зовут X», «мне 16», «у меня 1000 сум», «купил за 456», «мой кот Барсик» → факты профиля
     (ключ → значение, числа — с арифметикой). «как меня зовут / сколько осталось / в каком я классе» → ответ из профиля.
  3. СХЕМЫ РАССУЖДЕНИЯ. Ответ-эталон вида «Думаю так: 1) … 2) … Значит: X» разбирается на шаги; каждый шаг — отношение
     между сущностями реплики (старше/младше на N, в/внутри, стоит/осталось, время). Схема = последовательность
     операций над слотами; на новой задаче с той же формой слоты заполняются из реплики, шаги пересчитываются, ответ
     печатается ШАГАМИ. Это и есть CoT, выученный с одного показа (как few-shot у GPT-3, но для цепочки).
  4. РОСТ ОТ ИСПРАВЛЕНИЙ. «лучше сказать так: …» → новый случай с весом 2, старый ответ на этот вход — вес вниз.
     «нет, шаг 2 неверный: …» → схема помечается, шаг заменяется. Всё — в состоянии (converse.json), читаемо.
  5. ЧЕСТНОЕ «НЕ ЗНАЮ». Нет случая ближе порога и нет факта → так и говорит, спрашивает; ответ собеседника → профиль/факт.

Обучение из диалогов (teach_dialog): реплика собеседника → СВОЙ ответ → эталон как исправление (если далеко по смыслу).
Так корпус проходит через неё как живой разговор, а не как таблица.
"""
import os, re, json, collections, random
import numpy as np

STOP = set("и а но или же ли бы не ни это этот эта эти тот та то те он она оно они его ее их ему ей им в во на с со у к ко о об от до за из по при для без через между как что кто где когда почему зачем да нет ну вот уже еще ещё очень так там тут здесь тоже также мне тебе нам вам мой моя мое мои твой наш ваш свой я ты мы вы меня тебя себя есть был была было были будет значит думаю так".split())
STOP_EN = set("the a an and or but of to in on at for with by from as is are was were be been it its this that these those i you he she we they me my your his her our their do does did have has had not no yes so very just also too".split())
NUM_WORDS = {"ноль": 0, "один": 1, "одна": 1, "два": 2, "две": 2, "три": 3, "четыре": 4, "пять": 5, "шесть": 6, "семь": 7, "восемь": 8, "девять": 9, "десять": 10,
             "одиннадцать": 11, "двенадцать": 12, "тринадцать": 13, "четырнадцать": 14, "пятнадцать": 15, "шестнадцать": 16, "семнадцать": 17, "восемнадцать": 18,
             "девятнадцать": 19, "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50, "сто": 100, "тысяча": 1000, "тысячу": 1000}

def toks(s): return re.findall(r"[а-яёa-z0-9:]+", s.lower().replace("ё", "е"))
def nums(s):
    out = [int(x) for x in re.findall(r"\b\d+\b", s)]
    for w in toks(s):
        if w in NUM_WORDS: out.append(NUM_WORDS[w])
    return out

def skeleton(s):
    t = toks(s); k = set()
    if "?" in s or (t and t[0] in ("что", "кто", "где", "когда", "почему", "зачем", "как", "какой", "какая", "какие", "сколько", "помнишь", "знаешь", "а")): k.add("Q")
    if any(w in ("я", "мне", "меня", "мой", "моя", "мои", "моего", "моей", "у") for w in t): k.add("I")
    if any(w in ("ты", "тебе", "тебя", "твой", "твоя") for w in t): k.add("YOU")
    if any(w.endswith(("ал", "ала", "ил", "ила", "ел", "ела", "ул", "ула", "ался", "илась")) for w in t): k.add("PAST")
    if any(w in ("завтра", "буду", "будет", "скоро", "потом", "хочу", "собираюсь", "планирую") for w in t): k.add("FUT")
    if any(w in ("боюсь", "страшно", "грустно", "плохо", "устал", "устала", "обидно", "злюсь", "переживаю", "волнуюсь", "бесит", "надоело") for w in t): k.add("NEG")
    if any(w in ("рад", "рада", "ура", "круто", "класс", "отлично", "здорово", "наконец", "получилось", "сдал", "сдала") for w in t): k.add("POS")
    if nums(s): k.add("NUM")
    return k

# ---------- профиль собеседника: закрытая таблица ФОРМ (грамматика), значения — из разговора ----------
PROFILE_RX = [
    ("name", re.compile(r"(?:меня зовут|моё имя|мое имя|я)\s+([А-ЯЁ][а-яё]+)\b")),
    ("age", re.compile(r"\bмне\s+(\d{1,2})\b(?!\s*(?:сум|минут|час|лет назад))")),
    ("grade", re.compile(r"\b(?:я\s+)?(?:учусь\s+)?в\s+(\d{1,2}|первом|втором|третьем|четвертом|пятом|шестом|седьмом|восьмом|девятом|десятом|одиннадцатом)\s*(?:-?м\s+)?классе")),
    ("city", re.compile(r"\b(?:я\s+)?(?:живу|живём|живем|из)\s+(?:в\s+)?([А-ЯЁ][а-яё]+(?:е|и)?)\b")),
    ("money", re.compile(r"\bу меня\s+(?:в кошельке\s+|есть\s+|осталось\s+)?(\d+)\s*(?:сум|сумов|тысяч|долларов|\$)?")),
    ("pet", re.compile(r"\b(?:мо(?:й|я|его)\s+)?(?:кот|кошк[ау]|собак[ау]|пёс|пес|хомяк|попугай)[а-я]*\s+(?:зовут\s+)?([А-ЯЁ][а-яё]+)\b")),
    ("likes", re.compile(r"\b(?:я\s+)?(?:люблю|обожаю|нравится)\s+([а-яё]+(?:\s+[а-яё]+)?)")),
]
GRADE_WORDS = {"первом": 1, "втором": 2, "третьем": 3, "четвертом": 4, "пятом": 5, "шестом": 6, "седьмом": 7, "восьмом": 8, "девятом": 9, "десятом": 10, "одиннадцатом": 11}
SPEND_RX = re.compile(r"\b(?:купил[а]?|потратил[а]?|отдал[а]?|заплатил[а]?)\b[^.]*?\b(?:за\s+)?(\d+)\b")
EARN_RX = re.compile(r"\b(?:нашел|нашла|нашёл|дали|получил[а]?|заработал[а]?|подарили)\b[^.]*?\b(\d+)\b")

PROFILE_RX_EN = [
    ("name", re.compile(r"\b(?:my name is|i am|i'm|call me)\s+([A-Z][a-z]+)\b")),
    ("age", re.compile(r"\bi(?:'m| am)\s+(\d{1,2})\s*(?:years? old)?\b")),
    ("grade", re.compile(r"\b(?:in)\s+(?:the\s+)?(\d{1,2})(?:st|nd|rd|th)?\s+(?:grade|form|class)")),
    ("city", re.compile(r"\bi (?:live in|am from|'m from)\s+([A-Z][a-z]+)\b")),
    ("money", re.compile(r"\bi have\s+(\d+)\s*(?:sum|soum|dollars|\$|bucks)?")),
    ("pet", re.compile(r"\bmy\s+(?:cat|dog|hamster|parrot)(?:'s name)?\s+(?:is\s+)?(?:called\s+|named\s+)?([A-Z][a-z]+)\b")),
    ("likes", re.compile(r"\bi (?:like|love|enjoy)\s+([a-z]+(?:\s+[a-z]+)?)")),
]
PROFILE_Q_EN = [
    ("name", re.compile(r"what(?:'s| is) my name|who am i\b|remember my name")),
    ("age", re.compile(r"how old am i|my age")),
    ("grade", re.compile(r"what grade am i|which (?:grade|class) am i")),
    ("city", re.compile(r"where (?:am i from|do i live)|my city")),
    ("money", re.compile(r"how much (?:money )?do i have|how much (?:is|have i got) left")),
    ("pet", re.compile(r"what(?:'s| is) my (?:cat|dog|pet)(?:'s)? name")),
    ("likes", re.compile(r"what do i (?:like|love)")),
]
PROFILE_SAY_EN = {
    "name": ["You're {v}.", "Your name is {v}.", "{v} — of course I remember."],
    "age": ["You're {v}.", "{v}, you told me.", "You said you're {v}."],
    "grade": ["Grade {v}.", "You're in grade {v}, you said."],
    "city": ["You're from {v}.", "You said {v}."],
    "money": ["You have {v}.", "{v} left.", "{v}, if you haven't spent anything since."],
    "pet": ["{v}.", "Your pet's name is {v}."],
    "likes": ["You like {v}.", "You told me you like {v}."],
}
SPEND_RX_EN = re.compile(r"\b(?:bought|spent|paid)\b[^.]*?\b(\d+)\b")
PROFILE_Q = [
    ("name", re.compile(r"как (?:меня|мен[яе]) зовут|кто я\b|моё имя|мое имя|помнишь.{0,12}(?:имя|зовут)")),
    ("age", re.compile(r"сколько мне (?:лет|годиков)|мой возраст|помнишь.{0,10}лет")),
    ("grade", re.compile(r"в каком (?:я )?классе|какой (?:у меня )?класс")),
    ("city", re.compile(r"где я живу|откуда я|мой город")),
    ("money", re.compile(r"сколько (?:у меня )?(?:осталось|денег|сум)|сколько.{0,10}кошельке")),
    ("pet", re.compile(r"как зовут мо(?:его|ю|й) (?:кота|кошку|собаку|пса|хомяка|попугая)|имя (?:моего )?(?:кота|пса)")),
    ("likes", re.compile(r"что я люблю|что мне нравится")),
]
PROFILE_SAY = {
    "name": ["Ты {v}.", "Тебя зовут {v}.", "{v} — конечно, помню."],
    "age": ["Тебе {v}.", "{v} лет, ты говорил.", "Тебе {v} — ты сам сказал."],
    "grade": ["В {v}-м классе.", "Ты в {v} классе, ты говорил.", "{v}-й класс."],
    "city": ["Ты из {v}.", "Ты говорил — {v}.", "{v}, если я правильно запомнил."],
    "money": ["У тебя {v} сум.", "Осталось {v}.", "Сейчас {v} сум, если ничего не тратил."],
    "pet": ["{v}.", "Твоего кота зовут {v}.", "{v}, ты говорил."],
    "likes": ["Ты любишь {v}.", "Ты говорил, что любишь {v}."],
}

# ---------- схемы рассуждения: разбор эталона «Думаю так: 1) … Значит: …» ----------
STEP_RX = re.compile(r"\d\)\s*(.*?)(?=\s+\d\)|\s+Значит:|$)")
FINAL_RX = re.compile(r"Значит:\s*(.*)$")
# СЛОВА-ОТНОШЕНИЯ: форма задачи. Это грамматика рассуждения (закрытая таблица), не знание. Схема переносится только
# на задачу с теми же отношениями и тем же числом чисел/имён — иначе получается «Программистом Египта — Каир».
REL_WORDS = {"старше", "младше", "выше", "ниже", "больше", "меньше", "дороже", "дешевле", "дальше", "ближе", "быстрее", "медленнее", "раньше", "позже",
             "в", "на", "из", "до", "от", "за", "по", "через", "между", "около",
             "столица", "область", "страна", "город", "район", "регион", "материк", "континент",
             "осталось", "потратил", "потратила", "купил", "купила", "стоит", "стоят", "сдача", "сум", "долларов", "рублей",
             "минут", "часов", "час", "часа", "дней", "день", "дня", "недель", "неделя", "лет", "года", "год", "начинается", "заканчивается", "перемена", "урок",
             "делить", "умножить", "плюс", "минус", "каждый", "каждая", "всего", "по", "раз", "страниц", "километров", "км", "метров"}
def shape(s):
    t = toks(s); rel = tuple(sorted({w for w in t if w in REL_WORDS})); names = re.findall(r"\b[А-ЯЁ][а-яё]{2,}\b", s)
    return dict(rel=rel, n_num=len(nums(s)), n_name=len(set(names)), q=("Q" in skeleton(s)))

class Converse:
    RU_NPZ = os.path.expanduser("~/Developer/mini-ai-runs/learning-core/meaning_ru.npz")
    def __init__(self, mind, state_dir=None):
        self.mind = mind; self.M_en = getattr(mind, "meaning", None); self.M_ru = None
        try:
            from mini_ai.meaning import Meaning
            if os.path.exists(self.RU_NPZ): self.M_ru = Meaning(None); self.M_ru.load(self.RU_NPZ)
        except Exception: self.M_ru = None
        self.M = self.M_ru or self.M_en
        self.path = os.path.join(state_dir, "converse.json") if state_dir else None
        self.cases = []            # dict(inp, out, cons, skel(list), name, w, vec(np))
        self.profile = {}; self.history = []; self.schemes = []; self.log = []
        self.stats = collections.Counter(); self._vecs = None; self.rng = random.Random(7)
        self.load()

    # ---------- смысл ----------
    @staticmethod
    def lang(s): return "ru" if re.search(r"[а-яё]", s.lower()) else "en"
    def meaning_for(self, s): return (self.M_ru if self.lang(s) == "ru" else self.M_en) or self.M
    def vec(self, s):
        M = self.meaning_for(s)
        if M is None: return None
        ws = [w for w in toks(s) if w not in STOP and w not in STOP_EN and not w.isdigit()]
        v = M.sent_vec(ws) if ws else None
        return None if v is None else v.astype(np.float32)
    def _stack(self):
        if self._vecs is None or len(self._vecs) != len(self.cases):
            D = 4096
            self._vecs = np.stack([c["vec"] if c.get("vec") is not None else np.zeros(D, np.float32) for c in self.cases]) if self.cases else np.zeros((0, D), np.float32)
        return self._vecs

    # ---------- профиль ----------
    def observe(self, text):
        """факты о собеседнике из его реплики (формы — грамматика; значения — из разговора)"""
        learned = []; self.observe_fact(text)
        if self.lang(text) == "en":
            for key, rx in PROFILE_RX_EN:
                m = rx.search(text)
                if not m: continue
                v = next((g for g in m.groups() if g), None)
                if not v or (key == "name" and v.lower() in STOP_EN): continue
                self.profile[key] = v; learned.append((key, v))
            m = SPEND_RX_EN.search(text)
            if m and str(self.profile.get("money", "")).isdigit(): self.profile["money"] = str(int(self.profile["money"]) - int(m.group(1))); learned.append(("money", self.profile["money"]))
            return learned
        for key, rx in PROFILE_RX:
            m = rx.search(text)
            if not m: continue
            v = m.group(1)
            if key == "grade": v = str(GRADE_WORDS.get(v.lower(), v))
            if key == "name" and (v.lower() in STOP or len(v) < 3): continue
            if key == "city": v = re.sub(r"е$", "", v) if v.endswith(("ане", "ене", "оне", "ине")) else v
            self.profile[key] = v; learned.append((key, v))
        m = SPEND_RX.search(text)
        if m and "money" in self.profile and str(self.profile["money"]).isdigit():
            self.profile["money"] = str(int(self.profile["money"]) - int(m.group(1))); learned.append(("money", self.profile["money"]))
        m = EARN_RX.search(text)
        if m and "money" in self.profile and str(self.profile["money"]).isdigit() and not SPEND_RX.search(text):
            self.profile["money"] = str(int(self.profile["money"]) + int(m.group(1))); learned.append(("money", self.profile["money"]))
        return learned

    # ОБЩАЯ ПАМЯТЬ О СОБЕСЕДНИКЕ (не закрытый список ключей): любое утверждение от первого лица → факт (слова-ключи → фраза)
    FIRST_PERSON = re.compile(r"\b(?:у меня|я|мне|мой|моя|мои|моего|моей|меня|i|i'm|my|me|mine)\b", re.I)
    def observe_fact(self, text):
        if "?" in text or not self.FIRST_PERSON.search(text): return None
        if re.match(r"^(?:как|что|кто|где|когда|сколько|какой|какая|какие|почему|помнишь|what|who|where|when|how|which|why|do|does|did|remember)\b", text.strip().lower()): return None
        ws = [w for w in toks(text) if w not in STOP and w not in STOP_EN and len(w) > 2]
        if len(ws) < 2: return None
        self.facts = getattr(self, "facts", []); self.facts.append(dict(text=text.strip(), words=set(ws), vec=self.vec(text)))
        return ws

    def answer_fact(self, text):
        """«какой у меня пояс?», «кем я работаю?», «что я люблю?», «как зовут моего брата?» → факт собеседника с теми же ключевыми словами"""
        facts = getattr(self, "facts", [])
        if not facts or not self.FIRST_PERSON.search(text): return None
        tl = text.lower()
        if not ("?" in text or re.match(r"^(?:как|что|кто|где|когда|сколько|какой|какая|какие|кем|помнишь|напомни|вспомни|what|who|where|which|how|do|remember|remind)\b", tl.strip())): return None
        qw = {w for w in toks(text) if w not in STOP and w not in STOP_EN and len(w) > 2 and w not in ("помнишь", "напомни", "вспомни", "забыл", "забыла", "скажи", "подскажешь", "зовут", "remember", "remind", "tell", "name", "called")}
        qf = {min(self.mind._forms(w), key=len) if hasattr(self.mind, "_forms") else w[:5] for w in qw}
        best = None
        for f in reversed(facts):                                        # свежее — важнее
            ff = {w[:5] for w in f["words"]}; common = len(qf & ff) + len({w[:5] for w in qw} & ff)
            sim = float(self.vec(text) @ f["vec"]) if f.get("vec") is not None and self.vec(text) is not None else 0.0
            sc = common + sim
            if common >= 1 and (not best or sc > best[0]): best = (sc, f)
        if not best: return None
        t = best[1]["text"]
        if self.lang(t) == "en":
            t = re.sub(r"\bI'm\b|\bI am\b", "you're", t); t = re.sub(r"\bI\b", "you", t); t = re.sub(r"\bmy\b", "your", t, flags=re.I); t = re.sub(r"\bme\b", "you", t); t = re.sub(r"\bmine\b", "yours", t)
            t = t.strip(" .,!"); return self.rng.choice(["{t}.", "{t}, you told me.", "You said: {t}."]).format(t=t[:1].upper() + t[1:])
        # первое лицо → второе: «у меня синий пояс» → «У тебя синий пояс», «я работаю программистом» → «Ты работаешь программистом»
        t = re.sub(r"\bу меня\b", "у тебя", t, flags=re.I); t = re.sub(r"\bмне\b", "тебе", t, flags=re.I); t = re.sub(r"\bменя\b", "тебя", t, flags=re.I)
        t = re.sub(r"\bмо(й|я|и|его|ей|ю)\b", lambda m: "тво" + m.group(1), t, flags=re.I)
        t = re.sub(r"\bя\s+(\w+?)(ю|у)\b", lambda m: "ты " + m.group(1) + ("ешь" if m.group(2) == "ю" else "ешь"), t, flags=re.I)
        t = re.sub(r"\bя\b", "ты", t, flags=re.I)
        t = t.strip(" .,!"); return self.rng.choice(["{t}.", "{t}, ты говорил.", "Ты говорил: {t}."]).format(t=t[:1].upper() + t[1:])

    def answer_profile(self, text):
        tl = text.lower()
        if self.lang(text) == "en":
            for key, rx in PROFILE_Q_EN:
                if rx.search(tl):
                    if key in self.profile: return self.rng.choice(PROFILE_SAY_EN[key]).format(v=self.profile[key]), key
                    return ("You haven't told me your name yet. Tell me and I'll remember." if key == "name" else "You haven't told me that. Tell me and I'll remember."), key
            return None, None
        for key, rx in PROFILE_Q:
            if rx.search(tl):
                if key in self.profile: return self.rng.choice(PROFILE_SAY[key]).format(v=self.profile[key]), key
                return ("Ты мне не говорил, как тебя зовут. Скажи — запомню." if key == "name" else "Этого ты мне не говорил. Расскажи — запомню."), key
        return None, None

    # ---------- конструкции ----------
    def make_cons(self, inp, out, name):
        inw = {w for w in toks(inp) if w not in STOP and len(w) > 3}; innums = nums(inp); cons = []
        for w in re.findall(r"[а-яёА-ЯЁa-zA-Z0-9:]+|[^\w\s]", out):
            lw = w.lower().replace("ё", "е")
            if name and w == name: cons.append("[NAME]")
            elif w.isdigit() and int(w) in innums: cons.append(f"[N:{w}]")
            elif lw in inw: cons.append(f"[U:{lw}]")
            else: cons.append(w)
        return cons

    def fill(self, cons, inp):
        inw = [w for w in toks(inp) if w not in STOP and len(w) > 3 and not w.isdigit()]; innums = nums(inp); out = []
        for t in cons:
            if t == "[NAME]": out.append(self.profile.get("name", ""))
            elif t.startswith("[N:"): out.append(str(innums[0]) if innums else t[3:-1])
            elif t.startswith("[U:"):
                w0 = t[3:-1]; best, bs = None, -1.0
                for w in inw:
                    M = self.meaning_for(inp); s = M.similar(w0, w) if M else None
                    if s is not None and s > bs: best, bs = w, s
                out.append(best or (inw[0] if inw else w0))
            else: out.append(t)
        s = " ".join(x for x in out if x); s = re.sub(r"\s+([,.!?…:;)])", r"\1", s); s = re.sub(r"\(\s+", "(", s); s = re.sub(r"\s+", " ", s).strip()
        return s[:1].upper() + s[1:] if s else s

    # ---------- схемы рассуждения ----------
    @staticmethod
    def _typed_cons(inp, text):
        """конструкция шага с ТИПИЗИРОВАННЫМИ дырками: [NAME:k] — k-е имя задачи, [N:k] — k-е число, остальное — каркас.
        Слово задачи без типа в каркас не попадает как дырка (иначе перенос ломает смысл)."""
        names = []
        for n in re.findall(r"\b[А-ЯЁ][а-яё]{2,}\b", inp):
            if n not in names: names.append(n)
        numv = [str(x) for x in nums(inp)]; out = []
        for w in re.findall(r"[а-яёА-ЯЁa-zA-Z0-9:]+|[^\w\s]", text):
            if w in names: out.append(f"[NAME:{names.index(w)}]")
            elif w.isdigit() and w in numv: out.append(f"[N:{numv.index(w)}]")
            else: out.append(w)
        return out, len(names), len(numv)

    def parse_scheme(self, inp, out):
        if "Думаю так" not in out and not re.search(r"\b1\)", out): return None
        steps = [s.strip(" .") for s in STEP_RX.findall(out)]; fm = FINAL_RX.search(out)
        if len(steps) < 2 or not fm: return None
        tsteps = [self._typed_cons(inp, s)[0] for s in steps]; tfinal, n_name, n_num = self._typed_cons(inp, fm.group(1).strip(" ."))
        # схема без переносимых дырок (все слова — каркас) = конкретный факт, а не способ думать → не схема
        n_slots = sum(1 for c in tsteps + [tfinal] for t in c if t.startswith("["))
        if n_slots == 0: return None
        return dict(inp=inp, steps=tsteps, final=tfinal, shape=shape(inp), skel=sorted(skeleton(inp)), nums=n_num, names=n_name, vec=self.vec(inp), w=1.0)

    @staticmethod
    def _fill_typed(cons, inp):
        names = []
        for n in re.findall(r"\b[А-ЯЁ][а-яё]{2,}\b", inp):
            if n not in names: names.append(n)
        numv = [str(x) for x in nums(inp)]; out = []
        for t in cons:
            if t.startswith("[NAME:"):
                k = int(t[6:-1]); out.append(names[k] if k < len(names) else "")
            elif t.startswith("[N:"):
                k = int(t[3:-1]); out.append(numv[k] if k < len(numv) else "")
            else: out.append(t)
        s = " ".join(x for x in out if x); s = re.sub(r"\s+([,.!?…:;)])", r"\1", s); s = re.sub(r"\(\s+", "(", s); s = re.sub(r"\s+", " ", s).strip()
        return s[:1].upper() + s[1:] if s else s

    def apply_scheme(self, inp):
        """схема переносится ТОЛЬКО на задачу той же формы: те же слова-отношения (≥2 общих и ни одного чужого ключевого),
        столько же чисел и имён; среди подходящих — ближайшая по смыслу. Иначе — None (честнее, чем бред шагами)."""
        if not self.schemes: return None
        sh = shape(inp)
        if not sh["rel"] and sh["n_num"] == 0: return None
        v = self.vec(inp); best = None
        for sc in self.schemes:
            s2 = sc.get("shape")
            if not s2: continue
            if s2["n_num"] != sh["n_num"] or s2["n_name"] != sh["n_name"]: continue
            common = set(s2["rel"]) & set(sh["rel"]); diff = set(s2["rel"]) ^ set(sh["rel"])
            if len(common) < max(1, min(2, len(s2["rel"]))) or len(diff) > len(common): continue
            sim = float(v @ sc["vec"]) if v is not None and sc.get("vec") is not None else 0.0
            score = sim + 0.15 * len(common) - 0.1 * len(diff) + 0.05 * sc["w"]
            if not best or score > best[0]: best = (score, sc, sim)
        if not best or best[2] < 0.2: return None
        sc = best[1]; steps = [self._calc(self._fill_typed(c, inp)) for c in sc["steps"]]; final = self._calc(self._fill_typed(sc["final"], inp))
        if any(not st.strip(" .") for st in steps) or not final.strip(" ."): return None
        return dict(text="Думаю так: " + " ".join(f"{i+1}) {s}." for i, s in enumerate(steps)) + f" Значит: {final}.", steps=steps, final=final, sim=round(best[2], 2))

    @staticmethod
    def arith(s):
        """«сколько будет 9 на 9», «2 в четвёртой степени», «7*8», «120000 делить на 4» — считаем, не ищем похожий случай"""
        t = s.lower().replace("ё", "е"); ns = re.findall(r"\d+", t)
        if len(ns) >= 2 and re.search(r"сколько будет|посчитай|реши|=|\?", t) or re.fullmatch(r"[\d\s+\-*/×х:()]+\??", t.strip()):
            a, b = (int(ns[0]), int(ns[1])) if len(ns) >= 2 else (None, None)
            if a is None: return None
            if re.search(r"\bна\b|умнож|\*|×|х\b", t) and not re.search(r"делить|раздел|/|:", t): return f"{a} на {b} — {a * b}."
            if re.search(r"делить|раздел|/|:(?!\d)", t): return (f"{a} делить на {b} — {a // b}." if b and a % b == 0 else (f"{a} делить на {b} — {round(a / b, 2)}." if b else None))
            if re.search(r"плюс|\+|прибав|сложи", t): return f"{a} плюс {b} — {a + b}."
            if re.search(r"минус|\-|отним|вычт", t): return f"{a} минус {b} — {a - b}."
        m = re.search(r"(\d+)\s+в\s+(квадрате|кубе|второй|третьей|четвертой|пятой|шестой|\d+-?й)\s*(?:степени)?", t)
        if m:
            p = {"квадрате": 2, "кубе": 3, "второй": 2, "третьей": 3, "четвертой": 4, "пятой": 5, "шестой": 6}.get(m.group(2)) or int(re.sub(r"\D", "", m.group(2)) or 0)
            if p: a = int(m.group(1)); return f"{a} в {m.group(2)} степени — {a ** p}."
        return None

    @staticmethod
    def _calc(s):
        """«3 + 2 = 5» внутри шага: пересчитать правую часть по левой (числа подставлены новые)"""
        def rep(m):
            try:
                val = eval(m.group(1).replace("×", "*").replace("х", "*"), {"__builtins__": {}}, {})
                return f"{m.group(1).strip()} = {int(val) if float(val).is_integer() else round(val, 2)}"
            except Exception: return m.group(0)
        return re.sub(r"(\d+(?:\s*[-+*/×х]\s*\d+)+)\s*=\s*\d+", rep, s)

    # ---------- ответ ----------
    FACT_Q_RU = re.compile(r"^(?:какая|какой|какое|какие|что такое|что за|где находится|кто такой|кто такая|сколько (?:жителей|лет|километров) )|\b(?:столица|валюта|население|площадь|находится|расположен|открыл|изобрел|изобрёл|написал|автор|год|году)\b", re.I)
    FACT_Q_EN = re.compile(r"^(?:what is the|what are the|which|who is|who was|where is|when was|when did|how many|what does|what do (?!i\b))", re.I)
    def is_factual(self, inp):
        """фактический вопрос (о мире, не о собеседнике) → ядро с памятью фактов, не речь"""
        if self.FIRST_PERSON.search(inp) and re.search(r"\b(?:я|мне|меня|мой|моя|у меня|my|me|am i|do i)\b", inp.lower()): return False
        return bool((self.FACT_Q_RU.search(inp) if self.lang(inp) == "ru" else self.FACT_Q_EN.search(inp)) and ("?" in inp or self.lang(inp) == "ru"))

    def respond(self, inp, k=5):
        learned = self.observe(inp)
        pa, key = self.answer_profile(inp)
        if pa: self.stats["profile"] += 1; return dict(text=pa, how="память о тебе: " + str(key), confidence=0.9)
        if self.is_factual(inp): return None                                     # факты о мире — ядру
        # собеседник рассказал о себе → подтвердить СВОИМИ словами из того, что записала (а не чужим случаем)
        if learned and not ("?" in inp):
            L = self.lang(inp); parts = []
            for key_, v in learned:
                parts.append({"name": ("Привет, {v}." if L == "ru" else "Hi, {v}."), "age": ("{v} — запомнила." if L == "ru" else "{v} — noted."),
                              "grade": ("{v}-й класс, запомнила." if L == "ru" else "Grade {v}, noted."), "city": ("{v} — запомнила." if L == "ru" else "{v} — noted."),
                              "money": ("Запомнила: у тебя {v} сум." if L == "ru" else "Noted: you have {v}."), "pet": ("{v} — хорошее имя." if L == "ru" else "{v} — nice name."),
                              "likes": ("{v} — запомнила." if L == "ru" else "{v} — noted.")}.get(key_, "{v}").format(v=v))
            self.stats["profile"] += 1
            return dict(text=" ".join(parts), how="память о тебе: записала", confidence=0.85)
        sch = self.apply_scheme(inp)
        if sch: self.stats["scheme"] += 1; return dict(text=sch["text"], how=f"схема рассуждения (сходство {sch['sim']})", confidence=0.7, thoughts=sch["steps"])
        if not (set(toks(inp)) & (REL_WORDS - {"в", "на", "из", "до", "от", "за", "по", "раз"})):     # чистая арифметика — без слов-отношений (старше/осталось/…)
            ar = self.arith(inp)
            if ar: self.stats["arith"] += 1; return dict(text=ar, how="арифметика", confidence=1.0)
        fa = self.answer_fact(inp)
        if fa: self.stats["profile"] += 1; return dict(text=fa, how="память о тебе: факт", confidence=0.85)
        if not self.cases: return None
        v = self.vec(inp); sk = skeleton(inp); V = self._stack(); L = self.lang(inp)
        sims = V @ v if v is not None else np.zeros(len(self.cases), np.float32)
        bonus = np.array([0.08 * len(sk & set(c["skel"])) - 0.05 * len(sk ^ set(c["skel"])) + 0.03 * c["w"] - (0.25 if c.get("used", 0) else 0.0) - (9.0 if c.get("lang", "ru") != L else 0.0) for c in self.cases], np.float32)
        score = sims + bonus; top = np.argsort(-score)[:k]
        same = [i for i in top if self.cases[int(i)].get("lang", "ru") == L]
        if not same or (float(sims[same[0]]) < 0.12 and ("Q" in sk or L == "en")):
            self.stats["unknown"] += 1
            return dict(text=("Не знаю, что на это ответить. Расскажи подробнее — запомню." if L == "ru" else "I don't know what to say to that yet. Tell me more and I'll remember."), how="речь: не знаю", confidence=0.2)
        i = int(self.rng.choice(list(same[:3]))); c = self.cases[i]; c["used"] = c.get("used", 0) + 1
        self.stats["case"] += 1
        return dict(text=self.fill(c["cons"], inp), how=f"речь: случай (сходство {float(sims[i]):.2f})", confidence=round(min(0.85, 0.4 + float(sims[i])), 2), case_id=i)

    # ---------- учёба ----------
    def learn_exchange(self, inp, out, w=1.0):
        name = self.profile.get("name")
        sc = self.parse_scheme(inp, out)
        if sc: sc["w"] = w; self.schemes.append(sc); self.stats["schemes_learned"] += 1
        self.cases.append(dict(inp=inp, out=out, cons=self.make_cons(inp, out, name), skel=sorted(skeleton(inp)), name=name, w=w, vec=self.vec(inp), lang=self.lang(inp))); self._vecs = None
        self.stats["cases_learned"] += 1

    def correct(self, inp, my_out, better, my_case=None):
        """«лучше сказать так: …» — новый случай с весом 2; тот, что дал плохой ответ, теряет вес"""
        if my_case is not None and 0 <= my_case < len(self.cases): self.cases[my_case]["w"] = max(0.1, self.cases[my_case]["w"] * 0.5)
        self.learn_exchange(inp, better, w=2.0); self.cases[-1]["src"] = "teacher"; self.stats["corrections"] += 1
        self.log.append(dict(inp=inp[:80], was=(my_out or "")[:80], better=better[:80]))

    def teach_dialog(self, turns, sim_thr=0.45, verbose=False):
        """диалог как живой разговор: реплика → СВОЙ ответ → эталон как исправление, если ответ далеко от эталона по смыслу"""
        self.profile = {}; self.facts = []; res = collections.Counter()
        for i in range(len(turns) - 1):
            if turns[i]["role"] != "user" or turns[i + 1]["role"] != "bot": continue
            inp, gold = turns[i]["text"], turns[i + 1]["text"]
            mine = self.respond(inp)
            if mine is None: self.learn_exchange(inp, gold); res["no_reply→learn"] += 1; continue
            vm, vg = self.vec(mine["text"]), self.vec(gold)
            sim = float(vm @ vg) if vm is not None and vg is not None else 0.0
            same_steps = ("Думаю так" in gold) == ("Думаю так" in mine["text"])
            if sim >= sim_thr and same_steps and mine["how"].startswith(("память", "схема")): res["accepted"] += 1; self.learn_exchange(inp, gold, w=0.5)
            else: self.correct(inp, mine["text"], gold, mine.get("case_id")); res["corrected"] += 1
            if verbose: print(f"  Ю: {inp}\n  я: {mine['text'][:120]}  [{mine['how']}]\n  эталон: {gold[:120]}  sim={sim:.2f}")
        return res

    # ---------- состояние ----------
    def save(self):
        if not self.path: return
        light = [dict((k, v) for k, v in c.items() if k != "vec") for c in self.cases[-20000:]]
        sch = [dict((k, v) for k, v in s.items() if k != "vec") for s in self.schemes[-2000:]]
        json.dump(dict(cases=light, schemes=sch, log=self.log[-200:], stats=dict(self.stats)), open(self.path, "w"), ensure_ascii=False)
        vs = [c["vec"] for c in self.cases[-20000:]]
        if vs: np.save(self.path.replace(".json", "_vecs.npy"), np.stack([v if v is not None else np.zeros(4096, np.float32) for v in vs]).astype(np.float16))
        sv = [s["vec"] for s in self.schemes[-2000:]]
        if sv: np.save(self.path.replace(".json", "_svecs.npy"), np.stack([v if v is not None else np.zeros(4096, np.float32) for v in sv]).astype(np.float16))

    def load(self):
        if not self.path or not os.path.exists(self.path): return
        try:
            z = json.load(open(self.path, encoding="utf-8")); self.cases = z.get("cases", []); self.schemes = z.get("schemes", []); self.log = z.get("log", []); self.stats = collections.Counter(z.get("stats", {}))
            vp = self.path.replace(".json", "_vecs.npy")
            if os.path.exists(vp):
                V = np.load(vp).astype(np.float32)
                for c, v in zip(self.cases, V): c["vec"] = v
            sp = self.path.replace(".json", "_svecs.npy")
            if os.path.exists(sp):
                S = np.load(sp).astype(np.float32)
                for s, v in zip(self.schemes, S): s["vec"] = v
        except Exception: self.cases, self.schemes = [], []

    def report(self):
        return f"converse: {len(self.cases)} speech cases, {len(self.schemes)} reasoning schemes, {self.stats.get('corrections', 0)} corrections; replies: case {self.stats.get('case', 0)}, scheme {self.stats.get('scheme', 0)}, profile {self.stats.get('profile', 0)}, unknown {self.stats.get('unknown', 0)}"
