"""russian.py — РУССКИЙ ЯЗЫК для ядра, без градиента, по очереди (23.10.2026, «чтоб за сегодня полностью научился русскому»).

Ядро думает по-английски (факты, рамки вопросов, цепочки). Русский — это СЛОЙ ПЕРЕВОДА, который сам растёт:
  1. словарь ru→en: MUSE (≈100k пар, data/ru/dict_ru_en.txt) + ВЫУЧЕННЫЕ пары из параллельных предложений (OPUS-100 en-ru):
     слово-пара считается выученной, если в параллельных парах «ru-слово ↔ en-слово» встречаются вместе чаще, чем порознь
     (PMI по мешкам слов, без градиента; tools/learn_ru.py → data/ru/learned_ru_en.txt);
  2. вопрос по-русски → основа (срез окончаний: «столицей» → «столиц», таблица частых флексий) → словарь → английский вопрос
     того же скелета («какая столица франции» → «what is the capital of france») → ядро как обычно;
  3. ответ: короткая сущность (answer) переводится обратно en→ru по словарю (если есть), текст — шаблоном по-русски;
     факт-источник показываем по-английски, честно;
  4. чтение по-русски (ruwiki, TinyStories): смысл русских слов строится ТЕМ ЖЕ механизмом (meaning.py, контексты ±3) в отдельной
     базе meaning_ru.npz — для similar/neighbors по-русски и для моста по смыслу: ru-слово без словаря → ближайшее по смыслу
     ru-слово, у которого перевод есть.
Вопросительные слова, предлоги, частицы — закрытая таблица (это грамматика, не знание). Всё остальное — из данных.
"""
import os, re, collections

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RU_DIR = os.path.join(ROOT, "data/ru")
RU_BASE_NPZ = os.path.expanduser("~/Developer/mini-ai-runs/learning-core/meaning_ru.npz")

def rtoks(s): return re.findall(r"[а-яёa-z0-9]+", s.lower().replace("ё", "е"))

QWORDS = {"что": "what", "кто": "who", "где": "where", "когда": "when", "почему": "why", "зачем": "why", "как": "how", "какой": "what", "какая": "what", "какое": "what",
          "какие": "what", "каком": "what", "какую": "what", "каким": "what", "сколько": "how many", "который": "which", "которая": "which", "которое": "which", "чем": "what", "чего": "what", "кого": "who", "кому": "who", "кем": "who", "чему": "what"}
STOP_RU = set("и а но или же ли бы не ни это этот эта эти тот та то те он она оно они его ее их ему ей им в во на с со у к ко о об обо от до из за по под над при для про без через между "
              "есть был была было были будет будут быть являлся является являются называется называют еще уже очень самый самая самое самые все весь вся "
              "мы вы я ты мне тебе нам вам мой моя мое мои твой наш ваш свой своя свое свои такой такая такое такие тогда потом здесь там туда сюда так также тоже".split())
STOP_MAP = {"в": "in", "во": "in", "на": "on", "с": "with", "со": "with", "у": "at", "к": "to", "ко": "to", "о": "about", "об": "about", "от": "from", "до": "to", "из": "from", "за": "for", "по": "by",
            "под": "under", "над": "above", "при": "at", "для": "for", "без": "without", "через": "through", "между": "between", "и": "and", "или": "or", "а": "and", "но": "but", "не": "not",
            "есть": "is", "является": "is", "являются": "are", "называется": "is called", "называют": "is called", "был": "was", "была": "was", "было": "was", "были": "were", "будет": "will be"}
SUFFIXES = sorted(set(["иями", "ями", "ами", "ием", "ему", "ому", "ого", "его", "ыми", "ими", "ях", "ах", "ов", "ев", "ей", "ий", "ый", "ой", "ая", "яя", "ое", "ее", "ые", "ие", "ую", "юю",
                       "ом", "ем", "ам", "ям", "ью", "ия", "ии", "а", "я", "ы", "и", "у", "ю", "е", "о", "ь", "й"]), key=len, reverse=True)
def stem(w):
    for s in SUFFIXES:
        if w.endswith(s) and len(w) - len(s) >= 3: return w[: -len(s)]
    return w


class Russian:
    def __init__(self, mind, state_dir=None):
        self.mind = mind; self.state_dir = state_dir
        self.ru2en = collections.defaultdict(collections.Counter)
        self.en2ru = collections.defaultdict(collections.Counter)
        self.stem2ru = collections.defaultdict(set)
        self.stem2en = collections.defaultdict(collections.Counter)      # ВЫУЧЕННОЕ из параллельных текстов: основа → Counter(en) (tools/learn_ru.py)
        self.meaning_ru = None; self.n_pairs = 0; self.n_learned = 0
        self._load_dict(os.path.join(RU_DIR, "dict_ru_en.txt"))
        self._load_stems(os.path.join(RU_DIR, "learned_ru_en.txt"))
        if state_dir: self._load_dict(os.path.join(state_dir, "ru_learned.txt"))
        try:
            from mini_ai.meaning import Meaning
            if os.path.exists(RU_BASE_NPZ):
                self.meaning_ru = Meaning(None); self.meaning_ru.load(RU_BASE_NPZ)
        except Exception: self.meaning_ru = None

    def _load_dict(self, path):
        if not os.path.exists(path): return
        for line in open(path, encoding="utf-8"):
            p = line.strip().split()
            if len(p) < 2: continue
            ru, en = p[0].lower().replace("ё", "е"), p[1].lower(); w = float(p[2]) if len(p) > 2 else 1.0
            if not re.fullmatch(r"[а-я\-]+", ru) or not re.fullmatch(r"[a-z\-]+", en): continue
            self.ru2en[ru][en] += w; self.en2ru[en][ru] += w; self.stem2ru[stem(ru)].add(ru); self.n_pairs += 1

    def _load_stems(self, path):
        if not os.path.exists(path): return
        for line in open(path, encoding="utf-8"):
            p = line.strip().split()
            if len(p) < 2: continue
            self.stem2en[p[0]][p[1]] += float(p[2]) if len(p) > 2 else 1.0; self.n_pairs += 1

    def learn_pair(self, ru, en, w=2.0, persist=True):
        ru, en = ru.lower().replace("ё", "е"), en.lower()
        self.ru2en[ru][en] += w; self.en2ru[en][ru] += w; self.stem2ru[stem(ru)].add(ru); self.n_learned += 1
        if persist and self.state_dir:
            with open(os.path.join(self.state_dir, "ru_learned.txt"), "a", encoding="utf-8") as f: f.write(f"{ru} {en} {w}\n")

    def ru_word(self, w):
        """ru-слово → (en-слово, способ): грамматика → словарь → по основе → по смыслу (meaning_ru) → латиница как есть"""
        if w in STOP_MAP: return STOP_MAP[w], "grammar"
        if w in QWORDS: return QWORDS[w], "grammar"
        if w in self.ru2en: return self.ru2en[w].most_common(1)[0][0], "dict"
        st = stem(w); cands = collections.Counter()
        for r in self.stem2ru.get(st, ()):
            for e, c in self.ru2en[r].items(): cands[e] += c
        if cands: return cands.most_common(1)[0][0], "stem"
        if st in self.stem2en: return self.stem2en[st].most_common(1)[0][0], "learned"          # выучено из параллельных текстов
        st2 = stem(st)
        if st2 != st:
            for r in self.stem2ru.get(st2, ()):
                for e, c in self.ru2en[r].items(): cands[e] += c
            if cands: return cands.most_common(1)[0][0], "stem2"
            if st2 in self.stem2en: return self.stem2en[st2].most_common(1)[0][0], "learned2"
        if self.meaning_ru is not None and self.meaning_ru.vec(w) is not None:
            for nb, s in self.meaning_ru.neighbors(w, k=8, min_n=5):
                if s > 0.45 and nb in self.ru2en: return self.ru2en[nb].most_common(1)[0][0], f"meaning≈{nb}"
        if re.fullmatch(r"[a-z0-9]+", w): return w, "latin"
        return None, None

    def en_word(self, w):
        w = w.lower()
        if w not in self.en2ru: return None
        # начальная форма: среди переводов берём с лучшим весом, при равном — самый короткий (именительный короче косвенных: «кислород» < «кислорода»)
        c = self.en2ru[w]; top = max(c.values())
        return min((r for r, v in c.items() if v >= top * 0.8), key=len)

    def to_en(self, q):
        """русский вопрос → английский + отчёт (слово, перевод, способ) + неизвестные слова"""
        t = rtoks(q); out = []; report = []; unknown = []
        for w in t:
            e, how = self.ru_word(w)
            if e is None: unknown.append(w); continue
            if w in STOP_RU and w not in QWORDS and w not in STOP_MAP: continue
            # число: русское слово в ед. числе (нет флексий мн. ч.) → английское ед. ч. («солнцу» → sun, не suns)
            if how in ("stem", "stem2", "dict") and e.endswith("s") and not e.endswith("ss") and len(e) > 3 and not re.search(r"(?:ы|и|ов|ев|ам|ям|ами|ями|ах|ях)$", w) and (e[:-1] in self.en2ru): e = e[:-1]
            out.append(e); report.append((w, e, how))
        # устойчивые обороты (грамматика): «ближе всего» → closest, «больше всего» → most, «не так ли»
        s = " ".join(out)
        s = re.sub(r"\bclos(?:e|er) (?:total|all|most)\b", "closest", s); s = re.sub(r"\bmore (?:total|all)\b", "most", s)
        s = re.sub(r"\b(\w+) (?:for|to) (?:breath|breathing)\b", r"\1 to breathe", s)
        s = re.sub(r"^(what|which) (?!is|are|do|does|did)(\w+) (\w+)$", r"\1 is the \2 of \3", s)
        s = re.sub(r"^(what|which) (?!is|are|do|does|did)([\w ]+)$", r"\1 is \2", s)
        s = re.sub(r"^where (?!is|are)([\w ]+)$", r"where is \1", s)
        s = re.sub(r"^who (?!is|are|was|were)([\w ]+)$", r"who is \1", s)
        return s.strip() + "?", report, unknown

    def answer_ru(self, en_answer, proper=False):
        """en-ответ → ru: имена собственные (с заглавной в факте) не переводим (Mercury-планета ≠ ртуть); глагольные хвосты (pumps) убираем"""
        if not en_answer: return None
        ws = re.findall(r"[a-zA-Z0-9\-']+", str(en_answer)); out = []
        for i, w in enumerate(ws):
            if proper and w[:1].isupper(): out.append(self._translit(w, names_only=True)); continue        # имя собственное → имя из словаря или транслит
            if i > 0 and len(w) > 4 and w.endswith("s") and not w.endswith("ss") and len(ws) > 1 and w[:-1] in self.en2ru and w not in self.en2ru: continue   # «pumps» после «heart» — глагол
            r = self.en_word(w.lower())
            if r is None and w.lower().endswith("s"): r = self.en_word(w.lower()[:-1])
            out.append(r or w)
        return " ".join(out)

    TRANSLIT = {"a": "а", "b": "б", "c": "к", "d": "д", "e": "е", "f": "ф", "g": "г", "h": "х", "i": "и", "j": "дж", "k": "к", "l": "л", "m": "м", "n": "н", "o": "о", "p": "п", "q": "к", "r": "р", "s": "с", "t": "т", "u": "у", "v": "в", "w": "в", "x": "кс", "y": "и", "z": "з",
                "ch": "ч", "sh": "ш", "th": "т", "ph": "ф", "ck": "к", "ee": "и", "oo": "у", "ay": "ей", "ey": "ей"}
    def _translit(self, w, names_only=False):
        """имя собственное: перевод берём только если он сам — имя (в словаре стоит с той же формой: paris→париж), иначе транслит
        (mercury→ртуть — нарицательное, для планеты не подходит)"""
        r = self.en_word(w.lower())
        if r and (not names_only or self._is_name_pair(w.lower(), r)): return r[:1].upper() + r[1:]
        s = w.lower(); out = ""; i = 0
        while i < len(s):
            if s[i:i + 2] in self.TRANSLIT: out += self.TRANSLIT[s[i:i + 2]]; i += 2
            else: out += self.TRANSLIT.get(s[i], s[i]); i += 1
        return out[:1].upper() + out[1:]

    def _is_name_pair(self, en, ru):
        """пара «имя»: перевод фонетически близок к оригиналу (paris/париж, london/лондон, einstein/эйнштейн), а не нарицательное (mercury/ртуть)"""
        t = self._translit_raw(en); 
        common = sum(1 for a, b in zip(t, ru) if a == b)
        return common >= max(3, int(0.5 * min(len(t), len(ru))))
    def _translit_raw(self, s):
        out = ""; i = 0
        while i < len(s):
            if s[i:i + 2] in self.TRANSLIT: out += self.TRANSLIT[s[i:i + 2]]; i += 2
            else: out += self.TRANSLIT.get(s[i], s[i]); i += 1
        return out

    def say(self, out, en_q, report, unknown):
        ans = out.get("answer")
        if ans:
            # имя собственное? — из данных: слово в памяти встречается С ЗАГЛАВНОЙ и НИКОГДА со строчной (Mercury, Paris — да; Oxygen — нет: «…oxygen…»)
            a0 = str(ans); proper = a0[:1].isupper() and self._proper(a0.split()[0])
            ru = self.answer_ru(ans, proper=proper) or a0
            src = (out.get("facts") or [""])[0]; m = re.match(r"^(.*?) —(.*?)→ (.*)$", src)
            fact_en = (f"{m.group(1)} {m.group(2).replace('_', ' ')} {m.group(3)}".strip() if m else src)[:160]
            txt = f"{ru[:1].upper() + ru[1:]}." + (f" Я поняла вопрос как «{en_q}»." if en_q else "") + (f" Источник (англ.): {fact_en}" if fact_en else "")
            if ru != str(ans).lower(): txt += f" (по-английски: {ans})"
        else:
            txt = f"Я поняла вопрос как «{en_q}», но в памяти не нашла ответа."
        if unknown: txt += " Не знаю слов: " + ", ".join(unknown) + " — скажи «по-русски <слово> значит <english>», и я запомню."
        return txt

    def report(self):
        return self._report()

    def _proper(self, w):
        kb = getattr(self.mind, "kb", None)
        if kb is None or not getattr(kb, "sents", None): return True
        lw = w.lower(); up = low = 0
        for si in kb.sent_by_word.get(lw, [])[:60]:
            s = kb.sents[si][0]
            for m in re.finditer(r"\b" + re.escape(lw) + r"\b", s, re.I):
                if m.group(0)[:1].isupper(): up += 1
                else: low += 1
        return up > 0 and low == 0

    def _report(self):
        return f"русский: словарь {self.n_pairs} пар, выучено в разговоре {self.n_learned}, смысл ru-слов: {len(self.meaning_ru.sem) if self.meaning_ru is not None else 0}"
