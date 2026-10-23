"""analogy.py — ОДНА МАТЕРИЯ вместо списка правил (23.10.2026, после «мы не должны всё как правило добавлять, GPT не так работает»).

Нет типов правил, нет таблицы отношений, нет разборщиков «если…то» / «формула» / «обратное» / «пример».
Любая фраза — факт, объяснение, пример, исправление, вопрос — читается ОДНИМ способом в тройки (кто, слово-связь, кто, число).
  «Диёр старше Сардора на 2»        → (Диёр, старше, Сардор, 2)
  «Риштан в Ферганской области»     → (Риштан, в, Ферганская область)
  «Поезд едет 80 км/ч»              → (Поезд, км/ч, #, 80)        — число тоже «кто»
  «если X старше Y, а Y старше Z, то X старше Z»  → посылки + вывод, X/Y/Z — просто имена
  «смотри: Диёр старше Сардора, Сардор старше Камрона — значит Диёр старше Камрона» → то же самое
Всё, что сказал учитель, хранится как СЛУЧАЙ: (посылки, вывод) с именами, забытыми в переменные по порядку.
Новая задача → ищется случай той же ФОРМЫ (связи между позициями совпадают, слова-связи близки по смыслу —
meaning_ru + сравнительная степень), имена подставляются, вывод переносится. Замыкание — до 4 шагов. Число в выводе —
операция (+ − × ÷), подобранная ИЗ ПРИМЕРА (3 и 2 → 5 ⇒ «+»; 320 и 80 → 4 ⇒ «÷»).
Правило = случай, у которого забыли имена. Обобщение = смысл слов: случай про «старше» применяется к «тяжелее».
Исправление («нет, Илхом, потому что Карим старше Илхома») = ещё один случай + минус тому, которым ответила.
Что здесь всё-таки моя рука (честно): токенизация, падежи имён → основа, «на N» = число связи, вопросительные слова
(кто/где/сколько/на сколько) = дырка в тройке. Это грамматика, как предлоги; знания в ней нет.
"""
import re, json, os, collections, itertools

STOP = set("и а но или же ли бы не ни это этот эта то те он она оно они его ее их ему ей им на с со у к ко о об от до из за по под над при для про без через между есть был была было были будет быть "
           "мы вы я ты мне тебе нам вам мой моя наш ваш свой так также тоже уже еще очень как что кто где когда который чем тем самый самая самое год года лет см кг км сум раз если то значит поэтому "
           "следовательно выходит получается тогда смотри например вот пример пусть будь за до".split())
CONCL = re.compile(r"\s*(?:[—\-]+|,)?\s*\b(?:значит|поэтому|следовательно|выходит|получается|то есть|тогда|то)\b\s*[,:]?\s*", re.I)
NUM = re.compile(r"\d+(?:[.,]\d+)?")
CMP = re.compile(r"(?:ее|ей|ше|же|че|ще|вле)$")
QW = ("на сколько", "сколько", "кто", "что", "где", "какой", "какая", "какое", "за сколько")
VERBS_SKIP = set("бежит бегает едет идет живет находится входит лежит весит стоит прыгает плавает растет ехать идти доедет займет зайдет был была будет есть имею имеет".split())

def _n(s): return float(s.replace(",", "."))

def stem(w):
    w = w.lower().replace("ё", "е")
    for e in ("ами", "ями", "ого", "его", "ому", "ему", "ой", "ей", "ом", "ем", "ую", "юю", "ая", "яя", "ие", "ые", "ах", "ях", "а", "я", "ы", "и", "у", "ю", "е", "о", "й", "ь"):
        if w.endswith(e) and len(w) - len(e) >= 3: return w[: -len(e)]
    return w

def same(a, b):
    """одна сущность? основы равны или одна — префикс другой (чай/чая, самса/самсы, Зухра/Зухры)"""
    if " " in a or " " in b: return " ".join(stem(x) for x in a.split()) == " ".join(stem(x) for x in b.split())
    a, b = stem(a), stem(b)
    if a == b: return True
    if len(a) >= 3 and len(b) >= 3 and (a.startswith(b) or b.startswith(a)) and abs(len(a) - len(b)) <= 2: return True
    a2, b2 = a.rstrip("аяыиуюеой"), b.rstrip("аяыиуюеой")      # короткие слова: чай/чая → ч; требуем длину ≥2 ИЛИ равные остатки при исходной длине ≤4
    return a2 == b2 and (len(a2) >= 2 or (len(a) <= 4 and len(b) <= 4 and a2))

def nominative(w):
    if " " in w or not w[:1].isupper(): return w
    for e, r in (("ы", "а"), ("ки", "ка"), ("ги", "га"), ("хи", "ха"), ("ури", "ура"), ("и", "я"), ("я", "ь"), ("ура", "ур"), ("ара", "ар"), ("ира", "ир"), ("а", ""), ("ом", ""), ("ем", "ь"), ("у", ""), ("ю", "ь"), ("е", "")):
        if w.endswith(e) and len(w) - len(e) >= 3: return w[: -len(e)] + r
    return w

CLASS = ("область", "области", "район", "районе", "город", "городе", "страна", "стране", "азии", "азия")

class Analogy:
    CONCL_RX = CONCL
    def __init__(self, mind=None, state_dir=None, meaning=None):
        self.mind = mind; self.path = os.path.join(state_dir, "analogy.json") if state_dir else None
        self.cases = []; self.bad = collections.Counter(); self.links = {}; self.log = []; self._M = meaning; self._vc = {}
        self.load()

    # ================= ЧТЕНИЕ: одна функция для всего =================
    def clauses(self, text):
        """фраза → [(тройки, is_conclusion)]: по предложениям и запятым перед Именем/классом/«слово связь»; последняя часть после
        значит/поэтому/то — вывод"""
        text = text.strip().rstrip(".")
        m = CONCL.search(text); prem, concl = (text[: m.start()], text[m.end():]) if m and m.start() > 0 else (text, None)
        prem = re.sub(r"^(?:если|когда|смотри|например|вот|пример)\s*[:,]?\s*", "", prem, flags=re.I)
        out = []
        for part in re.split(r"[.;!?]\s*|,\s*(?=[А-ЯЁ][а-яё]+\s)|,\s*(?=(?:у|в|на)\s+[А-ЯЁ][а-яё]+\s)|,\s*(?=(?:" + "|".join(CLASS) + r")\s)|,\s*(?=[а-яё]+\s+[а-яё]+(?:ее|ей|ше|же|че|ле)\s)|,\s*(?=[а-яё]+\s+в\s)|\s+а\s+(?=[А-ЯЁXYZ])", prem):
            if part.strip(): out.append((part.strip(), False))
        if concl: out.append((concl.strip(), True))
        return out

    UNITS = ("км/ч", "м/с", "км", "м", "ч", "час", "часа", "часов", "мин", "минут", "минуты", "сум", "сумов", "кг", "см", "лет", "года", "год", "рублей", "долларов", "штук")

    def triples(self, clause):
        """одна часть → тройки. Принцип один: в части есть сущности (Имя / X,Y,Z / «я» / строчное существительное у края
        сравнительного слова / число) и слова между ними = связь. «на N» после пары = число связи. Число с единицей = (субъект, единица, #, N)."""
        toks = re.findall(r"[А-ЯЁа-яёA-Za-z]+(?:/[а-яё]+)?|\d+(?:[.,]\d+)?|,", clause)
        commas = {i for i, t in enumerate(toks) if t == ","}; toks = [t for t in toks if t != ","]
        # позиции запятых в терминах индексов оставшихся токенов: запятая ПЕРЕД токеном j
        after_comma = set(); k = 0
        for i, t in enumerate(re.findall(r"[А-ЯЁа-яёA-Za-z]+(?:/[а-яё]+)?|\d+(?:[.,]\d+)?|,", clause)):
            if t == ",": after_comma.add(k)
            else: k += 1
        if not toks: return []
        low = [t.lower().replace("ё", "е") for t in toks]
        kind = ["w"] * len(toks)
        for i, t in enumerate(toks):
            if NUM.fullmatch(t): kind[i] = "n"
            elif t in ("X", "Y", "Z", "W"): kind[i] = "e"
            elif low[i] in ("я", "меня", "мне"): kind[i] = "e"
            elif t[0].isupper() and len(t) >= 2 and low[i] not in STOP and low[i] not in VERBS_SKIP and not (i == 0 and low[i] in ("у", "в", "на", "из", "до", "за")): kind[i] = "e"
        # «на N» — число связи, не сущность
        for i in range(1, len(toks)):
            if kind[i] == "n" and low[i - 1] == "на" and not (i + 1 < len(toks) and low[i + 1] in self.UNITS and low[i + 1] in ("км/ч", "м/с")): kind[i] = "d"
        # класс-слово после Имени — часть имени; «область в …» в начале = предыдущее место (решается в solve через stem)
        # строчные сущности у сравнительного слова
        cmp_i = [i for i, t in enumerate(low) if kind[i] == "w" and CMP.search(t) and len(t) >= 5 and t not in STOP]
        for j in cmp_i:
            L = [i for i in range(j) if kind[i] == "w" and low[i] not in STOP and low[i] not in VERBS_SKIP and len(low[i]) >= 3 and not CMP.search(low[i])]
            R = [i for i in range(j + 1, len(toks)) if kind[i] == "w" and low[i] not in STOP and low[i] not in VERBS_SKIP and len(low[i]) >= 3 and not CMP.search(low[i]) and low[i] not in self.UNITS]
            if not any(kind[i] == "e" for i in range(j)) and L: kind[L[-1]] = "e"
            if not any(kind[i] == "e" for i in range(j + 1, len(toks))) and R: kind[R[0]] = "e"
        ents = []   # (start, end, text)
        i = 0
        while i < len(toks):
            if kind[i] == "e":
                name = toks[i] if toks[i] in ("X", "Y", "Z", "W") else ("я" if low[i] in ("я", "меня", "мне") else toks[i])
                if i + 1 < len(toks) and low[i + 1] in CLASS and kind[i + 1] != "e": name += " " + toks[i + 1]; ents.append((i, i + 2, name)); i += 2; continue
                ents.append((i, i + 1, name if name == "я" or name in ("X", "Y", "Z", "W") else name[0].upper() + name[1:])); i += 1; continue
            if kind[i] == "n":
                # единица = следующее слово, если это не сущность/связь: «5 книг», «20 яблок», «3 ручки», «15 км»
                nxt_ok = i + 1 < len(toks) and (i + 1) not in after_comma
                unit = low[i + 1] if nxt_ok and (low[i + 1] in self.UNITS or (kind[i + 1] == "w" and low[i + 1] not in STOP and low[i + 1] not in VERBS_SKIP and not CMP.search(low[i + 1]) and len(low[i + 1]) >= 3 and not re.search(r"(?:ил|ла|ло|ли|ал|ет|ит|ят|ут|ют|ал[аи]?|ел[аи]?|ала|ала)$", low[i + 1]))) else ""
                ents.append((i, i + 2 if unit else i + 1, ("#", float(low[i].replace(",", ".")), unit))); i += 2 if unit else 1; continue
            i += 1
        trip = []
        def after_comma_before(pos):   # индексы токенов, отделённые запятой от pos (слева)
            cut = max([c for c in after_comma if c <= pos], default=0); return set(range(0, cut))
        subj = next((e[2] for e in ents if isinstance(e[2], str)), "я")
        for k, (s1, e1, A) in enumerate(ents):
            if not isinstance(A, str):
                # число: связь = единица или слово перед числом (было/потратил/отдал), субъект = ближайшая сущность слева (или «я»)
                used = {j for (a0, b0, X) in ents if not isinstance(X, str) for j in range(a0, b0)}
                cand = [low[j] for j in range(s1 - 1, -1, -1) if kind[j] == "w" and j not in used and j not in after_comma_before(s1) and (low[j] not in STOP or low[j] in ("есть", "было", "был", "была"))]
                verbs = [w for w in cand if re.search(r"(?:ил|ла|ло|ли|ал|ел|ит|ет|ят|ут|ют|ала|ыл|ыла|дал|ёл|ел)$", w) or w in ("есть", "было", "был", "была", "стоит", "стоят", "всего", "сдача", "осталось")]
                prev_w = (verbs[:1] or cand[:1])
                left = next((e[2] for e in reversed(ents[:k]) if isinstance(e[2], str)), subj)
                pw = prev_w[0] if prev_w else ""
                if pw and kind[[j for j in range(s1 - 1, -1, -1) if kind[j] == "w" and low[j] == pw][0]] == "w" and s1 - 1 >= 0 and kind[s1 - 1] == "e": pw = ""   # «у Лолы 5 книг»: слово перед числом — сущность, не глагол
                rel = " ".join(x for x in (pw, A[2]) if x).strip()
                if rel: trip.append((left, rel, "#", A[1]))
                continue
            nxt = next(((s2, e2, B) for (s2, e2, B) in ents[k + 1:] if isinstance(B, str)), None)
            if not nxt: continue
            s2, e2, B = nxt
            if any(isinstance(e[2], tuple) for e in ents if s1 < e[0] < s2):   # между ними число с единицей — не связь
                continue
            rel = " ".join(low[j] for j in range(e1, s2) if kind[j] == "w" and (low[j] not in STOP or low[j] == "в") and low[j] not in VERBS_SKIP)
            if not rel: continue
            d = next((float(low[j].replace(",", ".")) for j in range(e2, min(e2 + 3, len(toks))) if kind[j] == "d"), None)
            trip.append((A, rel, B, d))
        return trip

    def question(self, text):
        """вопрос → (дырка, слово-связь, сущности-кандидаты): «кто старше: Илхом или Карим» → ("who", "старше", [Илхом, Карим])"""
        q = text.strip().rstrip("?").lower().replace("ё", "е")
        hole = next((w for w in QW if re.search(r"(?<![а-яё])" + w + r"(?![а-яё])", q)), None)
        if re.search(r"\b(?:кто|что)\b", q) and re.search(r"\bили\b", q): hole = "кто"
        if hole in ("какой", "какая", "какое") and not re.search(r"\bсам", q): hole = "сколько"        # «какая сдача?» = числовой вопрос про «сдача»
        if hole in ("кто", "что") and re.search(r"\bсам", q): hole = "кто"
        if not hole: return None
        toks = re.findall(r"[а-яё]+", q)
        cands = [t for t in re.findall(r"[А-ЯЁ][а-яё]+", text) if t.lower() not in ("кто", "что", "где", "сколько", "какой", "какая", "за", "на")]
        mo = re.search(r"([а-яё]+)\s+или\s+([а-яё]+)", q)
        if mo and len(cands) < 2: cands = [mo.group(1).capitalize(), mo.group(2).capitalize()]
        mg = re.search(r"где\s+(?:находится\s+)?([а-яё]+)", q)
        if mg and not cands: cands = [mg.group(1).capitalize()]
        rel = next((t for t in toks if t not in STOP and t not in hole.split() and (CMP.search(t) or re.search(r"(?:ший|шая|кий|кая|ый|ая|ой)$", t)) and len(t) >= 5), None)
        superl = bool(re.search(r"\bсам(?:ый|ая|ое)\b", q))
        return dict(hole=hole, rel=rel, cands=cands, superl=superl, words=[t for t in toks if t not in STOP], text=q)

    # ================= СМЫСЛ связи =================
    def _vec(self, w):
        if w in self._vc: return self._vc[w]
        v = None
        if self._M is not None:
            try:
                import numpy as np
                x = self._M.vec(w.replace("ё", "е").split()[-1])
                v = None if x is None else x / (float(np.linalg.norm(x)) + 1e-9)
            except Exception: v = None
        self._vc[w] = v; return v

    def rel_sim(self, a, b):
        """близость двух слов-связей: тождество 1; иначе смысл (meaning_ru) + оба в сравнительной степени; штраф из исправлений"""
        if a == b: return 1.0
        if stem(a) == stem(b): return 0.95
        wa, wb = a.split(), b.split()
        gen = lambda w: len(w) == 1 and w[0] not in self.UNITS and not CMP.search(w[0]) and not re.search(r"(?:ил|ла|ло|ли|ал|ел|ит|ет|ят|ут|ют|ыл|дал|ёл|ть|ось)$", w[0]) and w[0] not in ("было", "есть", "всего", "сдача", "стоит")
        if gen(wa) and gen(wb): return 0.6        # «книг» vs «ручки»: оба — просто предметы, которые считают; форма «у кого N штук» одна
        if len(wa) > 1 or len(wb) > 1:
            ua, ub = (wa[-1] if wa[-1] in self.UNITS else ""), (wb[-1] if wb[-1] in self.UNITS else "")
            if ua and ub and ua != ub: return 0.0
            ma, mb = (wa[0] if wa[0] not in self.UNITS else ""), (wb[0] if wb[0] not in self.UNITS else "")
            if ua and ua == ub: return 0.9 if (not ma or not mb or ma == mb or self.rel_sim(ma, mb) > 0.2) else 0.5
            if ma and mb: return self.rel_sim(ma, mb)
            return 0.0
        key = tuple(sorted((a, b)))
        if self.links.get(key): return 0.9
        if self.links.get(tuple(sorted((stem(a), stem(b))))): return 0.9
        if self.bad.get(key, 0) >= 1: return 0.0
        s = 0.0
        va, vb = self._vec(a), self._vec(b)
        if va is not None and vb is not None: s = float(va @ vb)
        both_cmp = bool(CMP.search(a.split()[-1]) and CMP.search(b.split()[-1]))
        return max(0.0, 0.6 * s + (0.4 if both_cmp else 0.0)) if (both_cmp or s > 0) else 0.0

    # ================= СЛУЧАИ =================
    def tell(self, text):
        """учитель говорит что угодно: пример, объяснение, факт с выводом → случай. Возвращает пересказ."""
        ml = re.match(r"^([а-яё]+)\s*[—\-:]?\s*(?:это\s+)?(?:то же,? что|значит|означает|как|равно|=)\s+([а-яё]+)$", text.strip().rstrip(".").lower().replace("ё", "е"))
        if ml:   # связь двух СЛОВ (не сущностей): учитель уточняет смысл — это правит rel_sim, как исправление, только в плюс
            a, b = ml.group(1), ml.group(2); self.links[tuple(sorted((a, b)))] = 1.0; self._vc.clear(); self.save()
            return None, f"Поняла: «{a}» и «{b}» — одно и то же."
        parts = self.clauses(text)
        prem = [t for c, is_c in parts if not is_c for t in self.triples(c)]
        concl = [t for c, is_c in parts if is_c for t in self.triples(c)]
        if not prem: return None, "Не вижу здесь связей вида «кто — как — с кем»."
        if not concl:
            # нет вывода — это просто факт(ы), запоминаем как факты текущего разговора
            self.log.append(dict(kind="fact", t=text[:100])); return None, "Запомнила: " + "; ".join(self._say(t) for t in prem) + "."
        names = {}; var = itertools.cycle("XYZWVU"); seen = set()
        def v(x, create=True):
            if x == "#" or x == "я": return x
            k = next((kk for kk in names if same(kk, x)), None)
            if k is None:
                if not create: return None
                nm = next(var)
                while nm in seen: nm = next(var)
                seen.add(nm); names[x] = nm; return nm
            return names[k]
        P = [(v(a), r, v(b), n) for a, r, b, n in prem]
        C = [(v(a, False), r, v(b, False), n) for a, r, b, n in concl]
        if any(x is None for t in C for x in (t[0], t[2])): return None, "Не поняла: в выводе есть кто-то, кого не было в условии."
        # число в выводе — какая операция даёт его из чисел посылок (по порядку)?
        op = None; nums = [n for *_, n in P if n is not None]; cn = C[0][3]
        if cn is not None and len(nums) == 1 and abs(nums[0] - cn) < 1e-9: op = ("=", 0, 0)
        OPS = {"+": lambda x, y: x + y, "-": lambda x, y: x - y, "*": lambda x, y: x * y, "/": lambda x, y: (x / y if y else None)}
        close = lambda v: v is not None and abs(v - cn) < 1e-6 + 1e-3 * abs(cn)
        if op is None and cn is not None and len(nums) == 3:     # сначала формула из ВСЕХ чисел — иначе 300+200=500 совпадёт случайно
            for (n1, f), (n2, g) in itertools.product(OPS.items(), repeat=2):
                for i, j, k in itertools.permutations(range(3), 3):
                    v = f(nums[i], nums[j])
                    if v is not None and close(g(v, nums[k])): op = (n1, i, j, n2, k); break
                if op: break
        if op is None and cn is not None and len(nums) >= 2:
            for n1, f in OPS.items():
                for i, j in itertools.permutations(range(len(nums)), 2):
                    if close(f(nums[i], nums[j])): op = (n1, i, j); break
                if op: break
        if op is None and cn is not None and len(nums) >= 3:     # три числа: ((a ∘ b) ∘ c) — «было 1000, потратил 300 и 200 → 500»
            for (n1, f), (n2, g) in itertools.product(OPS.items(), repeat=2):
                for i, j, k in itertools.permutations(range(len(nums)), 3):
                    v = f(nums[i], nums[j])
                    if v is not None and close(g(v, nums[k])): op = (n1, i, j, n2, k); break
                if op: break
        case = dict(prem=P, concl=C[0], op=op, text=text[:160], used=0, ok=0)
        if any(c["prem"] == P and c["concl"][:3] == C[0][:3] for c in self.cases): return case, "Это я уже знаю: " + self.retell(case)
        self.cases.append(case); self.save(); return case, self.retell(case)

    def retell(self, c):
        say = lambda t: f"{t[0]} {t[1]} {t[2]}" + (f" на {t[3]:g}" if t[3] is not None and t[2] != "#" else "") if t[2] != "#" else f"{t[0]}: {t[1]} = {t[3]:g}"
        s = "Поняла, имена не важны: если " + " и ".join(say(t) for t in c["prem"]) + ", то " + say(c["concl"])
        if c["op"]: s += f" (число: {c['op'][0]}{(' затем ' + c['op'][3]) if len(c['op']) == 5 else ''} от чисел посылок)"
        return s + "."

    def _say(self, t): return f"{t[0]} {t[1]} {t[2]}" + (f" на {t[3]:g}" if t[3] is not None and t[2] != "#" else "") if t[2] != "#" else f"{t[0]} {t[1]} {t[3]:g}"

    # ================= ПРИМЕНЕНИЕ: перенос случая по форме =================
    def _match(self, case, known, thr=0.35):
        """все подстановки переменных случая в известные тройки; связи — по смыслу (≥ thr); возвращает (binding, score, numbers)"""
        res = [({}, 1.0, [], [])]
        # вывод вводит НОВОЕ слово-связь (младше → старше): перенос по смыслу невозможен — не знаем «обратное» к тяжелее; нужны те же слова
        strict = case["concl"][1] not in {p[1] for p in case["prem"]} and case["concl"][2] != "#"
        for A, r, B, n in case["prem"]:
            nxt = []
            for bnd, sc, nums, rels in res:
                for ki, (a, r2, b, n2) in enumerate(known):
                    if ki in bnd.get("@used", ()): continue      # одна тройка задачи — одна посылка случая
                    s = self.rel_sim(r, r2)
                    if s < thr: continue
                    if A == "#" or B == "#":
                        if not (B == "#" and b == "#"): continue
                    elif b == "#": continue
                    if strict and s < 0.95: continue
                    b2 = dict(bnd); ok = True
                    if b2.get("@" + r, r2) != r2: continue
                    b2["@" + r] = r2; b2["@used"] = bnd.get("@used", ()) + (ki,)
                    for var, val in ((A, a), (B, b)):
                        if var == "#" or B == "#": continue       # у числовых троек субъект не связываем (поезд/велик — не важно кто)
                        if var in b2 and b2[var] != val: ok = False; break
                        b2[var] = val
                    vals = [v for k, v in b2.items() if k != "#" and not k.startswith("@")]
                    if ok and len(set(vals)) == len(vals): nxt.append((b2, sc * s, nums + [n2], rels + [r2]))
            res = nxt
            if not res: return []
        return res

    def closure(self, facts, depth=4):
        known = list(dict.fromkeys(facts)); derived = []
        for _ in range(depth):
            new = []
            # порядок: доверие учителя (ok) → случай с числом (op) раньше случая без числа → больше посылок; «использовался» не штраф
            for c in sorted(self.cases, key=lambda c: (-c["ok"], 0 if c["op"] else 1, -len(c["prem"]))):
                A, r, B, cn = c["concl"]
                for bnd, sc, nums, rels in self._match(c, known):
                    a, b = bnd.get(A, A), bnd.get(B, B)
                    # связь вывода: то же слово, если оно есть в посылках случая; иначе ближайшее по смыслу из задачи
                    rel = r
                    if r in {p[1] for p in c["prem"]}:
                        # связь вывода = связь посылки → в новой задаче это та связь, которая встала на место этой посылки
                        k = next(i for i, p in enumerate(c["prem"]) if p[1] == r)
                        rel = rels[k] if k < len(rels) else r
                    n = None
                    idx = ([c["op"][1], c["op"][2]] + ([c["op"][4]] if len(c["op"]) == 5 else [])) if c["op"] else []
                    if c["op"] and all(i < len(nums) and nums[i] is not None for i in idx):
                        F = {"+": lambda x, y: x + y, "-": lambda x, y: x - y, "*": lambda x, y: x * y, "/": lambda x, y: (x / y if y else None), "=": lambda x, y: x}
                        o = c["op"]; n = F[o[0]](nums[o[1]], nums[o[2]])
                        if len(o) == 5 and n is not None: n = F[o[3]](n, nums[o[4]])
                    t = (a, rel, b, n)
                    dup = next((k for k in known + new if k[:3] == t[:3] and (k[2] != "#" or k[3] == t[3])), None)
                    if dup is not None and dup[3] is None and n is not None:      # было без числа — заменяем на вывод с числом
                        if dup in known: known[known.index(dup)] = t
                        else: new[new.index(dup)] = t
                        derived.append((t, c, sc, [tuple(sorted((p[1], r2))) for p, r2 in zip(c["prem"], rels) if p[1] != r2])); continue
                    if dup is None:
                        pairs = [tuple(sorted((p[1], r2))) for p, r2 in zip(c["prem"], rels) if p[1] != r2]   # переносы по смыслу, не по слову
                        new.append(t); derived.append((t, c, sc, pairs))
            if not new: break
            known += new
        return known, derived

    def solve(self, text):
        """задача (факты + вопрос) → ответ из замыкания; None если не нашла."""
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        qs = [p for p in parts if p.endswith("?")] or ([parts[-1]] if parts else [])
        body = " ".join(p for p in parts if p not in qs)
        facts = [t for c, _ in self.clauses(body) for t in self.triples(c)] if body else []
        # падежи → именительный, одна сущность
        groups = []   # [форма, была_подлежащим]
        for a, r, b, n in facts:
            for x, is_s in ((a, True), (b, False)):
                if x in ("#", "я"): continue
                g = next((g for g in groups if same(g[0], x)), None)
                if g is None: groups.append([x, is_s])
                elif is_s and not g[1]: g[0] = x; g[1] = True
        def cn(x):
            if x in ("#", "я"): return x
            g = next((g for g in groups if same(g[0], x)), None)
            return (g[0] if g[1] else nominative(g[0])) if g else x
        facts = [(cn(a), r, cn(b), n) for a, r, b, n in facts]
        q = self.question(qs[0]) if qs else None
        if not facts or not q: return None
        known, derived = self.closure(facts)
        if any((t[1], t[2], t[0]) == (u[1], u[0], u[2]) for t in known for u in known if t[2] != "#" and t is not u):   # A r B и B r A — противоречие
            self._contradiction = [t for t in known if any((t[1], t[2], t[0]) == (u[1], u[0], u[2]) for u in known if u is not t)][:2]; return None
        ents = [x for t in known for x in (t[0], t[2]) if x not in ("#", "я")]; ents = list(dict.fromkeys(ents))
        name = lambda w: next((e for e in ents if same(e, w)), w)
        ans = None; why = []
        rel_words = list(dict.fromkeys(t[1] for t in known if t[2] != "#"))
        def rel_of(word):
            if not word: return None
            if word in rel_words: return word
            for r in rel_words:
                if stem(r)[:4] == stem(word)[:4]: return r
            best = max(rel_words, key=lambda r: self.rel_sim(word, r), default=None)
            if best and self.rel_sim(word, best) >= 0.35: return best
            # «старший»/«высокая»/«тяжёлый» → основа
            base = re.sub(r"(?:ший|шая|шее|кий|кая|кое|ый|ая|ое|ой|ий|ие)$", "", word)
            for r in rel_words:
                if stem(r)[:4] == stem(word)[:4] or CMP.sub("", r)[:4] == base[:4]: return r
            lk = self.links.get(word) or next((v for (a_, b_), v in [((k[0], k[1]), k) for k in self.links] if a_ == word or b_ == word), None)
            return None
        def rel_any(word):
            """связь из вопроса, которой нет в фактах, но есть случай-обратное: «кто самый сильный» при фактах «слабее» →
            закрытие уже вывело «сильнее»-тройки, если был пример; иначе None (честно)"""
            return rel_of(word)
        if q["hole"] == "на сколько":
            r = rel_of(q["rel"])
            ws = [w for w in q["words"] if w != q["rel"] and w not in ("на", "сколько")]
            cs = [name(c) for c in q["cands"]] if len(q["cands"]) >= 2 else [name(w) for w in ws if name(w) != w or any(same(e, w) for e in ents)]
            if len(cs) >= 2:
                a, b = cs[0], cs[-1]
                for t in known:
                    if r and t[1] == r and t[0] == a and t[2] == b and t[3] is not None: ans = f"на {t[3]:g}"; why = [t]
        elif q["hole"] in ("кто", "что") and "или" in q["text"]:
            r = rel_of(q["rel"]); cs = [name(c) for c in q["cands"]] or [name(w) for w in q["words"] if w in {stem(e) for e in ents} or name(w) != w][:2]
            if r and len(cs) >= 2:
                for a, b in ((cs[0], cs[1]), (cs[1], cs[0])):
                    t = next((t for t in known if t[1] == r and t[0] == a and t[2] == b), None)
                    if t: ans = nominative(a) if not any(k[0] == a for k in facts) else a; why = [t]; break
                if ans and "на сколько" in q["text"] and why[0][3] is not None: ans += f", на {why[0][3]:g}"
        elif q["hole"] in ("кто", "что") and q["superl"]:
            r = rel_of(q["rel"])
            if r:
                tops = [p for p in ents if all(any(t[1] == r and t[0] == p and t[2] == o for t in known) for o in ents if o != p)]
                if tops: ans = tops[0]; why = [t for t in known if t[1] == r and t[0] == ans]
        elif q["hole"] == "где":
            x = name(q["cands"][0]) if q["cands"] else None
            chain = [t for t in known if t[0] == x and t[2] != "#"]
            if chain:
                conts = [t[2] for t in chain]
                outer = [b for b in conts if not any(t[0] == b and t[2] in conts for t in known)]
                ans = outer[0] if outer else conts[-1]; why = chain
        elif q["hole"] in ("сколько", "за сколько"):
            # числовой вопрос: случай с числовым выводом, чья форма (единицы/слова связей) совпадает с задачей
            nd = [(t, c, sc) for (t, c, sc, _) in derived if t[2] == "#" and t[3] is not None]
            def qfit(t):
                ws = t[1].split(); return max([1.0 if w in q["words"] else max([self.rel_sim(w, x) for x in q["words"]] + [0]) for w in ws] + [0])
            if nd:
                t, c, sc = max(nd, key=lambda d: (round(qfit(d[0]), 1), len(d[1]["prem"]), d[1]["ok"], d[2])); ans = f"{t[3]:g}"; why = [t]
        if ans is None: return None
        for t, c, sc, _ in derived: c["used"] += 1
        steps = [self._say(t) for t in facts] + [f"по случаю «{c['text'][:60]}…»: {self._say(t)}" for t, c, sc, _ in derived[:5]]
        return dict(answer=ans, steps=steps, derived=derived, why=why, confidence=0.8 if derived else 0.6)

    # ================= ИСПРАВЛЕНИЕ =================
    def correct(self, task, wrong, right_text):
        """Исправление учителя — тот же механизм, не отдельный:
        «нет, Илхом, потому что Карим старше Илхома»  → посылки задачи + вывод учителя = ещё один случай;
                                                       случаи, по которым пришёл неверный ответ, получают минус (ниже в очереди);
        «нет, старше — не про рост» / «старше ≠ выше»  → связь двух слов получает штраф: rel_sim → 0, перенос по ней больше не делается.
        Возвращает, что поняла. Один и тот же тип ошибки должен уходить с ОДНОГО исправления — это и меряем (exp70)."""
        rt = right_text.strip().rstrip(".")
        # (а) «X — не про Y» / «X не то же, что Y» / «X ≠ Y» → штраф паре связей
        mb = re.match(r"^(?:нет,?\s*)?([а-яё]+)\s*(?:—|-|:)?\s*(?:не\s+(?:про|то же,? что|как|значит|означает)|≠|не равно)\s+([а-яё]+)$", rt.lower().replace("ё", "е"))
        if mb:
            a, b = mb.group(1), mb.group(2); self.bad[tuple(sorted((a, b)))] += 1; self._vc.clear()
            # также все связи с той же основой (высокая/выше), чтобы штраф не обходился формой слова
            self.log.append(dict(kind="bad_link", a=a, b=b)); self.save(); return f"Поняла: «{a}» и «{b}» — разное, больше не смешиваю."
        # (б) минус случаям, которые привели к неверному выводу — только тем, чья связь вывода совпала с ответом
        if wrong and wrong.get("derived"):
            why = wrong.get("why") or []
            for t, c, sc, pairs in wrong["derived"]:
                if not why or any(t[:3] == w[:3] for w in why): c["ok"] -= 1
            self._vc.clear()
        # (в) «потому что …» → случай: посылки задачи, вывод учителя
        m = re.search(r"потому что\s+(.+)$", rt, re.I)
        if m:
            expl = m.group(1).strip()
            if CONCL.search(expl):                      # учитель дал полный пример (посылки — значит вывод) → как есть
                case, say = self.tell(expl)
            else:                                       # дал только вывод → посылки = факты задачи
                parts = re.split(r"(?<=[.!?])\s+", task.strip()); body = " ".join(p.rstrip(".") for p in parts if not p.endswith("?"))
                case, say = self.tell(body + " — значит " + expl)
            self.save(); return say
        self.save(); return "Поняла."

    def say_steps(self, res): return "Думаю так: " + " ".join(f"{i+1}) {s}." for i, s in enumerate(res["steps"])) + f" Значит: {res['answer']}."
    def report(self): return f"analogy: {len(self.cases)} случаев, связей слов {len(self.links)}, штрафов {len(self.bad)}"
    def save(self):
        if self.path: json.dump(dict(cases=self.cases, bad={"|".join(k): v for k, v in self.bad.items()}, links={"|".join(k): v for k, v in self.links.items()}, log=self.log[-100:]), open(self.path, "w"), ensure_ascii=False, indent=1)
    def load(self):
        if not self.path or not os.path.exists(self.path): return
        try:
            z = json.load(open(self.path, encoding="utf-8")); self.cases = z.get("cases", []); self.bad = collections.Counter({tuple(k.split("|")): v for k, v in z.get("bad", {}).items()}); self.links = {tuple(k.split("|")): v for k, v in z.get("links", {}).items()}; self.log = z.get("log", [])
            for c in self.cases: c["prem"] = [tuple(p) for p in c["prem"]]; c["concl"] = tuple(c["concl"]); c["op"] = tuple(c["op"]) if c["op"] else None
        except Exception: self.cases = []


# ======================= РЕЧЬ ТОЙ ЖЕ МАТЕРИЕЙ (23.10, его выбор «2») =======================
# Реплика → ответ = тоже случай от учителя: (форма входа → форма ответа). Имена/числа → слоты. Новая реплика → ближайший
# случай ПО СМЫСЛУ (средний вектор слов meaning_ru + общие слова), слоты заполняются из новой реплики. Ниже порога — честно:
# «не знаю, как ответить; скажи: на это отвечай …». Никаких случаев из датасетов — только то, чему научил учитель (Фарход или я).
SPEECH_STOP = set("и а но или же ли бы не ни это этот эта то те он она оно они его ее их ему ей им в во на с со у к ко о об от до из за по под над при для про без через есть был была было были будет быть ты вы мне тебе нам вам мой моя твой твоя наш ваш так тоже уже еще очень как что кто где когда который чем тем".split())

class Speech:
    def __init__(self, state_dir=None, meaning=None):
        self.path = os.path.join(state_dir, "speech.json") if state_dir else None
        self.cases = []; self.links = {}; self._M = meaning; self._vc = {}; self.log = []; self.load()

    def link(self, a, b):
        """учитель: «благодарю — то же, что спасибо» → одно слово для сходства"""
        a, b = a.lower().replace("ё", "е"), b.lower().replace("ё", "е"); root = self.links.get(b, b); self.links[a] = root; self.save()
        return f"Поняла: «{a}» — как «{b}»."

    # ---- форма реплики: слова + слоты ----
    @staticmethod
    def slots(text):
        names = [n for n in re.findall(r"\b[А-ЯЁ][а-яё]{2,}\b", text) if not text.startswith(n)]   # Имя не в начале фразы
        nums = NUM.findall(text)
        return names, nums
    def words(self, text):
        all_ = [w for w in re.findall(r"[а-яёa-z]+", text.lower().replace("ё", "е")) if len(w) > 1]
        ws = [w for w in all_ if w not in SPEECH_STOP]
        ws = ws if len(ws) >= 2 else all_          # короткая реплика («ну как ты») — форма и есть смысл, служебные слова оставляем
        return [self.links.get(w, w) for w in ws]   # слова, которые учитель связал («глупая = тупая»), считаем одним
    def vec(self, text):
        if self._M is None: return None
        import numpy as np
        ws = self.words(text); vs = []
        for w in ws:
            if w not in self._vc:
                x = self._M.vec(w); self._vc[w] = None if x is None else x / (float(np.linalg.norm(x)) + 1e-9)
            if self._vc[w] is not None: vs.append(self._vc[w])
        if not vs: return None
        v = np.mean(vs, 0); return v / (float(np.linalg.norm(v)) + 1e-9)
    def sim(self, a, b):
        """смысл (вектор) 0.7 + общие значимые слова 0.3; без базы — только слова"""
        wa, wb = set(self.words(a)), set(self.words(b)); j = len(wa & wb) / max(1, len(wa | wb))
        va, vb = self.vec(a), self.vec(b)
        if va is None or vb is None: return j
        return 0.7 * max(0.0, float(va @ vb)) + 0.3 * j

    # ---- учитель ----
    def teach(self, inp, out, w=1.0):
        names, nums = self.slots(inp); o = out
        for n in names: o = re.sub(r"\b" + re.escape(n) + r"\b", "[ИМЯ]", o)
        for n in nums: o = o.replace(n, "[ЧИСЛО]", 1) if n in o else o
        if any(c["inp"] == inp for c in self.cases):
            for c in self.cases:
                if c["inp"] == inp: c["out"] = o; c["w"] = max(c["w"], w)
        else: self.cases.append(dict(inp=inp, out=o, w=w, used=0))
        self.save(); return f"Поняла: на «{inp}» — «{out}». И на похожее по смыслу тоже."

    def parse_teach(self, text):
        """«на „как ты сам“ отвечай „нормально, учусь“» / «когда говорят X, отвечай Y» / «X → Y»"""
        t = text.strip().rstrip(".")
        m = re.match(r"^(?:на|когда (?:говорят|спрашивают|пишут))\s*[«\"„']?(.+?)[»\"“']?\s*[,]?\s*(?:отвечай|говори|скажи|ответ)\s*[:\-—]?\s*[«\"„']?(.+?)[»\"“']?$", t, re.I)
        if not m: m = re.match(r"^[«\"„']?(.+?)[»\"“']?\s*(?:→|->|=>)\s*[«\"„']?(.+?)[»\"“']?$", t)
        return (m.group(1).strip(), m.group(2).strip()) if m else None

    # ---- ответ ----
    def reply(self, inp, thr=0.55):
        if not self.cases: return None
        scored = sorted(((self.sim(inp, c["inp"]) * (0.8 + 0.2 * min(c["w"], 2)), c) for c in self.cases), key=lambda x: -x[0])
        s, c = scored[0]
        wa, wb = set(self.words(inp)), set(self.words(c["inp"]))
        if s < thr or (not (wa & wb) and s < 0.8): return dict(text=None, sim=round(s, 2), near=c["inp"], how=f"речь: нет случая (ближайшее {s:.2f})")
        names, nums = self.slots(inp); out = c["out"]
        out = out.replace("[ИМЯ]", names[0] if names else "ты").replace("[ЧИСЛО]", nums[0] if nums else "")
        c["used"] += 1
        return dict(text=out, sim=round(s, 2), near=c["inp"], how=f"речь: случай учителя (сходство {s:.2f})", case=c)

    def correct(self, inp, better, bad_case=None):
        if bad_case is not None:
            bad_case["w"] = max(0.1, bad_case["w"] * 0.5)
            self.teach(bad_case["inp"], better, w=2.0)   # исправленный ответ ставим на ТУ форму, что сработала, иначе старый случай побеждает на её перефразах
        return self.teach(inp, better, w=2.0)

    def report(self): return f"speech: {len(self.cases)} случаев от учителя"
    def save(self):
        if self.path: json.dump(dict(cases=self.cases, links=self.links), open(self.path, "w"), ensure_ascii=False, indent=1)
    def load(self):
        if self.path and os.path.exists(self.path):
            try: z = json.load(open(self.path, encoding="utf-8")); self.cases = z.get("cases", []); self.links = z.get("links", {})
            except Exception: self.cases = []
