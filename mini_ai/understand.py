"""understand.py — ПОНИМАНИЕ ОБЪЯСНЕНИЯ УЧИТЕЛЯ (без параметров, без градиента). 23.10.2026, задание Фархода:
«не надо, чтоб он от меня учился — нужно, чтоб ты его учил, и он понимал тебя; как ChatGPT или Claude».

Проверка понимания одна: учитель объясняет ОДИН раз → модель пересказывает своими словами → применяет к НОВОЙ задаче,
где слова другие, а структура та же. Не пересказала — не поняла; не применила — не поняла. Копирование фраз = не понимание.

Как объяснение становится не текстом, а структурой:
  объяснение  «если X старше Y, то X родился раньше Y»          → ПРАВИЛО: cond [X старше Y] ⇒ concl [X родился раньше Y]
  объяснение  «старше — это транзитивно: X>Y, Y>Z ⇒ X>Z»          → ПРАВИЛО с цепочкой
  объяснение  «время = путь / скорость»                          → ФОРМУЛА: time = dist / speed (роли: км → dist, км/ч → speed)
  объяснение  «остаток = было − потратил»                        → ФОРМУЛА
  объяснение  «если X в Y и Y в Z, то X в Z»                     → ПРАВИЛО (вложенность)
Правило хранится как (условия над отношениями, вывод над отношениями). Задача разбирается в те же отношения
(parse: «Шерзод старше Азизы на 3» → (Шерзод, старше, Азиза, 3)). Применение = сопоставление структур + вычисление.
Пересказ = правило → фраза по шаблону отношения (без заучивания формулировки объяснения).

Отношения — закрытая таблица грамматики (как предлоги): старше/младше/выше/ниже/больше/меньше/дороже/дешевле/быстрее/
медленнее/раньше/позже/в(внутри)/стоит/осталось/путь/скорость/время. Знание (правила, формулы) — только от учителя.
"""
import re, json, os, collections

# ---------- отношения (грамматика): слово → (имя отношения, обратное, тип) ----------
REL = {
    "старше": ("older", "younger", "cmp"), "младше": ("younger", "older", "cmp"), "моложе": ("younger", "older", "cmp"),
    "выше": ("taller", "shorter", "cmp"), "ниже": ("shorter", "taller", "cmp"),
    "больше": ("more", "less", "cmp"), "меньше": ("less", "more", "cmp"),
    "дороже": ("pricier", "cheaper", "cmp"), "дешевле": ("cheaper", "pricier", "cmp"),
    "быстрее": ("faster", "slower", "cmp"), "медленнее": ("slower", "faster", "cmp"),
    "раньше": ("earlier", "later", "cmp"), "позже": ("later", "earlier", "cmp"),
    "тяжелее": ("heavier", "lighter", "cmp"), "легче": ("lighter", "heavier", "cmp"),
    "сильнее": ("stronger", "weaker", "cmp"), "слабее": ("weaker", "stronger", "cmp"),
    "в": ("in", None, "loc"), "внутри": ("in", None, "loc"), "входит в": ("in", None, "loc"), "находится в": ("in", None, "loc"), "часть": ("in", None, "loc"),
    "родился раньше": ("born_earlier", "born_later", "cmp"), "родился позже": ("born_later", "born_earlier", "cmp"),
}
REL_WORDS = sorted(REL, key=len, reverse=True)
REL_SAY = {"older": "старше", "younger": "младше", "taller": "выше", "shorter": "ниже", "more": "больше", "less": "меньше", "pricier": "дороже", "cheaper": "дешевле",
           "faster": "быстрее", "slower": "медленнее", "earlier": "раньше", "later": "позже", "heavier": "тяжелее", "lighter": "легче", "stronger": "сильнее", "weaker": "слабее",
           "in": "в", "born_earlier": "родился раньше", "born_later": "родился позже"}
VARS = {"x", "y", "z", "а", "б", "в", "икс", "игрек", "зет", "a", "b", "c"}
NAME_RX = re.compile(r"\b([А-ЯЁ][а-яё]{2,}|[XYZ]|[XYZ][a-z]?)\b")
NUM_RX = re.compile(r"(\d+(?:[.,]\d+)?)")
UNITS = {"dist": ("км", "километр", "километра", "километров", "м", "метр", "метров"), "speed": ("км/ч", "км / ч", "м/с"), "time": ("ч", "час", "часа", "часов", "мин", "минут", "минуты"),
         "money": ("сум", "сумов", "доллар", "долларов", "рубл", "$"), "price": ("стоит", "стоят", "по", "цена")}

def _num(s): return float(s.replace(",", "."))

class Rule:
    """cond: список (rel, A, B) с переменными; concl: (rel, A, B) ИЛИ формула"""
    def __init__(self, cond, concl, text, kind="rule"):
        self.cond, self.concl, self.text, self.kind = cond, concl, text, kind
        self.applied = 0; self.ok = 0
    def to_json(self): return dict(cond=self.cond, concl=self.concl, text=self.text, kind=self.kind, applied=self.applied, ok=self.ok)
    @staticmethod
    def from_json(d): r = Rule(d["cond"], d["concl"], d["text"], d.get("kind", "rule")); r.applied = d.get("applied", 0); r.ok = d.get("ok", 0); return r

class Understand:
    def __init__(self, mind=None, state_dir=None):
        self.mind = mind; self.path = os.path.join(state_dir, "understand.json") if state_dir else None
        self.rules = []; self.formulas = {}; self.log = []
        self.load()

    # ================= 1. ОБЪЯСНЕНИЕ → СТРУКТУРА =================
    def explain(self, text):
        """учитель объясняет → правило/формула. Возвращает (правило, пересказ) или (None, что не понято)."""
        t = text.strip().rstrip(".")
        r = self._parse_formula(t) or self._parse_transitive(t) or self._parse_if_then(t) or self._parse_equiv(t)
        if r is None: return None, "Не поняла объяснение: не вижу в нём отношения между X и Y (старше/в/стоит…) или формулы (A = B / C)."
        if any(x.text == r.text for x in self.rules): return r, "Это я уже знаю: " + self.retell(r)
        self.rules.append(r); self.save(); self.log.append(dict(kind="explain", text=text[:120]))
        return r, self.retell(r)

    def _rel_in(self, s):
        sl = s.lower()
        for w in REL_WORDS:
            if re.search(r"(?<![а-яё])" + re.escape(w) + r"(?![а-яё])", sl): return REL[w], w
        return None, None

    def _vars(self, s):
        """переменные в объяснении: X, Y, Z / икс, игрек / имена-заглушки (Петя, Вася) → порядковые"""
        found = []
        for m in re.finditer(r"\b([XYZxyzАБВ]|икс|игрек|зет|[А-ЯЁ][а-яё]{2,})\b", s):
            w = m.group(1); k = w.lower()
            if k in ("если", "то", "значит", "тогда", "и", "а", "это", "когда", "чем", "тем"): continue
            if w not in found: found.append(w)
        return found

    def _parse_if_then(self, t):
        m = re.match(r"^(?:если|когда)\s+(.+?)[,\s]+(?:то|значит|тогда)\s+(.+)$", t, re.I)
        if not m: return None
        cond_s, concl_s = m.group(1), m.group(2)
        conds = []
        for part in re.split(r"\s*,?\s+(?:и|а)\s+|\s*,\s*", cond_s):
            rel, w = self._rel_in(part)
            if not rel: continue
            vs = [v for v in self._vars(part) if v.lower() != w]
            if len(vs) >= 2: conds.append([rel[0], vs[0], vs[1]])
            elif len(vs) == 1 and rel[2] == "loc": conds.append([rel[0], vs[0], "?"])
        rel2, w2 = self._rel_in(concl_s)
        if not conds or not rel2: return None
        vs2 = [v for v in self._vars(concl_s) if v.lower() != w2]
        if len(vs2) < 2: return None
        return Rule(conds, [rel2[0], vs2[0], vs2[1]], t)

    def _parse_transitive(self, t):
        """«старше — это транзитивно», «если X старше Y, а Y старше Z, то X старше Z», «старше переносится по цепочке»"""
        if re.search(r"^(?:любое|всякое|все|всякие|любые)\s+сравнени", t, re.I) and re.search(r"транзитив|по цепочке|переносится|передаётся|передается", t, re.I):
            return Rule([["*cmp", "X", "Y"], ["*cmp", "Y", "Z"]], ["*cmp", "X", "Z"], t, kind="transitive")  # класс отношений, не одно слово
        rel, w = self._rel_in(t)
        if not rel: return None
        if re.search(r"транзитив|по цепочке|переносится|передаётся|передается", t, re.I):
            return Rule([[rel[0], "X", "Y"], [rel[0], "Y", "Z"]], [rel[0], "X", "Z"], t, kind="transitive")
        return None

    def _parse_equiv(self, t):
        """«старше — значит родился раньше», «младше это то же, что моложе»"""
        m = re.match(r"^(.+?)\s*[—\-:]\s*(?:это\s+)?(?:значит|то же,? что|то есть|означает)\s+(.+)$", t, re.I)
        if not m: return None
        a, b = self._rel_in(m.group(1)), self._rel_in(m.group(2))
        if not a[0] or not b[0]: return None
        return Rule([[a[0][0], "X", "Y"]], [b[0][0], "X", "Y"], t, kind="equiv")

    def _parse_formula(self, t):
        """«время = путь / скорость», «остаток = было − потратил», «сдача = дал − стоит»"""
        m = re.match(r"^([а-яё\s]+?)\s*(?:=|равно|это)\s*([а-яё\s]+?)\s*([-−+*/×:]|делить на|умножить на|плюс|минус)\s*([а-яё\s]+)$", t, re.I)
        if not m: return None
        out, a, op, b = (m.group(1).strip().lower(), m.group(2).strip().lower(), m.group(3).strip().lower(), m.group(4).strip().lower())
        op = {"делить на": "/", "умножить на": "*", "плюс": "+", "минус": "-", "−": "-", "×": "*", ":": "/"}.get(op, op)
        roles = {"время": "time", "путь": "dist", "расстояние": "dist", "скорость": "speed", "остаток": "left", "было": "had", "потратил": "spent", "потратила": "spent",
                 "сдача": "change", "дал": "gave", "дала": "gave", "стоит": "price", "цена": "price", "сумма": "total", "количество": "count", "штук": "count", "всего": "total"}
        ro, ra, rb = roles.get(out, out), roles.get(a, a), roles.get(b, b)
        r = Rule([[ra, "?", "?"], [rb, "?", "?"]], [ro, ra, op, rb], t, kind="formula"); self.formulas[ro] = r; return r

    # ================= 2. ПЕРЕСКАЗ (из структуры, не из текста) =================
    def retell(self, r):
        if r.kind == "formula":
            ro, a, op, b = r.concl; name = {"time": "время", "dist": "путь", "speed": "скорость", "left": "остаток", "had": "было", "spent": "потрачено", "change": "сдача", "gave": "дал", "price": "цена", "total": "всего", "count": "количество"}
            ops = {"/": "разделить на", "*": "умножить на", "+": "сложить с", "-": "вычесть"}
            return f"Поняла: чтобы найти {name.get(ro, ro)}, надо {name.get(a, a)} {ops[op]} {name.get(b, b)}."
        if r.kind == "transitive":
            if r.concl[0] == "*cmp": return "Поняла: любое сравнение (старше, выше, тяжелее, дороже…) переносится по цепочке — если первый больше второго, а второй третьего, то первый больше третьего."
            s = REL_SAY.get(r.concl[0], r.concl[0]); return f"Поняла: «{s}» переносится по цепочке — если первый {s} второго, а второй {s} третьего, то первый {s} третьего."
        if r.kind == "equiv":
            return f"Поняла: «{REL_SAY.get(r.cond[0][0])}» и «{REL_SAY.get(r.concl[0])}» — одно и то же."
        cs = " и ".join(f"{a} {REL_SAY.get(rel, rel)} {b}" for rel, a, b in r.cond); rel, a, b = r.concl
        return f"Поняла: если {cs}, то {a} {REL_SAY.get(rel, rel)} {b}."

    # ================= 3. РАЗБОР ЗАДАЧИ В ОТНОШЕНИЯ =================
    # --- имена: русские падежи → одна сущность по основе («Сардора» = «Сардор», «Азизы» = «Азиза») ---
    CLASS_WORDS = ("область", "области", "район", "районе", "города", "городе", "город", "страна", "стране")
    QWORDS = ("кто", "что", "где", "сколько", "какой", "какая", "в чём", "в чем", "или")
    VERBS = ("бежит", "бегает", "едет", "идёт", "идет", "живёт", "живет", "стоит", "весит", "находится", "входит", "лежит", "прыгает", "плавает", "растёт", "растет", "есть", "был", "была", "будет")

    @staticmethod
    def _stem(w):
        w = w.lower().replace("ё", "е")
        for e in ("ами", "ями", "ого", "его", "ому", "ему", "ой", "ей", "ом", "ем", "ую", "юю", "ая", "яя", "ие", "ые", "ах", "ях", "а", "я", "ы", "и", "у", "ю", "е", "о", "й", "ь"):
            if w.endswith(e) and len(w) - len(e) >= 3: return w[: -len(e)]
        return w

    def _phrase_stem(self, p): return " ".join(self._stem(x) for x in p.split())

    def _entity(self, side, first):
        """имя-сущность с левого/правого края отношения: «Диёр на 2 года» → Диёр; «Ферганской области» → Ферганской области"""
        toks = [t for t in re.findall(r"[А-ЯЁа-яё][а-яё]+", side) if t.lower() not in self.VERBS]
        toks = [t for t in toks if t.lower() not in ("на", "в", "и", "а", "но", "у", "меня", "же", "то", "ли", "см", "год", "года", "лет", "км", "сум", "раз", "раза", "чем", "самый", "самая")]
        if not toks: return None
        if first:
            ent = toks[0]
            if len(toks) > 1 and toks[1].lower() in self.CLASS_WORDS: ent += " " + toks[1]
        else:
            ent = toks[-1]
            if len(toks) > 1 and ent.lower() in self.CLASS_WORDS: ent = toks[-2] + " " + ent
        return ent

    def _canon_all(self, names, text):
        """карта основа → каноническая форма (та, что стоит подлежащим или в вопросе после «или/где»)"""
        pref = set(re.findall(r"(?:или|где(?: находится)?|кто|что)\s+([А-ЯЁа-яё]+(?: (?:область|район|город))?)", text, re.I))
        canon = {}
        for n, is_subj in names:
            s = self._phrase_stem(n)
            if s not in canon or (is_subj and not canon[s][1]): canon[s] = (n, is_subj)
        for n in pref:  # форма из вопроса («кто старше: Нигора или Азиза») — именительный падеж
            s = self._phrase_stem(n)
            if s in canon: canon[s] = (n, True)
        return {s: (v[0] if v[1] else self._nominative(v[0])) for s, v in canon.items()}

    @staticmethod
    def _nominative(w):
        """имя встречено только в косвенном падеже → угадать именительный: Сардора→Сардор, Акмаля→Акмаль, Азизы→Азиза, Малики→Малика"""
        if " " in w: return w
        for e, r in (("ы", "а"), ("ки", "ка"), ("ги", "га"), ("хи", "ха"), ("и", "я"), ("я", "ь"), ("а", ""), ("ом", ""), ("ем", "ь"), ("у", ""), ("ю", "ь"), ("е", "")):
            if w.endswith(e) and len(w) - len(e) >= 3: return w[: -len(e)] + r
        return w

    def parse_task(self, text):
        facts = []; quants = {}; raw = []; last_place = None
        sents = re.split(r"[.;!?]\s*|,\s*(?=[а-яёА-ЯЁ][^,]*\b(?:" + "|".join(map(re.escape, REL_WORDS)) + r")\b)", text)
        for sent in sents:
            sl = sent.lower().strip()
            if not sl or any(re.search(r"(?<![а-яё])" + q + r"(?![а-яё])", sl) for q in self.QWORDS): continue  # вопрос — не факт
            rel, w = self._rel_in(sent)
            if rel:
                m = re.search(r"(?<![а-яё])" + re.escape(w) + r"(?![а-яё])", sl)
                left, right = sent[: m.start()], sent[m.end():]
                m_n = re.search(r"\bна\s+(\d+(?:[.,]\d+)?)", sent)
                right_core = re.split(r"\bна\s+\d", right)[0]
                a = self._entity(left, first=True); b = self._entity(right_core, first=True)
                if a is None and re.search(r"\b(?:я|мне|меня|у меня)\b", left.lower()): a = "я"
                # «…области, область в Узбекистане» — класс-слово в начале = предыдущее место
                if a and a.lower() in self.CLASS_WORDS and last_place and self._stem(a) in self._phrase_stem(last_place): a = last_place
                if a and b:
                    facts.append((rel[0], a, b, _num(m_n.group(1)) if m_n else None)); raw.append((a, True)); raw.append((b, False))
                    if rel[2] == "loc": last_place = b
            for role, units in UNITS.items():
                for u in units:
                    m = re.search(r"(\d+(?:[.,]\d+)?)\s*" + re.escape(u) + r"(?![а-яё/])", sent)
                    if m and role not in quants: quants[role] = _num(m.group(1))
        # роли по словам: «было 250000», «потратил 40000», «дал 5000», «стоит 3500», «едет 15 км/ч» (скорость без ч)
        for role, rx in [("had", r"(?:было|есть|имею|у меня)\s+(\d[\d\s]*)"), ("spent", r"(?:потратил[а]?|купил[а]?[^0-9]{0,30}за|отдал[а]?|ушло)\s+(\d[\d\s]*)"),
                         ("gave", r"(?:дал[а]?|заплатил[а]?|протянул[а]?)\s+(\d[\d\s]*)"), ("price", r"(?:стоит|стоят|цена|по)\s+(\d[\d\s]*)"), ("speed", r"(\d+)\s*км\s*/\s*ч"), ("dist", r"(\d+)\s*км(?!\s*/)")]:
            m = re.search(rx, text, re.I)
            if m and role not in quants: quants[role] = _num(m.group(1).replace(" ", ""))
        # падежи → одна сущность: ключ = основа, показ = именительная форма (подлежащее / из вопроса)
        canon = self._canon_all(raw, text); self._canon = canon
        facts = [(rel, canon.get(self._phrase_stem(a), a) if a != "я" else "я", canon.get(self._phrase_stem(b), b), n) for rel, a, b, n in facts]
        return facts, quants

    def _name(self, w):
        """имя из вопроса (любой падеж) → каноническая форма из фактов"""
        c = getattr(self, "_canon", {}); s = self._phrase_stem(w)
        if s in c: return c[s]
        for k, v in c.items():
            if k.startswith(s) or s.startswith(k): return v
        return w

    # ================= 4. ПРИМЕНЕНИЕ =================
    def apply(self, text):
        facts, quants = self.parse_task(text)
        q = text.lower()
        # --- формулы ---
        for ro, r in self.formulas.items():
            _, a, op, b = r.concl
            ask = {"time": r"сколько (?:ехать|времени|займ|идти|минут|часов)|за сколько", "left": r"сколько (?:у меня )?осталось|остаток|остан", "change": r"сдач", "total": r"сколько всего|итого|сумм"}.get(ro)
            if ask and re.search(ask, q) and a in quants and b in quants:
                va, vb = quants[a], quants[b]; val = {"+": va + vb, "-": va - vb, "*": va * vb, "/": (va / vb if vb else None)}[op]
                if val is None: continue
                r.applied += 1
                unit = {"time": "ч", "left": "сум", "change": "сум", "total": ""}.get(ro, "")
                shown = f"{val:g}"
                if ro == "time" and val < 1: shown = f"{val:g} ч = {round(val * 60)} мин"
                names = {"time": "время", "dist": "путь", "speed": "скорость", "left": "остаток", "had": "было", "spent": "потратил", "change": "сдача", "gave": "дал", "price": "цена", "total": "всего"}
                steps = [f"{names[a]} = {va:g}", f"{names[b]} = {vb:g}", f"по правилу «{r.text}»: {va:g} {op} {vb:g} = {val:g}"]
                return dict(answer=shown, steps=steps, rule=r.text, kind="formula", confidence=0.9)
        # --- правила над отношениями: замыкание ---
        if not facts: return None
        known = {(rel, a, b): n for rel, a, b, n in facts}
        # обратные отношения тоже известны
        for (rel, a, b), n in list(known.items()):
            inv = next((v[1] for v in REL.values() if v[0] == rel and v[1]), None)
            if inv: known.setdefault((inv, b, a), n)
        derived = []; changed = True; guard = 0
        while changed and guard < 6:
            changed = False; guard += 1
            for r in self.rules:
                if r.kind == "formula": continue
                for binding, nsum in self._match(r.cond, known):
                    rel, A, B = r.concl; a, b = binding.get(A, A), binding.get(B, B)
                    if rel == "*cmp": rel = binding["*cmp"]
                    if (rel, a, b) not in known:
                        known[(rel, a, b)] = nsum; derived.append((r, rel, a, b, nsum)); changed = True
                        inv = next((v[1] for v in REL.values() if v[0] == rel and v[1]), None)
                        if inv: known.setdefault((inv, b, a), nsum)
        # --- вопрос ---
        ans = None; why = []
        m = re.search(r"кто (?:самый |самая |из них )?(старш|младш|выш|ниж|больш|меньш|дорож|дешевл|быстр|медленн|старее|раньше|позже|тяжел|легч|сильн|слаб|высок|низк|дорог|дешев|тяжёл|лёгк|сильн|слаб)", q)
        if m and not re.search(r"\bили\b", q):
            key = m.group(1); key = {"высок": "выше", "низк": "ниже", "дорог": "дороже", "дешев": "дешевле", "тяжёл": "тяжелее", "лёгк": "легче"}.get(key, key)
            rel = next((REL[w][0] for w in REL if w.startswith(key[:4])), None)
            people = {x for (_, a, b) in known for x in (a, b) if x != "?"}
            if rel and people:
                tops = [p for p in people if all((rel, p, o) in known for o in people if o != p)]
                if tops: ans = tops[0]; why = [f"{a} {REL_SAY[rel]} {b}" + (f" на {n:g}" if n else "") for (r_, a, b), n in known.items() if r_ == rel and a == ans]
        m = re.search(r"(?:где|в чём|в чем)\s+(?:находится\s+)?([А-ЯЁ][а-яё]+)", text, re.I)
        if m and not ans:
            x = self._name(m.group(1)); chain = [(b, n) for (r_, a, b), n in known.items() if r_ == "in" and a == x]
            if chain:
                # ответ — то, что ИЗВЕСТНО (прямо или по правилу) как содержащее x; «по-крупному» = самое внешнее из известных.
                # Без правила вложенности известно только прямое место — цепочку она сама не проходит (это и есть проверка).
                conts = [b for b, _ in chain]
                outer = [b for b in conts if not any((("in", b, c) in known) for c in conts if c != b)]
                ans = outer[0] if outer else conts[-1]; why = [f"{x} в {b}" for b in conts]
        m = re.search(r"(?:кто|что)\s+(" + "|".join(w for w in REL_WORDS if " " not in w and REL[w][2] == "cmp") + r")\s*[:—-]?\s*([А-ЯЁа-яё]+)\s+или\s+([А-ЯЁа-яё]+)", text, re.I)
        if m and not ans:
            rel = REL[m.group(1)][0]; a, b = self._name(m.group(2)), self._name(m.group(3))
            if (rel, a, b) in known: ans = a; why = [f"{a} {REL_SAY[rel]} {b}" + (f" на {known[(rel, a, b)]:g}" if known[(rel, a, b)] else "")]
            elif (rel, b, a) in known: ans = b; why = [f"{b} {REL_SAY[rel]} {a}" + (f" на {known[(rel, b, a)]:g}" if known[(rel, b, a)] else "")]
            if ans and re.search(r"на сколько", q):
                k = (rel, ans, b if ans == a else a)
                if known.get(k) is not None: ans = f"{ans}, на {known[k]:g}"
        m = re.search(r"на сколько\b", q)
        if m and ans is None and len(facts) >= 2:
            # просто «на сколько X старше Z» / «на сколько самса дороже чая»
            mm = re.search(r"на сколько\s+([А-ЯЁа-яё]+)\s+(старше|младше|выше|ниже|больше|меньше|дороже|дешевле)\s+([А-ЯЁа-яё]+)", text, re.I)
            if mm:
                rel = REL[mm.group(2)][0]; a, b = self._name(mm.group(1)), self._name(mm.group(3)); k = (rel, a, b)
                if k in known and known[k] is not None: ans = f"на {known[k]:g}"; why = [f"{a} {REL_SAY[rel]} {b} на {known[k]:g}"]
        if ans is None: return None
        used = [r.text for r, *_ in derived]
        for r, *_ in derived: r.applied += 1
        steps = [f"{a} {REL_SAY.get(rel, rel)} {b}" + (f" на {n:g}" if n else "") for rel, a, b, n in facts]
        steps += [f"по правилу «{r.text}»: {a} {REL_SAY.get(rel, rel)} {b}" + (f" на {n:g}" if n else "") for r, rel, a, b, n in derived[:4]]
        return dict(answer=ans, steps=steps, rule=used[0] if used else None, kind="rule", confidence=0.85 if derived else 0.7)

    def _match(self, cond, known):
        """все подстановки переменных, при которых все условия есть в known; суммирует числа («на 3» + «на 2» → 5)"""
        results = [({}, 0.0)]
        cmp_rels = {v[0] for v in REL.values() if v[2] == "cmp"}
        for rel, A, B in cond:
            nxt = []
            for binding, nsum in results:
                for (r_, a, b), n in known.items():
                    if rel == "*cmp":
                        if r_ not in cmp_rels or binding.get("*cmp", r_) != r_: continue
                    elif r_ != rel: continue
                    bnd = dict(binding); ok = True
                    if rel == "*cmp": bnd["*cmp"] = r_
                    for var, val in ((A, a), (B, b)):
                        if var in bnd and bnd[var] != val: ok = False; break
                        bnd[var] = val
                    if ok and len(set(v for k, v in bnd.items() if k != "*cmp")) == len([k for k in bnd if k != "*cmp"]): nxt.append((bnd, nsum + (n or 0.0)))
            results = nxt
            if not results: return []
        return [(b, (n if n else None)) for b, n in results]

    # ================= 5. ОБРАТНАЯ СВЯЗЬ УЧИТЕЛЯ =================
    def feedback(self, rule_text, correct):
        for r in self.rules:
            if r.text == rule_text: r.ok += 1 if correct else 0
        self.save()

    def say_steps(self, res):
        return "Думаю так: " + " ".join(f"{i+1}) {s}." for i, s in enumerate(res["steps"])) + f" Значит: {res['answer']}."

    def report(self):
        return f"understand: {len(self.rules)} правил от учителя ({sum(1 for r in self.rules if r.kind == 'formula')} формул), применено {sum(r.applied for r in self.rules)} раз"

    def save(self):
        if self.path: json.dump(dict(rules=[r.to_json() for r in self.rules], log=self.log[-100:]), open(self.path, "w"), ensure_ascii=False, indent=1)
    def load(self):
        if not self.path or not os.path.exists(self.path): return
        try:
            z = json.load(open(self.path, encoding="utf-8")); self.rules = [Rule.from_json(d) for d in z.get("rules", [])]; self.log = z.get("log", [])
            for r in self.rules:
                if r.kind == "formula": self.formulas[r.concl[0]] = r
        except Exception: self.rules = []
