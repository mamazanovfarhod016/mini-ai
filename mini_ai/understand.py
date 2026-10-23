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

# ---------- отношения ----------
# ГРАММАТИКА (закрытая, как предлоги): только «в» и составные «родился раньше».
# Все слова-сравнения (старше/тяжелее/сильнее/…) НЕ прописаны: они ОТКРЫВАЮТСЯ из формы фразы «A <слово> B (на N)»
# + смысла слова (meaning_ru: близость к уже известным сравнениям) + морфологии (-ее/-ше/-же/-че). MINI_AI_OPEN_REL=0 → старая закрытая таблица.
REL_GRAMMAR = {
    "в": ("in", None, "loc"), "внутри": ("in", None, "loc"), "входит в": ("in", None, "loc"), "находится в": ("in", None, "loc"), "часть": ("in", None, "loc"),
    "родился раньше": ("born_earlier", "born_later", "cmp"), "родился позже": ("born_later", "born_earlier", "cmp"),
}
REL_CLOSED = {
    "старше": ("older", "younger", "cmp"), "младше": ("younger", "older", "cmp"), "моложе": ("younger", "older", "cmp"),
    "выше": ("taller", "shorter", "cmp"), "ниже": ("shorter", "taller", "cmp"),
    "больше": ("more", "less", "cmp"), "меньше": ("less", "more", "cmp"),
    "дороже": ("pricier", "cheaper", "cmp"), "дешевле": ("cheaper", "pricier", "cmp"),
    "быстрее": ("faster", "slower", "cmp"), "медленнее": ("slower", "faster", "cmp"),
    "раньше": ("earlier", "later", "cmp"), "позже": ("later", "earlier", "cmp"),
    "тяжелее": ("heavier", "lighter", "cmp"), "легче": ("lighter", "heavier", "cmp"),
    "сильнее": ("stronger", "weaker", "cmp"), "слабее": ("weaker", "stronger", "cmp"),
}
OPEN_REL = os.environ.get("MINI_AI_OPEN_REL", "1") != "0"
REL = dict(REL_GRAMMAR) if OPEN_REL else {**REL_CLOSED, **REL_GRAMMAR}
REL_WORDS = sorted(REL, key=len, reverse=True)
REL_SAY = {"older": "старше", "younger": "младше", "taller": "выше", "shorter": "ниже", "more": "больше", "less": "меньше", "pricier": "дороже", "cheaper": "дешевле",
           "faster": "быстрее", "slower": "медленнее", "earlier": "раньше", "later": "позже", "heavier": "тяжелее", "lighter": "легче", "stronger": "сильнее", "weaker": "слабее",
           "in": "в", "born_earlier": "родился раньше", "born_later": "родился позже"}
CMP_SUFFIX = re.compile(r"(?:ее|ей|ше|же|че|ще|ьше)$")
STOP_REL = set("меня тебя нас вас него нее них у и а но или же ли бы не ни это этот эта то те он она оно они его ее их ему ей им на с со у к ко о об от до из за по под над при для про без через между есть был была было были будет быть "
               "мы вы я ты мне тебе нам вам мой моя наш ваш свой так также тоже уже еще очень как что кто где когда который чем тем самый самая самое год года лет см кг км сум раз".split())
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
        self.open_rels = {}          # слово → (имя, обратное|None, "cmp"): отношения, ОТКРЫТЫЕ из текста/объяснений, не из таблицы
        self.rel_seen = collections.Counter()
        self._M = None               # meaning_ru (ленивая загрузка) — смысл слова как второй свидетель «это сравнение»
        self.load()

    # ---------- открытые отношения ----------
    def _meaning(self):
        if self._M is None:
            self._M = False
            try:
                import numpy as np
                from mini_ai.meaning import Meaning
                from mini_ai.russian import RU_BASE_NPZ
                if os.path.exists(RU_BASE_NPZ):
                    R = getattr(self.mind, "russian", None); M = getattr(R, "meaning_ru", None) if R else None   # одна копия базы на сервер
                    if M is None: M = Meaning(None); M.load(RU_BASE_NPZ)
                    self._M = M
            except Exception as e: print("understand: meaning off:", e)
        return self._M or None

    def _cmp_sim(self, w):
        """близость смысла слова к уже известным сравнениям (из meaning_ru), 0 если базы нет"""
        M = self._meaning(); seeds = [x for x, v in self.open_rels.items() if v[2] == "cmp"] + [x for x, v in REL_CLOSED.items()][:0]
        if not M or not seeds: return 0.0
        import numpy as np
        def v(x):
            a = M.vec(x.replace("ё", "е")); return None if a is None else a / (np.linalg.norm(a) + 1e-9)
        a = v(w)
        if a is None: return 0.0
        S = [v(x) for x in seeds[:30]]; S = [t for t in S if t is not None]
        return max((float(a @ t) for t in S), default=0.0)

    def _all_rels(self): return {**REL, **self.open_rels}

    def _add_open_rel(self, w, inverse=None, why="form"):
        w = w.lower().replace("ё", "е")
        if w in REL or w in self.open_rels: return self.open_rels.get(w)
        self.open_rels[w] = (w, inverse, "cmp"); REL_SAY.setdefault(w, w); self.rel_seen[w] += 1
        self.log.append(dict(kind="open_rel", word=w, why=why)); return self.open_rels[w]

    def _discover_rel(self, sent):
        """«A <слово> B (на N)» без известного отношения → кандидат = слово между двумя сущностями.
        Свидетели: форма (между сущностями, «на N» рядом), морфология сравнительной степени, смысл (близость к известным сравнениям).
        Нужны ≥2 свидетеля ИЛИ форма + «на N»."""
        toks = re.findall(r"[А-ЯЁа-яё]+|\d+", sent)
        for i, t in enumerate(toks):
            tl = t.lower().replace("ё", "е")
            if t[0].isupper() or tl in STOP_REL or tl in self.VERBS or t.isdigit() or len(tl) < 4: continue
            ent = lambda x: len(x) >= 3 and not x.isdigit() and x.lower() not in STOP_REL and x.lower() not in self.VERBS
            left = [x for x in toks[:i] if ent(x)]; right = [x for x in toks[i+1:] if ent(x)]
            if not left or not right: continue
            num = 1 if re.search(r"\bна\s+\d", sent[sent.find(t):]) else 0
            morph = 1 if CMP_SUFFIX.search(tl) else 0
            sim = self._cmp_sim(tl); sem = 1 if sim >= 0.2 else 0
            # форма «A w B» + (сравнительная степень ИЛИ «на N» вместе со смыслом)
            if morph or (num and sem):
                self._add_open_rel(tl, why=f"form{'+на N' if num else ''}{'+morph' if morph else ''}{f'+sim{sim:.2f}' if sem else ''}")
                return self.open_rels[tl], tl
        return None, None

    # ================= 1. ОБЪЯСНЕНИЕ → СТРУКТУРА =================
    def explain(self, text):
        """учитель объясняет → правило/формула. Возвращает (правило, пересказ) или (None, что не понято)."""
        t = text.strip().rstrip(".")
        self._words_as_rels(t)
        m0 = re.match(r"^([а-яё]+)\s*[—\-:]?\s*(?:это\s+)?(?:транзитив|переносится|по цепочке|сравнение)", t.lower().replace("ё", "е"))
        if m0 and m0.group(1) not in ("любое", "всякое", "все", "каждое", "смотри", "например", "вот"): self._add_open_rel(m0.group(1), why="teacher: transitive")
        # пример с конкретными именами (≥2 Имени с большой буквы или ≥2 посылок) разбирается раньше формулировок
        ex = self._parse_example(t) if (len(re.findall(r"\b[А-ЯЁ][а-яё]{2,}", t)) >= 2 or t.count(",") >= 1) and re.search(r"значит|поэтому|следовательно|выходит|получается", t, re.I) and not re.match(r"^(?:если|когда)", t, re.I) else None
        r = ex or self._parse_formula(t) or self._parse_inverse(t) or self._parse_transitive(t) or self._parse_if_then(t) or self._parse_equiv(t)
        if r is None: return None, "Не поняла объяснение: не вижу в нём отношения между X и Y (старше/в/стоит…) или формулы (A = B / C)."
        if any(x.text == r.text for x in self.rules) and r.kind != "inverse": return r, "Это я уже знаю: " + self.retell(r)
        self.rules.append(r); self.save(); self.log.append(dict(kind="explain", text=text[:120]))
        return r, self.retell(r)

    def _rel_in(self, s):
        sl = s.lower().replace("ё", "е"); rels = self._all_rels()
        for w in sorted(rels, key=len, reverse=True):
            if re.search(r"(?<![а-яё])" + re.escape(w) + r"(?![а-яё])", sl): return rels[w], w
        return None, None

    def _rel_by_stem(self, word):
        """«старший/старшая/самый тяжёлый/высокая» → отношение по общей основе (старш↔старше, тяжел↔тяжелее, высок↔выше?)"""
        w = word.lower().replace("ё", "е"); rels = self._all_rels()
        if w in rels: return rels[w][0]
        base = re.sub(r"(?:ый|ий|ой|ая|яя|ое|ее|ые|ие|ого|ему|ым|их)$", "", w)
        best = None
        for rw, v in rels.items():
            if v[2] != "cmp" or " " in rw: continue
            rb = CMP_SUFFIX.sub("", rw)
            k = 0
            while k < min(len(base), len(rb)) and base[k] == rb[k]: k += 1
            if k >= 4 and (best is None or k > best[0]): best = (k, v[0])
        if best: return best[1]
        for a, b in (("высок", "выше"), ("низк", "ниже"), ("дорог", "дороже"), ("дешев", "дешевле"), ("тяжел", "тяжелее"), ("легк", "легче"), ("быстр", "быстрее"), ("медленн", "медленнее"), ("стар", "старше"), ("молод", "младше"), ("сильн", "сильнее"), ("слаб", "слабее"), ("больш", "больше"), ("маленьк", "меньше")):
            if base.startswith(a) and b in rels: return rels[b][0]
        return None

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

    def _parse_inverse(self, t):
        """«ниже — обратное к выше», «младше — это наоборот старше», «дешевле противоположно дороже»"""
        m = re.match(r"^([а-яё]+)\s*[—\-:]?\s*(?:это\s+)?(?:обратно[е]?\s+(?:к|для)?|наоборот|противоположно(?:\s+к)?|антоним(?:\s+к)?)\s*([а-яё]+)$", t.lower().replace("ё", "е"), re.I)
        if not m: return None
        a, b = m.group(1), m.group(2)
        self._add_open_rel(a, inverse=b, why="teacher: inverse"); self._add_open_rel(b, inverse=a, why="teacher: inverse")
        self.open_rels[a] = (a, b, "cmp"); self.open_rels[b] = (b, a, "cmp"); self.save()
        return Rule([[a, "X", "Y"]], [b, "Y", "X"], t, kind="inverse")

    def _words_as_rels(self, t):
        """слова объяснения вида «если X <w> Y …» — открыть <w> как отношение (учитель назвал его явно)"""
        for m in re.finditer(r"\b(?:X|Y|Z|[А-ЯЁ][а-яё]{2,})\s+([а-яё]+(?:\s+[а-яё]+)?)\s+(?:X|Y|Z|[А-ЯЁ][а-яё]{2,})\b", t):
            w = m.group(1).lower().replace("ё", "е")
            if w in REL or w in self.open_rels or w.split()[0] in STOP_REL: continue
            if CMP_SUFFIX.search(w.split()[-1]) or self._cmp_sim(w) >= 0.2 or re.search(r"транзитив|по цепочке|переносится", t, re.I): self._add_open_rel(w, why="teacher: rule")

    def _parse_example(self, t):
        """ПРАВИЛО ИЗ ПРИМЕРА, не из формулировки: «смотри: Диёр старше Сардора, Сардор старше Камрона — значит Диёр старше Камрона».
        Факты-посылки и вывод разбираются как задача; имена заменяются переменными по порядку появления → правило с X/Y/Z.
        Это индукция из ОДНОГО примера: обобщение = «имена не важны, важна структура»."""
        m = re.match(r"^(?:смотри|например|пример|вот)?\s*[:,]?\s*(.+?)\s*(?:[—\-]+|,)?\s*(?:значит|поэтому|следовательно|выходит|то есть|получается)\s*[,:]?\s*(.+)$", t, re.I)
        if not m: return None
        prem, concl = m.group(1), m.group(2)
        pf, _ = self.parse_task(prem + "."); cf, _ = self.parse_task(concl + ".")
        if len(pf) < 1 or len(cf) != 1: return None
        names = {}; var = iter("XYZWV")
        def v(n):
            k = self._phrase_stem(n)
            if k not in names: names[k] = next(var)
            return names[k]
        cond = [[rel, v(a), v(b)] for rel, a, b, _ in pf]
        rel, a, b, _ = cf[0]
        if self._phrase_stem(a) not in names or self._phrase_stem(b) not in names: return None   # вывод про кого-то, кого не было в посылках — не правило
        concl_s = [rel, v(a), v(b)]
        if concl_s in cond: return None
        kind = "transitive" if (len(cond) == 2 and cond[0][0] == cond[1][0] == rel and cond[0][2] == cond[1][1] and concl_s == [rel, cond[0][1], cond[1][2]]) else "rule"
        return Rule(cond, concl_s, t, kind=kind)

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
        if r.kind == "inverse":
            return f"Поняла: «{r.cond[0][0]}» — это «{r.concl[0]}» наоборот: если первый {r.cond[0][0]} второго, то второй {r.concl[0]} первого."
        if r.kind == "equiv":
            return f"Поняла: «{REL_SAY.get(r.cond[0][0])}» и «{REL_SAY.get(r.concl[0])}» — одно и то же."
        cs = " и ".join(f"{a} {REL_SAY.get(rel, rel)} {b}" for rel, a, b in r.cond); rel, a, b = r.concl
        pre = "Поняла из примера, имена не важны: " if re.search(r"значит|поэтому|следовательно|выходит|получается", r.text) and not re.match(r"^(?:если|когда)", r.text, re.I) else "Поняла: "
        return f"{pre}если {cs}, то {a} {REL_SAY.get(rel, rel)} {b}."

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
        rels_rx = "|".join(map(re.escape, sorted(self._all_rels(), key=len, reverse=True)))
        # новое предложение = после «,» идёт Имя / класс-слово / «слово <отношение>» / «слово <сравнит.степень>»
        sents = re.split(r"[.;!?]\s*|,\s*(?=[А-ЯЁ][а-яё]+\s)|,\s*(?=(?:область|район|город|страна)\s)|,\s*(?=[а-яё]+\s+(?:" + rels_rx + r")(?![а-яё]))|,\s*(?=[а-яё]+\s+[а-яё]+(?:ее|ей|ше|же|че)\s)", text)
        for sent in sents:
            sl = sent.lower().strip()
            if not sl or any(re.search(r"(?<![а-яё])" + q + r"(?![а-яё])", sl) for q in self.QWORDS): continue  # вопрос — не факт
            rel, w = self._rel_in(sent)
            if not rel and OPEN_REL: rel, w = self._discover_rel(sent)
            if rel:
                sl = sl.replace("ё", "е"); m = re.search(r"(?<![а-яё])" + re.escape(w) + r"(?![а-яё])", sl)
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
            inv = self._inverse(rel)
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
                        inv = self._inverse(rel)
                        if inv: known.setdefault((inv, b, a), nsum)
        # --- вопрос ---
        ans = None; why = []
        m = re.search(r"кто (?:самый |самая |самое |из них )?([а-яё]+)", q)
        if m and not re.search(r"\bили\b", q):
            rel = self._rel_by_stem(m.group(1))
            people = {x for (_, a, b) in known for x in (a, b) if x != "?"}
            if rel and people:
                tops = [p for p in people if all((rel, p, o) in known for o in people if o != p)]
                if tops: ans = tops[0]; why = [f"{a} {REL_SAY.get(rel, rel)} {b}" + (f" на {n:g}" if n else "") for (r_, a, b), n in known.items() if r_ == rel and a == ans]
        m = re.search(r"(?:где|в чём|в чем)\s+(?:находится\s+)?([А-ЯЁ][а-яё]+)", text, re.I)
        if m and not ans:
            x = self._name(m.group(1)); chain = [(b, n) for (r_, a, b), n in known.items() if r_ == "in" and a == x]
            if chain:
                # ответ — то, что ИЗВЕСТНО (прямо или по правилу) как содержащее x; «по-крупному» = самое внешнее из известных.
                # Без правила вложенности известно только прямое место — цепочку она сама не проходит (это и есть проверка).
                conts = [b for b, _ in chain]
                outer = [b for b in conts if not any((("in", b, c) in known) for c in conts if c != b)]
                ans = outer[0] if outer else conts[-1]; why = [f"{x} в {b}" for b in conts]
        m = re.search(r"(?:кто|что)\s+([а-яё]+)\s*[:—-]?\s*([А-ЯЁа-яё]+)\s+или\s+([А-ЯЁа-яё]+)", text, re.I)
        if m and not ans:
            rel = self._rel_by_stem(m.group(1)); a, b = self._name(m.group(2)), self._name(m.group(3))
            if rel and (rel, a, b) in known: ans = a; why = [f"{a} {REL_SAY.get(rel, rel)} {b}" + (f" на {known[(rel, a, b)]:g}" if known[(rel, a, b)] else "")]
            elif rel and (rel, b, a) in known: ans = b; why = [f"{b} {REL_SAY.get(rel, rel)} {a}" + (f" на {known[(rel, b, a)]:g}" if known[(rel, b, a)] else "")]
            if ans and re.search(r"на сколько", q):
                k = (rel, ans, b if ans == a else a)
                if known.get(k) is not None: ans = f"{ans}, на {known[k]:g}"
        m = re.search(r"на сколько\b", q)
        if m and ans is None and len(facts) >= 2:
            # просто «на сколько X старше Z» / «на сколько самса дороже чая»
            mm = re.search(r"на сколько\s+([А-ЯЁа-яё]+)\s+([а-яё]+)\s+([А-ЯЁа-яё]+)", text, re.I)
            if mm and self._rel_by_stem(mm.group(2)):
                rel = self._rel_by_stem(mm.group(2)); a, b = self._name(mm.group(1)), self._name(mm.group(3)); k = (rel, a, b)
                if k in known and known[k] is not None: ans = f"на {known[k]:g}"; why = [f"{a} {REL_SAY[rel]} {b} на {known[k]:g}"]
        if ans is None: return None
        used = [r.text for r, *_ in derived]
        for r, *_ in derived: r.applied += 1
        steps = [f"{a} {REL_SAY.get(rel, rel)} {b}" + (f" на {n:g}" if n else "") for rel, a, b, n in facts]
        steps += [f"по правилу «{r.text}»: {a} {REL_SAY.get(rel, rel)} {b}" + (f" на {n:g}" if n else "") for r, rel, a, b, n in derived[:4]]
        return dict(answer=ans, steps=steps, rule=used[0] if used else None, kind="rule", confidence=0.85 if derived else 0.7)

    def _inverse(self, rel):
        for v in self._all_rels().values():
            if v[0] == rel and v[1]: return v[1]
        return None

    def _match(self, cond, known):
        """все подстановки переменных, при которых все условия есть в known; суммирует числа («на 3» + «на 2» → 5)"""
        results = [({}, 0.0)]
        cmp_rels = {v[0] for v in self._all_rels().values() if v[2] == "cmp"}
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
        return f"understand: {len(self.open_rels)} открытых отношений ({', '.join(list(self.open_rels)[:8])}), {len(self.rules)} правил от учителя ({sum(1 for r in self.rules if r.kind == 'formula')} формул), применено {sum(r.applied for r in self.rules)} раз"

    def save(self):
        if self.path: json.dump(dict(rules=[r.to_json() for r in self.rules], open_rels=self.open_rels, log=self.log[-100:]), open(self.path, "w"), ensure_ascii=False, indent=1)
    def load(self):
        if not self.path or not os.path.exists(self.path): return
        try:
            z = json.load(open(self.path, encoding="utf-8")); self.rules = [Rule.from_json(d) for d in z.get("rules", [])]; self.log = z.get("log", [])
            self.open_rels = {k: tuple(v) for k, v in z.get("open_rels", {}).items()}
            for k in self.open_rels: REL_SAY.setdefault(k, k)
            for r in self.rules:
                if r.kind == "formula": self.formulas[r.concl[0]] = r
        except Exception: self.rules = []
