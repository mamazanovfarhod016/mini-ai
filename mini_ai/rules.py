"""rules.py — машина вывода без обучения: отношения с ПРАВИЛАМИ (не веса), применяемые к памяти фактов.

Что в архитектуре (не выучено, не настроено на данных):
  • Отношение имеет свойства: транзитивно (bigger_than, left_of, fits_in), есть обратное (left_of ↔ right_of,
    north_of ↔ south_of, bigger_than ↔ fits_in-обратно), исключающее (место: один объект в одном месте — новый
    факт о месте вытесняет старый — «состояние мира»).
  • Дедукция по типу: «Wolves are afraid of mice» + «Gertrude is a wolf» ⇒ Gertrude afraid of mice (правило класса
    применяется к экземпляру); цепочка по типам — через think-подобный поиск.
  • Индукция по типу: «Lily is a swan. Lily is gray.» + «Greg is a swan» ⇒ Greg gray (свойства экземпляров
    того же типа голосуют; свойства — замкнутое множество значений, встречавшихся как «X is <adj>»).
  • Да/нет: вопрос «Is X in Y?» → ищем факт; есть точный → yes; есть другой (исключающее отношение) → no;
    есть «X is either Y or Z» (неопределённость) → maybe; нет ничего → unknown. Отрицание «X is no longer in Y» —
    снимает факт.
  • Путь: граф направлений + BFS, ответ — последовательность направлений.
Вход — простой английский (bAbI-стиль). Разбор правилами (регулярки по закрытому классу слов).
"""
import re, collections

DIRS = {"north": (0, 1), "south": (0, -1), "east": (1, 0), "west": (-1, 0)}
OPP = {"north": "south", "south": "north", "east": "west", "west": "east", "left": "right", "right": "left", "above": "below", "below": "above"}
PLURAL = {"wolves": "wolf", "mice": "mouse", "sheep": "sheep", "cats": "cat", "lions": "lion", "swans": "swan", "rhinos": "rhino", "frogs": "frog"}
def sing(w): return PLURAL.get(w, w[:-1] if w.endswith("s") and not w.endswith("ss") else w)
def obj(s):
    s = re.sub(r"^(?:the|a|an)\s+", "", s.strip().lower()); return re.sub(r"\s+", " ", s)

MOTIVE = collections.defaultdict(collections.Counter)     # ОБЩИЙ выученный опыт: состояние → куда идут / что берут (задача 20); растёт при чтении
MOTIVE_OBJ = collections.defaultdict(collections.Counter)

class World:
    def __init__(self):
        self.isa = {}                                        # экземпляр → тип
        self.cls_rel = collections.defaultdict(list)         # тип → [(rel, obj)]  «wolves are afraid of mice»
        self.inst_rel = collections.defaultdict(list)        # экземпляр → [(rel, obj)]
        self.prop = collections.defaultdict(set)             # экземпляр → {свойства}
        self.loc = {}; self.loc_hist = collections.defaultdict(list); self.maybe = {}
        self.order = collections.defaultdict(set)            # транзитивные: rel → {(a, b)}
        self.pos = {}                                        # комнаты: a → {dir: b}
        self.carry = collections.defaultdict(list); self._t = None
        self.last_actor = []                                 # для местоимений
        self.gave = []                                       # (giver, obj, receiver) в порядке событий (задача 5)
        self.obj_loc = {}; self.obj_hist = collections.defaultdict(list)   # предмет → комната (задачи 2, 3)
        self.step = 0
        self.state = {}                                      # человек → состояние (tired/hungry/thirsty/bored)
    # ---------- чтение ----------
    def read(self, sent):
        s = sent.strip().rstrip(".").lower()
        TIME = {"yesterday": 0, "this morning": 1, "this afternoon": 2, "this evening": 3}
        tm = next((v for k, v in TIME.items() if k in s), None)
        s = re.sub(r"\b(?:this morning|this afternoon|this evening|yesterday)\b\s*", "", s).strip()
        s = re.sub(r"^(?:then|after that|afterwards|following that|after this|later)\s+", "", s)
        self._t = tm
        # МЕСТОИМЕНИЯ: he/she → последний упомянутый человек; they → последняя пара (задачи 11, 13)
        if re.match(r"^(?:he|she) ", s) and self.last_actor: s = re.sub(r"^(?:he|she)\b", self.last_actor[0], s)
        elif re.match(r"^they ", s) and self.last_actor: s = re.sub(r"^they\b", " and ".join(self.last_actor), s)
        m = re.match(r"^(\w+) is (?:a|an) (\w+)$", s)
        if m: self.isa[m.group(1)] = sing(m.group(2)); return "isa"
        m = re.match(r"^(\w+) are afraid of (\w+)$", s)
        if m: self.cls_rel[sing(m.group(1))].append(("afraid_of", sing(m.group(2)))); return "cls"
        m = re.match(r"^(\w+) is afraid of (\w+)$", s)
        if m: self.inst_rel[m.group(1)].append(("afraid_of", sing(m.group(2)))); return "inst"
        m = re.match(r"^(\w+) is (\w+)$", s)
        if m and m.group(2) not in ("in", "no", "not", "either"):
            self.prop[m.group(1)].add(m.group(2)); self.state[m.group(1)] = m.group(2); self.last_actor = [m.group(1)]; return "prop"
        m = re.match(r"^(.+?) is (?:to the )?(left|right|above|below) (?:of )?(.+)$", s)
        if m: a, d, b = obj(m.group(1)), m.group(2), obj(m.group(3)); self.order[d].add((a, b)); self.order[OPP[d]].add((b, a)); return "order"
        m = re.match(r"^(.+?) is (north|south|east|west) of (.+)$", s)
        if m: a, d, b = obj(m.group(1)), m.group(2), obj(m.group(3)); self.pos.setdefault(b, {})[d] = a; self.pos.setdefault(a, {})[OPP[d]] = b; return "pos"
        m = re.match(r"^(.+?) (?:is bigger than|does not fit in|is larger than) (.+)$", s)
        if m: self.order["bigger"].add((obj(m.group(1)), obj(m.group(2)))); return "order"
        m = re.match(r"^(.+?) fits? inside (.+)$", s)
        if m: self.order["bigger"].add((obj(m.group(2)), obj(m.group(1)))); return "order"
        m = re.match(r"^(\w+) is either in the (\w+) or the (\w+)$", s)
        if m: self.maybe[m.group(1)] = {m.group(2), m.group(3)}; self.loc.pop(m.group(1), None); return "maybe"
        m = re.match(r"^(\w+) is (?:no longer|not) in the (\w+)$", s)
        if m:
            if self.loc.get(m.group(1)) == m.group(2): self.loc.pop(m.group(1), None)
            self.maybe.pop(m.group(1), None); self.inst_rel[m.group(1)].append(("not_in", m.group(2))); return "neg"
        self.step += 1
        m = re.match(r"^(\w+)(?: and (\w+))? (?:went|moved|journeyed|travelled|traveled|is|are|went back|got back|go|goes|came|ran|walked) (?:back )?(?:to|in|into) the (\w+)$", s)
        if m:
            actors = [p for p in (m.group(1), m.group(2)) if p]; self.last_actor = actors
            for p in actors:
                if p in self.state: MOTIVE[self.state[p]][m.group(3)] += 1          # ИНДУКЦИЯ МОТИВА: «tired» → «bedroom» (счётчик по опыту)
                self.loc[p] = m.group(3); self.loc_hist[p].append((self._t if self._t is not None else len(self.loc_hist[p]), m.group(3))); self.maybe.pop(p, None)
                for o in self.carry.get(p, []): self.obj_loc[o] = m.group(3); self.obj_hist[o].append((self.step, m.group(3)))   # предметы едут с человеком
            return "loc"
        m = re.match(r"^(\w+) (?:gave|handed|passed) the (\w+) to (\w+)$", s)
        if m:
            g, o, r = m.groups(); self.last_actor = [g]; self.gave.append((g, o, r))
            if o in self.carry[g]: self.carry[g].remove(o)
            self.carry[r].append(o)
            if r in self.loc: self.obj_loc[o] = self.loc[r]; self.obj_hist[o].append((self.step, self.loc[r]))
            return "give"
        m = re.match(r"^(\w+) (?:got|grabbed|took|picked up) the (\w+)", s)
        if m:
            p, o = m.groups(); self.last_actor = [p]; self.carry[p].append(o)
            if p in self.state: MOTIVE_OBJ[self.state[p]][o] += 1
            if p in self.loc: self.obj_loc[o] = self.loc[p]; self.obj_hist[o].append((self.step, self.loc[p]))
            return "carry"
        m = re.match(r"^(\w+) (?:dropped|discarded|put down|left|put) the (\w+)", s)
        if m:
            p, o = m.groups(); self.last_actor = [p]
            if o in self.carry[p]: self.carry[p].remove(o)
            if p in self.loc: self.obj_loc[o] = self.loc[p]; self.obj_hist[o].append((self.step, self.loc[p]))   # предмет остаётся в комнате
            return "drop"
        return None
    # ---------- вывод ----------
    def trans(self, rel, a, b, seen=None):
        """транзитивное замыкание: a rel ... rel b ?"""
        seen = seen or set()
        if (a, b) in self.order[rel]: return True
        for x, y in self.order[rel]:
            if x == a and y not in seen: 
                seen.add(y)
                if self.trans(rel, y, b, seen): return True
        return False
    def layout(self):
        """2D-раскладка объектов из отношений left/right/above/below (соединённые компоненты, BFS от первого)"""
        AX = {"left": (-1, 0), "right": (1, 0), "above": (0, 1), "below": (0, -1)}
        edges = collections.defaultdict(list)
        for d, pairs in self.order.items():
            if d in AX:
                for a, b in pairs: edges[a].append((b, (-AX[d][0], -AX[d][1]))); edges[b].append((a, AX[d]))
        coords = {}
        for start in edges:
            if start in coords: continue
            coords[start] = (0, 0); q = [start]
            while q:
                n = q.pop()
                for m, (dx, dy) in edges[n]:
                    if m not in coords: coords[m] = (coords[n][0] + dx, coords[n][1] + dy); q.append(m)
        return coords
    def afraid(self, x):
        rels = list(self.inst_rel.get(x, [])); t = self.isa.get(x)
        if t: rels += self.cls_rel.get(t, [])
        return [o for r, o in rels if r == "afraid_of"]
    def induce(self, x):
        """свойство экземпляра по другим экземплярам его типа (голосование)"""
        t = self.isa.get(x); votes = collections.Counter()
        for y, ty in self.isa.items():
            if ty == t and y != x:
                for p in self.prop.get(y, ()): votes[p] += 1
        return votes.most_common(1)[0][0] if votes else None
    def path(self, a, b):
        from collections import deque
        q = deque([(a, [])]); seen = {a}
        while q:
            n, p = q.popleft()
            if n == b: return p
            for d, m in self.pos.get(n, {}).items():
                if m not in seen: seen.add(m); q.append((m, p + [d]))
        return None
    def answer(self, q):
        s = q.strip().rstrip("?").lower()
        m = re.match(r"^what is (\w+) afraid of$", s)
        if m: a = self.afraid(m.group(1)); return (a[0], ["class rule → instance"]) if a else (None, [])
        m = re.match(r"^what color is (\w+)$", s)
        if m: p = self.prop.get(m.group(1)); return (next(iter(p)), ["direct"]) if p else (self.induce(m.group(1)), ["induction by type"])
        m = re.match(r"^is (.+?) (?:to the )?(left|right|above|below) (?:of )?(.+)$", s)
        if m:
            a, d, b = obj(m.group(1)), m.group(2), obj(m.group(3)); coords = self.layout()
            if a in coords and b in coords:
                dx, dy = coords[a][0] - coords[b][0], coords[a][1] - coords[b][1]
                ok = {"left": dx < 0, "right": dx > 0, "above": dy > 0, "below": dy < 0}[d]
                return ("yes" if ok else "no", ["2D layout from relations"])
            return ("yes" if self.trans(d, a, b) else "no", ["transitive " + d])
        m = re.match(r"^(?:is|does) (.+?) (?:bigger than|larger than|fit in|fit inside) (.+)$", s)
        if m:
            a, b = obj(m.group(1)), obj(m.group(2))
            if "fit" in s: return ("yes" if self.trans("bigger", b, a) else "no", ["transitive bigger"])
            return ("yes" if self.trans("bigger", a, b) else "no", ["transitive bigger"])
        m = re.match(r"^how do you go from the (\w+) to the (\w+)$", s)
        if m: p = self.path(m.group(1), m.group(2)); return (" ".join(p) if p else None, ["BFS on map"])
        m = re.match(r"^is (\w+) in the (\w+)$", s)
        if m:
            p, r = m.group(1), m.group(2)
            if self.loc.get(p) == r: return ("yes", ["state: location"])
            if p in self.maybe: return ("maybe" if r in self.maybe[p] else "no", ["indefinite"])
            if p in self.loc: return ("no", ["state: elsewhere"])
            if ("not_in", r) in self.inst_rel.get(p, []): return ("no", ["negation"])
            return (None, [])
        m = re.match(r"^where is (?:the )?(\w+)$", s)
        if m:
            x = m.group(1)
            if x in self.loc: return (self.loc[x], ["state"])
            if x in self.obj_loc: return (self.obj_loc[x], ["object follows carrier / left in room"])
            for p, objs in self.carry.items():                                   # предмет у кого-то в руках, комната носителя
                if x in objs and p in self.loc: return (self.loc[p], ["carried by " + p])
            return (None, [])
        m = re.match(r"^where was the (\w+) before the (\w+)$", s)
        if m:
            h = [r for _, r in sorted(self.obj_hist.get(m.group(1), []))]
            rooms = [r for i, r in enumerate(h) if i == 0 or h[i - 1] != r]      # без повторов подряд
            if m.group(2) in rooms:
                i = len(rooms) - 1 - rooms[::-1].index(m.group(2)); return (rooms[i - 1] if i > 0 else None, ["object timeline"])
            return (None, [])
        m = re.match(r"^what is (?:the )?(\w+) (north|south|east|west) of$", s)                 # «What is the bathroom east of?» → bathroom is east of X
        if m:
            a, d = m.group(1), m.group(2)
            for b, dirs in self.pos.items():
                if dirs.get(d) == a: return (b, ["map: " + a + " is " + d + " of " + b])
            return (None, [])
        m = re.match(r"^what is (north|south|east|west) of the (\w+)$", s)                    # «What is west of the kitchen?»
        if m: return (self.pos.get(m.group(2), {}).get(m.group(1)), ["map"])
        m = re.match(r"^what did (\w+) give to (\w+)$", s)
        if m:
            g, r = m.groups(); ev = [o for gg, o, rr in self.gave if gg == g and rr == r]; return (ev[-1] if ev else None, ["last give event"])
        m = re.match(r"^who gave the (\w+) to (\w+)$", s)
        if m:
            o, r = m.groups(); ev = [gg for gg, oo, rr in self.gave if oo == o and rr == r]; return (ev[-1] if ev else None, ["last give event"])
        m = re.match(r"^who did (\w+) give the (\w+) to$", s)
        if m:
            g, o = m.groups(); ev = [rr for gg, oo, rr in self.gave if gg == g and oo == o]; return (ev[-1] if ev else None, ["last give event"])
        m = re.match(r"^who received the (\w+)$", s)
        if m: ev = [rr for gg, oo, rr in self.gave if oo == m.group(1)]; return (ev[-1] if ev else None, ["last give event"])
        m = re.match(r"^where will (\w+) go$", s)
        if m:
            st = self.state.get(m.group(1))
            if st and MOTIVE.get(st): return (MOTIVE[st].most_common(1)[0][0], [f"learned motive: {st} → {MOTIVE[st].most_common(1)[0][0]} ({MOTIVE[st].most_common(1)[0][1]} cases)"])
            return (None, [])
        m = re.match(r"^why did (\w+) (?:go|get|grab|pick up|take) (?:back )?(?:to )?the (\w+)$", s)
        if m:
            st = self.state.get(m.group(1)); return (st, ["state → action"]) if st else (None, [])
        m = re.match(r"^what will (\w+) (?:get|grab|pick up|take)$", s)
        if m:
            st = self.state.get(m.group(1))
            if st and MOTIVE_OBJ.get(st): return (MOTIVE_OBJ[st].most_common(1)[0][0], ["learned motive (object)"])
            return (None, [])
        m = re.match(r"^who gave the (\w+)$", s)
        if m: ev = [gg for gg, oo, rr in self.gave if oo == m.group(1)]; return (ev[-1] if ev else None, ["last give event"])
        m = re.match(r"^who received the (\w+)$", s)
        if m: ev = [rr for gg, oo, rr in self.gave if oo == m.group(1)]; return (ev[-1] if ev else None, ["last give event"])
        m = re.match(r"^how many objects is (\w+) carrying$", s)
        if m: n = len(self.carry.get(m.group(1), [])); return (["none", "one", "two", "three", "four", "five"][n] if n < 6 else str(n), ["count carry state"])
        m = re.match(r"^what is (\w+) carrying$", s)
        if m: c = self.carry.get(m.group(1), []); return (",".join(c) if c else "nothing", ["carry state"])
        m = re.match(r"^where was (\w+) before the (\w+)$", s)
        if m:
            h = sorted(self.loc_hist.get(m.group(1), [])); rooms = [r for _, r in h]
            if m.group(2) in rooms:
                i = len(rooms) - 1 - rooms[::-1].index(m.group(2))                        # ПОСЛЕДНЕЕ посещение этой комнаты
                return (rooms[i - 1] if i > 0 else None, ["timeline"])
            return (None, [])
        m = re.match(r"^what is (\w+) carrying$", s)
        if m: c = self.carry.get(m.group(1), []); return (",".join(c) if c else "nothing", ["carry state"])
        return (None, [])
