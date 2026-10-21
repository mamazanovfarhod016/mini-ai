"""growth.py — РОСТ МЫШЛЕНИЯ без градиента: ядро учится не только фактам, но и тому, КАК думать.

Что растёт само (сохраняется в state_dir/growth.json, переживает перезапуск):
  1. Схемы вывода (fewshot.rules): из каждого исправления «нет, ответ — X» и каждого подтверждённого ответа
     извлекается путь по связям вопрос→ответ и привязывается к ФОРМЕ вопроса (без имён). Новая форма вопроса,
     однажды понятая, дальше решается сама. Чем больше разговоров — тем больше форм.
  2. Надёжность способов (how × форма вопроса → сколько раз сработал / промахнулся): «1 шаг» на форме «what do _
     use to make _» ошибался 3 раза, «отрывок» попадал — арбитр будет верить отрывку. Без этого все способы
     равны и выбирается первый сработавший.
  3. Связки слов (co-answer): пара (слово вопроса → слово ответа), встречавшаяся в подтверждённых ответах
     («viscosity» → «viscous», «humidity» → «water vapor»). Следующий вопрос с этим словом тянет связку как
     подсказку при поиске.
Ничего не обучается градиентом; всё — счётчики и пути, которые можно прочитать и объяснить.
"""
import os, re, json, collections, time
from mini_ai.memory_v2 import toks, content_words


def qform(q):
    """форма вопроса без имён и чисел: «What do most living things use to make ATP from glucose?» →
    «what do _ use to make _ from _» — редкие слова (термины) заменяются «_», служебные остаются"""
    s = q.strip().rstrip("?")
    s = s[:1].lower() + s[1:]                                                         # первое слово — не имя
    s = re.sub(r"\b[A-Z][\w\-]+(?: [A-Z][\w\-]+)*\b", "_", s)                         # имена
    s = re.sub(r"\d+(?:\.\d+)?", "_", s).lower()
    keep = set("what who where when which how why is are was were do does did can could the a an of in on at to for from by with and or "
               "many much kind type term called known name used use make made part most all some only not no".split())
    ws = [w if (w in keep or w == "_") else "_" for w in re.findall(r"[a-z_']+", s)]
    out = []
    for w in ws:
        if w == "_" and out and out[-1] == "_": continue
        out.append(w)
    return " ".join(out)


def qcoarse(q):
    """грубая форма: тип вопроса, чтобы опыт переносился между похожими вопросами —
    «what term _», «what kind of _», «_ called what», «what do _ use», «which type of _», «what is the _ of _»"""
    f = qform(q).split()
    if not f: return "?"
    if f[-1] == "what" and "called" in f: return "_ called what"
    if "called" in f: return "what _ called"
    head = f[:3] if f[0] in ("what", "which", "who", "where", "when", "how", "why", "in", "on", "the", "_") else f[:2]
    return " ".join(head)


def qclass(q):
    """самый грубый класс вопроса — для ПЕРЕНОСА опыта между типами: «definition» (what is/term/called/known as),
    «which-of» (which type/kind/what group), «how-many», «where», «who», «why», «process» (what happens/what do _ use),
    «fill» (вопрос-утверждение с пропуском: «_ are called?», «X is made of what?»)"""
    ql = q.strip().rstrip("?").lower()
    if re.search(r"\bhow (?:many|much)\b", ql): return "how-many"
    if re.match(r"^(?:where|in which (?:country|city|continent|place|organ|part))\b", ql): return "where"
    if re.match(r"^(?:who|whose)\b", ql): return "who"
    if re.match(r"^why\b", ql): return "why"
    if re.search(r"\b(?:called|known as|term(?:ed)?|name(?:d)? (?:for|of)|refers? to|is the (?:name|term))\b", ql): return "definition"
    if re.match(r"^(?:what|which) (?:type|kind|sort|class|group|category|form)s? of\b", ql): return "which-of"
    if re.match(r"^what (?:happens|occurs|is produced|is formed|do(?:es)? .+ (?:use|produce|make|form|cause|release))\b", ql): return "process"
    if not re.match(r"^(?:what|which|who|where|when|how|why|in|on|at|by|during|of)\b", ql): return "fill"
    return "what"


class Growth:
    def __init__(self, mind, state_dir=None):
        self.mind = mind; self.path = os.path.join(state_dir, "growth.json") if state_dir else None
        self.reliab = collections.defaultdict(lambda: [0, 0])     # (how, form) → [hits, misses]
        self.coans = collections.defaultdict(collections.Counter) # слово вопроса → Counter(слово ответа)
        self.log = []                                             # последние события роста (для объяснения)
        self.rules_saved = {}                                     # форма → [схемы] (копия fewshot.rules для сохранения)
        self.slot_rules = collections.defaultdict(collections.Counter)  # грубая форма → Counter((slot, rel, qwhere)) — для вопросов без имён
        self.attn = collections.defaultdict(lambda: [0, 0])       # слово вопроса → [нужное, лишнее] (выученное внимание)
        self._soft_list = None; self._soft_stamp = -1; self._soft_cache = {}
        self.pos = collections.defaultdict(collections.Counter)   # класс вопроса → Counter(позиция ответа в предложении)
        self.board_stats = collections.defaultdict(lambda: collections.Counter())   # «способ|класс» → place/answer/miss (учёба у проигравших)
        self.para = collections.defaultdict(collections.Counter)                    # слово вопроса → Counter(слово текста) — выученный перефраз
        self.diag = collections.Counter()                         # ДИАГНОЗ ошибок: почему промахнулась (silent/wrong-piece/wrong-fact/no-sentence)
        self.insights = []                                        # ОБОБЩЕНИЯ: правила, подтверждённые на многих типах → принцип
        self.load()

    # ---------- хранение ----------
    def load(self):
        if not self.path or not os.path.exists(self.path): return
        try: d = json.load(open(self.path))
        except Exception: return
        for k, v in d.get("reliab", {}).items(): self.reliab[k] = v
        for k, v in d.get("coans", {}).items(): self.coans[k] = collections.Counter(v)
        self.rules_saved = d.get("rules", {}); self.log = d.get("log", [])[-50:]
        for k, v in d.get("attn", {}).items(): self.attn[k] = v
        self.diag = collections.Counter(d.get("diag", {})); self.insights = d.get("insights", [])
        for k, v in d.get("pos", {}).items(): self.pos[k] = collections.Counter(v)
        for k, v in d.get("board_stats", {}).items(): self.board_stats[k] = collections.Counter(v)
        for k, v in d.get("para", {}).items(): self.para[k] = collections.Counter(v)
        for k, v in d.get("slot_rules", {}).items(): self.slot_rules[k] = collections.Counter({tuple(kk.split("|")): n for kk, n in v.items()})
        fs = getattr(self.mind, "fewshot", None)
        if fs:
            for k, schemas in self.rules_saved.items(): fs.rules[k] = {tuple(s) for s in schemas}
            for k, ex in d.get("examples", {}).items(): fs.examples[k] = [tuple(e) for e in ex]

    def save(self):
        if not self.path: return
        fs = getattr(self.mind, "fewshot", None)
        rules = {k: [list(s) for s in v] for k, v in fs.rules.items()} if fs else self.rules_saved
        examples = {k: [list(e) for e in v] for k, v in fs.examples.items()} if fs else {}
        json.dump(dict(reliab=dict(self.reliab), coans={k: dict(v) for k, v in self.coans.items()}, rules=rules, examples=examples,
                       slot_rules={k: {"|".join(kk): n for kk, n in v.items()} for k, v in self.slot_rules.items()}, attn=dict(self.attn), diag=dict(self.diag), insights=self.insights, pos={k: dict(v) for k, v in self.pos.items()}, board_stats={k: dict(v) for k, v in self.board_stats.items()}, para={k: dict(v) for k, v in self.para.items()},
                       log=self.log[-50:], when=time.strftime("%Y-%m-%d %H:%M")), open(self.path, "w"), ensure_ascii=False, indent=1)

    # ---------- 2. надёжность способов ----------
    @staticmethod
    def _howkey(how):
        return re.sub(r"[«»].*", "", (how or "").split(":")[0]).replace(" +читатель", "").strip() or "?"

    def trust(self, how, q):
        """множитель уверенности по опыту, С ПЕРЕНОСОМ: точная форма → тип → класс → вообще. Уровни смешиваются:
        точный опыт весит больше, но если его мало — тянет опыт класса («все вопросы-определения»). нет опыта → 1.0"""
        hk = self._howkey(how); h = m = 0.0
        for key, w in ((f"{hk}|{qform(q)}", 1.0), (f"{hk}|{qcoarse(q)}", 0.5), (f"{hk}|{qclass(q)}", 0.25), (f"{hk}|*", 0.1)):
            hh, mm = self.reliab.get(key, [0, 0]); h += w * hh; m += w * mm
        if h + m < 3.0: return 1.0                                   # 1–2 случая — не опыт, а шум
        # относительно базовой точности способа вообще: доверие = насколько ЗДЕСЬ он лучше/хуже, чем обычно (±25 %)
        H, Mm = self.reliab.get(f"{hk}|*", [0, 0]); base = (H + 1) / (H + Mm + 2); here = (h + 2 * base) / (h + m + 2)
        return round(max(0.75, min(1.25, 1.0 + (here - base) * 1.5)), 2)

    def boldness(self, q):
        """0..1 — насколько стоит ДОГАДЫВАТЬСЯ (отрывком) вместо «не знаю» на таком типе вопроса.
        Растёт, когда отрывок попадал, и когда «не знаю» оказывалось ошибкой (ответ был, а она молчала)."""
        def rate(hk, default):
            h = m = 0.0
            for key, w in ((f"{hk}|{qcoarse(q)}", 1.0), (f"{hk}|{qclass(q)}", 0.5), (f"{hk}|*", 0.2)):
                hh, mm = self.reliab.get(key, [0, 0]); h += w * hh; m += w * mm
            return (h + 1) / (h + m + 2) if h + m >= 2 else default
        span_ok = rate("отрывок предложения", 0.5)
        silent_wrong = 1 - rate("не знаю → уточнение", 0.5) if any(k.startswith("не знаю") for k in self.reliab) else 0.5
        return round(max(0.0, min(1.0, 0.6 * span_ok + 0.4 * silent_wrong)), 2)

    def _mark(self, how, q, hit):
        hk = self._howkey(how)
        for key in (f"{hk}|{qform(q)}", f"{hk}|{qcoarse(q)}", f"{hk}|{qclass(q)}", f"{hk}|*"):
            self.reliab[key][0 if hit else 1] += 1

    # ---------- 1в. ВЫУЧЕННОЕ ВНИМАНИЕ: какие слова вопроса оказались лишними ----------
    def _learn_attention(self, q, right):
        """исправление: находим предложение памяти с правильным ответом; слова вопроса, которых там нет, —
        были ЛИШНИМИ (уточнение, тип ответа, другой глагол). Счётчик «слово → лишнее» растёт; слово с опытом
        «лишнее ≥2 раза, нужное ≤ лишнее» дальше считается мягким для всех вопросов. Это attention, выученный
        из исправлений, а не заданный таблицей."""
        kb = self.mind.kb; rw = set(content_words(toks(right)))
        if not rw: return []
        qw = {w for w in content_words(toks(q)) if w not in self.mind.META}
        best = None
        for si, (sent, src) in enumerate(kb.sents):
            sl = sent.lower()
            if not all(w in sl for w in rw): continue
            sw = set(content_words(toks(sent))); sf = set().union(*[self.mind._forms(x) for x in sw]) if sw else set()
            cov = sum(1 for w in qw if self.mind._forms(w) & sf)
            if not best or cov > best[0]: best = (cov, sf, sent)
        if not best or best[0] == 0: return []
        extra = [w for w in qw if not (self.mind._forms(w) & best[1])]
        needed = [w for w in qw if self.mind._forms(w) & best[1]]
        for w in extra: self.attn[w][1] += 1
        for w in needed: self.attn[w][0] += 1
        return extra

    def is_soft_learned(self, w):
        n, x = self.attn.get(w, (0, 0))
        if x >= 3 and x >= 2 * n and self.mind.kb.idf(w) < 3.5: return True       # лишнее стабильно, и это не редкий термин
        # ПЕРЕНОС ПО СМЫСЛУ: слово без опыта, но по смыслу близко к выученному лишнему («objects»→«things», «organ»→«organs/tissue»)
        if n + x == 0 and len(w) > 3:
            M = getattr(self.mind, "meaning", None)
            if M is not None and self._soft_words:
                cache = self._soft_cache
                if w in cache: return cache[w]
                r = any((M.similar(w, sw) or 0) > 0.45 for sw in self._soft_words)
                if len(cache) < 20000: cache[w] = r
                return r
        return False

    @property
    def _soft_words(self):
        if self._soft_stamp != len(self.attn) or self._soft_list is None:
            self._soft_list = [w for w, (n, x) in self.attn.items() if x >= 3 and x >= 2 * n and self.mind.kb.idf(w) < 3.5][:300]; self._soft_stamp = len(self.attn); self._soft_cache = {}
        return self._soft_list

    # ---------- 1г. ПОЗИЦИЯ ОТВЕТА в предложении (выучивается): начало / после «called» / после «is a» / конец / середина ----------
    @staticmethod
    def _pos_of(sent, answer):
        sl = sent.lower(); i = sl.find(answer.lower())
        if i < 0: return None
        before = toks(sl[:i]); after = toks(sl[i + len(answer):])
        if len(before) <= 1: return "start"
        if before[-1] in ("called", "termed", "named") or (len(before) >= 2 and before[-2:] == ["known", "as"]): return "after-called"
        if before[-1] in ("a", "an", "the") and len(before) >= 2 and before[-2] in ("is", "are", "was", "were"): return "after-is-a"
        if not after or (len(after) <= 2 and after[0] in ("which", "that", "figure")): return "end"
        return "middle"

    def learn_position(self, q, right):
        kb = self.mind.kb; rl = right.lower()
        for sent, _ in reversed(kb.sents[-2000:]):
            if rl in sent.lower():
                p = self._pos_of(sent, right)
                if p: self.pos[qclass(q)][p] += 1; self.pos["*"][p] += 1
                return p
        return None

    def pos_prior(self, q, sent, piece):
        """множитель к куску по выученной позиции: где в предложении обычно стоит ответ на такой класс вопроса"""
        c = self.pos.get(qclass(q)) or self.pos.get("*") or {}
        tot = sum(c.values())
        if tot < 10: return 1.0
        p = self._pos_of(sent, piece) or "middle"
        share = c.get(p, 0) / tot
        return 0.7 + 0.9 * share            # позиция, где ответ бывает в 50 % случаев → ×1.15; никогда → ×0.7

    # ---------- 1б. схемы для вопросов без имён: ГДЕ в факте лежит ответ ----------
    def _slot_pattern(self, q, right):
        """из исправления: находим факт с ответом; запоминаем (грубая форма → слот факта, связь, где слова вопроса):
        «_ called what» → ответ = Y факта be_called, слова вопроса в X. Это правило: как читать такие вопросы."""
        kb = self.mind.kb; rw = set(content_words(toks(right))); qw = set(content_words(toks(q))) - self.mind.META
        if not rw: return None
        best = None
        for w in rw:
            for k in kb.by_word.get(w, [])[:300]:
                Xw, rel, Yw, X, Y, src = kb.facts[k]
                for slot, sw, ow in (("X", Xw, Yw), ("Y", Yw, Xw)):
                    if not rw <= sw: continue
                    score = len(qw & ow) * 2 + len(qw & sw) - len(sw - rw) * 0.3
                    if not best or score > best[0]: best = (score, slot, rel, "other" if qw & ow else "same")
        if not best or best[0] <= 0: return None
        return dict(slot=best[1], rel=best[2], qwhere=best[3])

    def slot_prefs(self, q):
        """выученные предпочтения «где ответ» С ПЕРЕНОСОМ: тип вопроса (вес 1) + класс (вес 0.4)"""
        c = collections.Counter()
        for k, n in (self.slot_rules.get(qcoarse(q)) or {}).items(): c[k] += n
        for k, n in (self.slot_rules.get("class:" + qclass(q)) or {}).items(): c[k] += 0.4 * n
        return c

    # ---------- 3. связки слов ----------
    def _link(self, q, answer):
        qw = [w for w in content_words(toks(q)) if w not in self.mind.META]; aw = content_words(toks(answer))
        for a in aw[:3]:
            for w in qw: self.coans[w][a] += 1
        return len(qw) * len(aw[:3])

    def hints(self, q):
        """слова-подсказки для поиска: что обычно отвечали на такие слова вопроса"""
        c = collections.Counter()
        for w in content_words(toks(q)):
            for a, n in self.coans.get(w, {}).items(): c[a] += n
        return [a for a, n in c.most_common(5) if n >= 2]

    # ---------- 1д. ПЕРЕФРАЗ: слово вопроса ↔ слово предложения-ответа (выучивается из исправлений) ----------
    def _learn_paraphrase(self, q, right):
        """исправление: предложение с ответом найдено; слова вопроса, которых там нет (лишние), и слова предложения,
        которых нет в вопросе (кроме ответа) — пары-кандидаты «так спрашивают ↔ так написано»:
        holds ↔ held, make ↔ produce, ability ↔ able, method ↔ by diffusion. Пара, встреченная ≥2 раз, становится
        словарём перефраза: при поиске слово вопроса расширяется его парами (как формы слова)."""
        kb = self.mind.kb; rl = right.lower()
        sent = next((s_ for s_, _ in reversed(kb.sents[-3000:]) if rl in s_.lower()), None)
        if not sent: return []
        qw = {w for w in content_words(toks(q)) if w not in self.mind.META and len(w) > 3}
        sw = {w for w in content_words(toks(sent)) if len(w) > 3} - set(content_words(toks(right)))
        forms = lambda w: self.mind._forms(w)
        q_only = [w for w in qw if not any(forms(w) & forms(x) for x in sw)]
        s_only = [w for w in sw if not any(forms(w) & forms(x) for x in qw)]
        if not q_only or not s_only or len(q_only) > 3 or len(s_only) > 6: return []
        M = getattr(self.mind, "meaning", None); pairs = []
        for a in q_only:
            best = None
            for b in s_only:
                sim = (M.similar(a, b) or 0) if M is not None else 0
                if sim > 0.25 and (not best or sim > best[0]): best = (sim, b)
            if best: self.para[a][best[1]] += 1; pairs.append((a, best[1]))
        return pairs

    def paraphrases(self, w):
        """выученные замены слова вопроса (≥2 подтверждения)"""
        return [b for b, n in self.para.get(w, {}).items() if n >= 2]

    # ---------- 1. схемы вывода из исправлений ----------
    def learn_from_correction(self, q, out, right):
        """пользователь поправил: «no, it's RIGHT». Запоминаем: (а) способ ошибся, (б) путь вопрос→RIGHT — схема,
        (в) связка слов. Возвращает человеку объяснение, что именно выросло."""
        grew = []
        if out and out.get("how"): self._mark(out["how"], q, False); grew.append(f"{self._howkey(out['how'])} is less trusted for «{qform(q)}»")
        fs = getattr(self.mind, "fewshot", None)
        if fs:
            r = fs.show(q, right)
            if r: grew.append(f"rule for «{qform(q)}»: {'→'.join(r[0]).replace('inv:', 'reverse ')}")
        sp = self._slot_pattern(q, right)
        if sp:
            self.slot_rules[qcoarse(q)][(sp["slot"], sp["rel"], sp["qwhere"])] += 1; self.slot_rules["class:" + qclass(q)][(sp["slot"], sp["rel"], sp["qwhere"])] += 1
            grew.append(f"for «{qcoarse(q)}» the answer sits in {sp['slot']} of a «{sp['rel']}» fact")
        extra = self._learn_attention(q, right)
        if extra: grew.append("extra words: " + ", ".join(extra[:4]))
        pp = self.learn_position(q, right)
        if pp: grew.append(f"answer position: {pp}")
        grew += self.learn_from_board(q, out, right)
        pr = self._learn_paraphrase(q, right)
        if pr: grew.append("paraphrase: " + ", ".join(f"{a}↔{b}" for a, b in pr))
        # ДИАГНОЗ: почему ошиблась? молчала (ответ был в памяти) / взяла не тот кусок предложения / не тот факт / ответа нет в памяти
        rl = right.lower(); in_mem = any(rl in s_.lower() for s_, _ in self.mind.kb.sents[-3000:])
        how = self._howkey(out.get("how") if out else "")
        if not in_mem: d = "no-sentence"
        elif how.startswith("не знаю"): d = "silent"
        elif how.startswith("отрывок"): d = "wrong-piece"
        else: d = "wrong-fact"
        self.diag[d] += 1; self.diag[f"{d}|{qclass(q)}"] += 1; grew.append(f"diagnosis: {d}")
        self._maybe_insight()
        if self._link(q, right): grew.append("word link " + ", ".join(f"{w}→{right.split()[0]}" for w in content_words(toks(q))[:2]))
        self.log.append(dict(t=time.strftime("%m-%d %H:%M"), kind="correction", q=q, right=right, grew=grew)); self.save()
        return grew

    def learn_from_confirmation(self, q, out):
        """пользователь подтвердил («yes», «right») или не поправил и пошёл дальше → способ надёжнее, связка слов сильнее"""
        if not out or not out.get("answer"): return []
        self._mark(out.get("how"), q, True); self._link(q, str(out["answer"]))
        self._learn_attention(q, str(out["answer"])); self.learn_position(q, str(out["answer"]))
        sp = self._slot_pattern(q, str(out["answer"]))
        if sp: self.slot_rules[qcoarse(q)][(sp["slot"], sp["rel"], sp["qwhere"])] += 1; self.slot_rules["class:" + qclass(q)][(sp["slot"], sp["rel"], sp["qwhere"])] += 1
        fs = getattr(self.mind, "fewshot", None)
        if fs and out.get("how", "").startswith(("2 шага", "few-shot")) is False and qform(q) not in fs.rules:
            fs.show(q, str(out["answer"]))                           # удачный ответ тоже становится схемой
        self.log.append(dict(t=time.strftime("%m-%d %H:%M"), kind="confirm", q=q, a=str(out["answer"]), how=out.get("how"))); self.save()
        return ["trust +1 for " + self._howkey(out.get("how"))]

    def diag_rates(self, q):
        """доли причин ошибок на классе вопроса: silent / wrong-piece / wrong-fact / no-sentence (с переносом на общий)"""
        cls = qclass(q); out = {}
        for d in ("silent", "wrong-piece", "wrong-fact", "no-sentence"):
            a = self.diag.get(f"{d}|{cls}", 0); b = self.diag.get(d, 0)
            out[d] = a + 0.3 * b
        tot = sum(out.values()) or 1.0
        return {k: v / tot for k, v in out.items()} if sum(self.diag.values()) >= 10 else {}

    # ---------- ОБОБЩЕНИЕ: из многих частных правил — принцип ----------
    def _maybe_insight(self):
        """раз в 50 исправлений: если одно и то же «где ответ» держится в ≥4 типах вопросов одного класса — это принцип класса,
        дальше применяется ко ВСЕМ вопросам класса, даже к типам без опыта. Если способ промахивается в ≥70 % по классу —
        принцип «не верить способу на классе». Записываем человеческим языком."""
        n = sum(self.diag.values())
        if n % 50: return
        new = []
        by_class = collections.defaultdict(lambda: collections.defaultdict(set))
        for form, cnt in self.slot_rules.items():
            if form.startswith("class:") or not cnt: continue
            top, cn = cnt.most_common(1)[0]
            if cn >= 2: by_class[qclass(form.replace("_", "x") + "?")][top].add(form)
        for cls, d in by_class.items():
            for top, forms in d.items():
                if len(forms) >= 4:
                    key = f"principle|{cls}|{'|'.join(top)}"
                    if key not in {i["key"] for i in self.insights}:
                        self.insights.append(dict(key=key, text=f"for {cls} questions the answer is usually {top[0]} of a «{top[1]}» fact ({len(forms)} question types agree)"))
                        self.slot_rules["class:" + cls][top] += 3; new.append(key)
        for key, (h, m) in list(self.reliab.items()):
            hk, form = key.split("|", 1)
            if form in ("definition", "which-of", "how-many", "where", "who", "why", "process", "fill", "what") and h + m >= 20 and m / (h + m) >= 0.7:
                k2 = f"distrust|{hk}|{form}"
                if k2 not in {i["key"] for i in self.insights}: self.insights.append(dict(key=k2, text=f"«{hk}» fails {m}/{h+m} on {form} questions — stop trusting it there")); new.append(k2)
        return new

    # ---------- УЧИТЬСЯ У ПРОИГРАВШИХ: что у каждого способа было верно, даже если ответ неверный ----------
    def learn_from_board(self, q, out, right):
        """исправление + доска арбитра (все кандидаты). Для каждого способа смотрим, что он сделал ВЕРНО:
          • «нашёл место»: его факт/предложение содержит правильный ответ (взял не тот кусок / не ту сторону);
          • «нашёл ответ»: его ответ и был верным, но арбитр выбрал другого (доверие/самопроверка подвели).
        Накапливаем по классу вопроса: (способ → место-верно / ответ-верно / мимо). Из этого — правило-комбинация:
        «на классе C способ A находит место (≥60 %), а не ответ → брать его предложение и доставать ответ как отрывок»."""
        board = (out or {}).get("arbiter") or []
        if not board: return []
        rl = right.lower(); cls = qclass(q); grew = []
        for row in board:
            how, ans, conf, trust, score, check, fact = (list(row) + [""] * 7)[:7]
            hk = self._howkey(how); key = f"{hk}|{cls}"
            if rl in str(ans).lower(): self.board_stats[key]["answer"] += 1; grew.append(f"{hk} had the right answer but lost the vote")
            elif rl in str(fact).lower(): self.board_stats[key]["place"] += 1; grew.append(f"{hk} found the right sentence but took the wrong piece")
            else: self.board_stats[key]["miss"] += 1
        # правило-комбинация, если накопилось: способ на классе находит место чаще, чем ответ
        for key, st in self.board_stats.items():
            hk, c = key.split("|", 1); tot = sum(st.values())
            if c == cls and tot >= 8 and st["place"] >= 0.5 * tot and st["place"] > st["answer"]:
                k2 = f"combine|{hk}|{c}"
                if k2 not in {i["key"] for i in self.insights}:
                    self.insights.append(dict(key=k2, text=f"on {c} questions «{hk}» finds the right sentence {st['place']}/{tot} but not the answer — reuse its sentence, extract the answer as a piece"))
                    grew.append("new rule: " + self.insights[-1]["text"])
            if c == cls and tot >= 8 and st["answer"] >= 0.4 * tot:
                k2 = f"undervoted|{hk}|{c}"
                if k2 not in {i["key"] for i in self.insights}:
                    self.insights.append(dict(key=k2, text=f"on {c} questions «{hk}» is right {st['answer']}/{tot} times when it loses the vote — trust it more there"))
                    self.reliab[f"{hk}|{c}"][0] += 5; grew.append("new rule: " + self.insights[-1]["text"])
        return grew

    def place_finders(self, q):
        """способы, которые на этом классе вопроса находят ПРАВИЛЬНОЕ ПРЕДЛОЖЕНИЕ (но не ответ): их факты — источник для отрывка"""
        cls = qclass(q); out = []
        for key, st in self.board_stats.items():
            hk, c = key.split("|", 1); tot = sum(st.values())
            if c == cls and tot >= 8 and st["place"] >= 0.5 * tot: out.append(hk)
        return out

    # ---------- отчёт ----------
    def report(self):
        fs = getattr(self.mind, "fewshot", None)
        n_rules = len(fs.rules) if fs else len(self.rules_saved)
        forms = sorted(((k, v) for k, v in self.reliab.items() if not k.endswith("|*")), key=lambda kv: -(kv[1][0] + kv[1][1]))[:8]
        soft = [w for w in self.attn if self.is_soft_learned(w)]
        npara = sum(1 for w in self.para for b, n in self.para[w].items() if n >= 2)
        lines = [f"rules learned: {n_rules} (named) + {sum(len(v) for v in self.slot_rules.values())} (where-in-fact, {len(self.slot_rules)} question types) + {npara} paraphrase pairs",
                 f"attention learned: {len(soft)} words found unimportant in questions: {', '.join(sorted(soft, key=lambda w: -self.attn[w][1])[:12])}",
                 f"question forms with experience: {len(forms)}",
                 f"word links: {sum(len(v) for v in self.coans.values())}"]
        if self.pos.get("*"):
            tot = sum(self.pos["*"].values()); lines.append("where answers sit: " + ", ".join(f"{k} {100*v//tot}%" for k, v in self.pos["*"].most_common()))
        if self.diag:
            tot = sum(v for k, v in self.diag.items() if "|" not in k) or 1
            lines.append("why I was wrong: " + ", ".join(f"{k} {v} ({100*v//tot}%)" for k, v in self.diag.most_common() if "|" not in k))
        for i in self.insights[-5:]: lines.append("  principle: " + i["text"])
        for k, (h, m) in forms: lines.append(f"  {k}: {h}✓ {m}✗ → trust {round(0.4 + 0.9 * (h + 1) / (h + m + 2), 2)}")
        for e in self.log[-5:]: lines.append(f"  [{e['t']}] {e['kind']}: {e.get('q', '')[:50]} → {e.get('right') or e.get('a', '')}")
        return "\n".join(lines)
