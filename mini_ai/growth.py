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


class Growth:
    def __init__(self, mind, state_dir=None):
        self.mind = mind; self.path = os.path.join(state_dir, "growth.json") if state_dir else None
        self.reliab = collections.defaultdict(lambda: [0, 0])     # (how, form) → [hits, misses]
        self.coans = collections.defaultdict(collections.Counter) # слово вопроса → Counter(слово ответа)
        self.log = []                                             # последние события роста (для объяснения)
        self.rules_saved = {}                                     # форма → [схемы] (копия fewshot.rules для сохранения)
        self.slot_rules = collections.defaultdict(collections.Counter)  # грубая форма → Counter((slot, rel, qwhere)) — для вопросов без имён
        self.attn = collections.defaultdict(lambda: [0, 0])       # слово вопроса → [нужное, лишнее] (выученное внимание)
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
                       slot_rules={k: {"|".join(kk): n for kk, n in v.items()} for k, v in self.slot_rules.items()}, attn=dict(self.attn),
                       log=self.log[-50:], when=time.strftime("%Y-%m-%d %H:%M")), open(self.path, "w"), ensure_ascii=False, indent=1)

    # ---------- 2. надёжность способов ----------
    @staticmethod
    def _howkey(how):
        return re.sub(r"[«»].*", "", (how or "").split(":")[0]).strip() or "?"

    def trust(self, how, q):
        """множитель уверенности по опыту: точная форма → грубая форма → вообще. нет опыта → 1.0; 3 промаха из 3 → 0.4"""
        hk = self._howkey(how)
        for key in (f"{hk}|{qform(q)}", f"{hk}|{qcoarse(q)}", f"{hk}|*"):
            h, m = self.reliab.get(key, [0, 0])
            if h + m >= (1 if key.endswith(qform(q)) else 2): return round(0.4 + 0.9 * (h + 1) / (h + m + 2), 2)
        return 1.0

    def boldness(self, q):
        """0..1 — насколько стоит ДОГАДЫВАТЬСЯ (отрывком) вместо «не знаю» на таком типе вопроса.
        Растёт, когда отрывок попадал, и когда «не знаю» оказывалось ошибкой (ответ был, а она молчала)."""
        def rate(hk, default):
            for key in (f"{hk}|{qcoarse(q)}", f"{hk}|*"):
                h, m = self.reliab.get(key, [0, 0])
                if h + m >= 3: return (h + 1) / (h + m + 2)
            return default
        span_ok = rate("отрывок предложения", 0.5)
        silent_wrong = 1 - rate("не знаю → уточнение", 0.5) if any(k.startswith("не знаю") for k in self.reliab) else 0.5
        return round(max(0.0, min(1.0, 0.6 * span_ok + 0.4 * silent_wrong)), 2)

    def _mark(self, how, q, hit):
        hk = self._howkey(how)
        for key in (f"{hk}|{qform(q)}", f"{hk}|{qcoarse(q)}", f"{hk}|*"):
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
        return x >= 2 and x >= n

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
        """выученные предпочтения для формы вопроса: Counter((slot, rel, qwhere))"""
        return self.slot_rules.get(qcoarse(q)) or collections.Counter()

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
            self.slot_rules[qcoarse(q)][(sp["slot"], sp["rel"], sp["qwhere"])] += 1
            grew.append(f"for «{qcoarse(q)}» the answer sits in {sp['slot']} of a «{sp['rel']}» fact")
        extra = self._learn_attention(q, right)
        if extra: grew.append("extra words: " + ", ".join(extra[:4]))
        if self._link(q, right): grew.append("word link " + ", ".join(f"{w}→{right.split()[0]}" for w in content_words(toks(q))[:2]))
        self.log.append(dict(t=time.strftime("%m-%d %H:%M"), kind="correction", q=q, right=right, grew=grew)); self.save()
        return grew

    def learn_from_confirmation(self, q, out):
        """пользователь подтвердил («yes», «right») или не поправил и пошёл дальше → способ надёжнее, связка слов сильнее"""
        if not out or not out.get("answer"): return []
        self._mark(out.get("how"), q, True); self._link(q, str(out["answer"]))
        self._learn_attention(q, str(out["answer"]))
        sp = self._slot_pattern(q, str(out["answer"]))
        if sp: self.slot_rules[qcoarse(q)][(sp["slot"], sp["rel"], sp["qwhere"])] += 1
        fs = getattr(self.mind, "fewshot", None)
        if fs and out.get("how", "").startswith(("2 шага", "few-shot")) is False and qform(q) not in fs.rules:
            fs.show(q, str(out["answer"]))                           # удачный ответ тоже становится схемой
        self.log.append(dict(t=time.strftime("%m-%d %H:%M"), kind="confirm", q=q, a=str(out["answer"]), how=out.get("how"))); self.save()
        return ["trust +1 for " + self._howkey(out.get("how"))]

    # ---------- отчёт ----------
    def report(self):
        fs = getattr(self.mind, "fewshot", None)
        n_rules = len(fs.rules) if fs else len(self.rules_saved)
        forms = sorted(((k, v) for k, v in self.reliab.items() if not k.endswith("|*")), key=lambda kv: -(kv[1][0] + kv[1][1]))[:8]
        soft = [w for w in self.attn if self.is_soft_learned(w)]
        lines = [f"rules learned: {n_rules} (named) + {sum(len(v) for v in self.slot_rules.values())} (where-in-fact, {len(self.slot_rules)} question types)",
                 f"attention learned: {len(soft)} words found unimportant in questions: {', '.join(sorted(soft, key=lambda w: -self.attn[w][1])[:12])}",
                 f"question forms with experience: {len(forms)}",
                 f"word links: {sum(len(v) for v in self.coans.values())}"]
        for k, (h, m) in forms: lines.append(f"  {k}: {h}✓ {m}✗ → trust {round(0.4 + 0.9 * (h + 1) / (h + m + 2), 2)}")
        for e in self.log[-5:]: lines.append(f"  [{e['t']}] {e['kind']}: {e.get('q', '')[:50]} → {e.get('right') or e.get('a', '')}")
        return "\n".join(lines)
