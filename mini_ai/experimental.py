"""Mini AI — ЭКСПЕРИМЕНТАЛЬНАЯ МОДЕЛЬ «память без градиента» (21.10.2026). Один вход для всего:
  ask(q)        — ответ по фактам: рамка (1 шаг) → 2 шага через мост → отказ «не знаю». Показывает факты-источники.
  tell(x)       — рассказать про сущность своими словами (speak.py) из её фактов.
  learn(text)   — прочитать текст: предложения → факты (reader.py) → память. Сразу доступно для ask.
  act(request)  — действие инструментами (agent.py): приёмы, генофонд, проверка.
  code(task, fname, tests) — написать функцию из образцов (coder.py), проверить запуском.
  remember/history — память разговора (последние реплики → контекст для «it/он/это»).
  exp40 (диалог, mini_ai/dialog.py): переспрос вместо «не знаю», нить разговора (эллипсис/местоимения/yes-no),
  разные формы ответа, инициатива. ask() → text (реплика целиком), answer (коротко), kind (answer|clarify|unknown).
Память: факты (список), словарь IDF, образцы кода, приёмы — всё на диске, не забывается.
Источники знаний при старте: школьный учебник (SciQ train, 50k предл.) + Simple Wikipedia (опционально, 1.16M фактов).
"""
import os, re, json, time, collections
import numpy as np
from mini_ai.memory_v2 import toks, content_words, STOP as STOPW
from mini_ai.reader import Reader
from mini_ai.question import parse_question, answer_by_frame, rel_ok
from mini_ai.speak import compose

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.expanduser("~/Developer/mini-ai-runs/memory-core")


class KnowledgeBase:
    """факты + индексы. Факт = (Xw:set, rel, Yw:set, X:str, Y:str, src:str)"""
    def __init__(self):
        self.facts = []; self.by_word = collections.defaultdict(list); self.by_x = collections.defaultdict(list)
        self.df = collections.Counter(); self.n_docs = 0; self.reader = Reader(); self.sources = collections.Counter()
        self.sents = []; self.sent_by_word = collections.defaultdict(list)        # сырые предложения (для ответа-отрывком, когда читатель не разобрал)
        self.sent_ctx_by_word = collections.defaultdict(list)                     # слово предыдущего предложения → это предложение (контекст абзаца)
        self.sent_glued = set()                                                   # предложения-определения, склеенные с предыдущим
        self.sent_para = []                                                       # номер абзаца (вызов read_text) каждого предложения — для цепочек по дискурсу
        self.n_para = 0

    def idf(self, w): return float(np.log((1 + self.n_docs) / (1 + self.df.get(w, 0))) + 1.0)

    def add_fact(self, X, rel, Y, src):
        Xw, Yw = set(content_words(toks(X))), set(content_words(toks(Y)))
        if not (Xw or Yw): return
        k = len(self.facts); self.facts.append((Xw, rel, Yw, X, Y, src))
        for w in Xw | Yw: self.by_word[w].append(k)
        self.by_x[" ".join(toks(X))].append(k); self.sources[src] += 1

    def read_text(self, text, src="user"):
        n = 0
        self.n_para += 1
        for s in re.split(r"(?<=[.!?])\s+(?=[A-ZА-Я0-9\"])", re.sub(r"\s+", " ", text)):
            s = s.strip()
            if len(s) < 15: continue
            if len(s) > 400: s = s[:400]                          # очень длинное — читаем начало (там главное)
            ws = set(content_words(toks(s))); self.n_docs += 1
            for w in ws: self.df[w] += 1
            if src != "user" and len(self.sents) < 400000:
                si = len(self.sents); self.sents.append((s, src)); self.sent_para.append(self.n_para)
                for w in ws: self.sent_by_word[w].append(si)
                # КОНТЕКСТ АБЗАЦА: «These are called X», «This is called X», «One such device is X» — подлежащее указывает на
                # предыдущее предложение. Слова соседа индексируются как слабый контекст этого предложения (вес 0.5 при поиске).
                if si > 0 and self.sents[si - 1][1] == src:
                    prev_ws = set(content_words(toks(self.sents[si - 1][0]))) - ws
                    # предложение-определение с отсылкой («This is called X», «These … are called X», «One such device is X»,
                    # «The process is called X») — сосед ПОЛНОСТЬЮ входит в его слова (это одно высказывание, разбитое на два)
                    # склеиваем ТОЛЬКО предложения-определения с отсылкой: «This/These/The process/One such … is called X» —
                    # у них ответ в этом предложении, а описание — в предыдущем. «They are found in…» — не определение, не склеиваем.
                    if re.match(r"^(?:this|these|that|those|one such|such a|such an|the process|the term|the name|it|they)\b.{0,40}?\b(?:is|are) (?:called|known as|termed|named|referred to as)\b", s, re.I) \
                       or re.match(r"^(?:one such|such a|such an|one example|an example|examples?)\b", s, re.I) \
                       or (re.match(r"^(?:this|these|it|they|that)\b", s, re.I) and len(toks(s)) <= 7):
                        for w in prev_ws: self.sent_by_word[w].append(si); self.sent_glued.add(si)
                    else:
                        for w in prev_ws: self.sent_ctx_by_word[w].append(si)
            for X, rel, Y in self.reader.read(s, src)[0]:
                if "@" not in rel: self.add_fact(X, rel, Y, src); n += 1
        return n

    def load_facts_jsonl(self, path, limit=None):
        n = 0
        with open(path) as f:
            for line in f:
                X, rel, Y, src = json.loads(line)[:4]
                self.add_fact(X, rel, Y, src); n += 1
                if limit and n >= limit: break
        self.n_docs += n
        for Xw, rel, Yw, X, Y, src in self.facts[-n:]:
            for w in Xw | Yw: self.df[w] += 1
        return n


class ExperimentalMind:
    def __init__(self, school=True, wiki=False, wiki_limit=300000, state_dir=None, dialog=True, seed=None):
        self.kb = KnowledgeBase(); self.history = []; self.use_dialog = dialog; self.seed = seed; self.state_dir = state_dir or os.path.join(RUNS, "experimental_state")
        os.makedirs(self.state_dir, exist_ok=True)
        t0 = time.time()
        if school:
            try:
                import pandas as pd
                df = pd.read_parquet(os.path.join(ROOT, "data/school/sciq/data/train-00000-of-00001.parquet"))
                for sup in df["support"].dropna(): self.kb.read_text(str(sup), "учебник")
            except Exception as e: print("учебник не загружен:", e)
        if wiki and os.path.exists(os.path.join(RUNS, "simplewiki_full/facts.jsonl")):
            self.kb.load_facts_jsonl(os.path.join(RUNS, "simplewiki_full/facts.jsonl"), wiki_limit)
        learned = os.path.join(self.state_dir, "learned.txt")
        from mini_ai.meaning import Meaning
        self.meaning = Meaning(self.state_dir)
        if os.path.exists(learned):
            txt = open(learned).read(); self._load_learned(txt)
            if not self.meaning.sem: self.meaning.read(txt); self.meaning.save()
        self.boot = dict(facts=len(self.kb.facts), seconds=round(time.time() - t0, 1), sources=dict(self.kb.sources.most_common(5)))
        self._agent = None; self._coder = None
        from mini_ai.think import Thinker
        from mini_ai.cot import CoT
        from mini_ai.ops import Ops
        from mini_ai.fewshot import FewShot
        self.thinker = Thinker(self.kb); self.cot = CoT(self); self.ops = Ops(self.kb); self.ops.meaning = self.meaning; self.fewshot = FewShot(self.kb)
        from mini_ai.growth import Growth
        from mini_ai.deep import Deep
        self.growth = Growth(self, self.state_dir); self._last_out = None; self._cur_q = ""; self.deep = Deep(self)
        from mini_ai.cases import Cases
        self.cases = Cases(self, self.state_dir); self._advice = None
        from mini_ai.reform import Reform
        self.reform = Reform(self, self.state_dir)
        from mini_ai.solver import Solver
        self.solver = Solver(self, self.state_dir); self._last_cands = None   # обучаемый арбитр без градиента (solver.py): признаки → решение уравнения                    # выученное перечитывание вопроса (уточнители, слово-тип, замены)
        from mini_ai.discourse import Discourse
        self.discourse = Discourse(self)                              # цепочка по абзацу: мост → сосед по связи дискурса → новый термин
        try:
            from mini_ai.russian import Russian
            self.russian = Russian(self, self.state_dir)              # русский: слой перевода (словарь + выученные пары + смысл ru-слов)
            from mini_ai.analogy import Analogy
            self.analogy = Analogy(self, self.state_dir, meaning=getattr(self.russian, "meaning_ru", None))   # одна материя: случаи от учителя, перенос по форме+смыслу (exp69/70)
            from mini_ai.converse import Converse
            self.converse = Converse(self, self.state_dir)            # собеседник без параметров: случаи речи, профиль, схемы рассуждения (23.10)
            self._ru_meaning = None
        except Exception as e: self.russian = None; print("russian off:", e)
        self.neural = None
        if os.environ.get("MINI_AI_NEURAL", "1") != "0":
            try:
                from mini_ai.neural import SentIndex; self.neural = SentIndex(self.kb)
            except Exception as e: print("neural off:", e)
        if self.use_dialog:
            from mini_ai.dialog import Dialog
            self.dialog = Dialog(self.kb, seed=self.seed)          # сервер подменяет на сессионный по sid
        try:
            from mini_ai.hybrid import Speaker
            self.speaker = Speaker()
        except Exception as e: self.speaker = None; print("говорилка не загружена:", e); self.dialog = None; self.new_session()

    def new_session(self):
        """новая нить разговора (сервер держит по одной на sid)"""
        from mini_ai.dialog import Dialog
        self.history = []; self.dialog = Dialog(self.kb, seed=self.seed); return self.dialog

    def _load_learned(self, text):
        """learned.txt: блоки «# wiki: Title» читаются с темой статьи (иначе заголовок склеивался с первым
        предложением и «It is bordered by…» теряло субъект); остальное — как «выучено»."""
        for i, chunk in enumerate(re.split(r"^# wiki: (.+)$", text, flags=re.M)):
            if i % 2 == 1: title = chunk.strip(); continue
            if i == 0: src = "выучено"
            else: src = title
            if chunk.strip(): self.kb.read_text(chunk, src)

    # ---------- понимание + рассуждение ----------
    def _resolve(self, q):
        """местоимения в вопросе → последняя сущность из истории"""
        if self.history and re.search(r"\b(it|this|that|he|she|they|оно|это|он|она)\b", q.lower()):
            last = self.history[-1].get("entity")
            if last: q = re.sub(r"\b(it|this|that|he|she|they)\b", last, q, count=1, flags=re.I)
        return q

    def _one_step(self, fr):
        known = set(fr["Y"] if fr["slot"] == "X" else fr["X"])
        if not known: return None
        cand = set.intersection(*[set(self.kb.by_word.get(w, [])) for w in known])
        if cand:
            res = answer_by_frame(fr, [self.kb.facts[k] for k in cand], {w: self.kb.idf(w) for w in known})
            if res: return res[0]
        return self._one_step_soft(fr, known)

    # слова-пустышки вопроса: «what TERM means…», «what TYPE of cell», «by what METHOD» — они задают форму ответа, а не содержание;
    # в факте их нет и быть не должно → вес ~0 и никогда не обязательны
    META = set("term type kind name word example else another following describes refers called considered known commonly often usually generally "
               "scientific technical medical common proper official formal mechanism device condition disorder disease phenomenon concept "
               "correct best proper right typical standard main specific particular "
               "typically sometimes main primary major important specific certain particular general overall such like thing things something "
               "purpose method feature structure form means mean meaning instrument way ways about far best describe refer describe described "
               "category class sort variety number amount piece unit".split())
    JUNK = set("today now then also usually often sometimes generally typically however thus therefore example instance figure below above here there "
               "several many some few most all each every both another other such one two three first second last next various numerous".split())
    def _stem(self, w): return w[:-2] if w.endswith("es") and len(w) > 5 else (w[:-1] if w.endswith("s") and len(w) > 4 else w)
    _FORMS_CACHE = {}
    def _forms(self, w):
        """множество грубых основ слова: plants/plant, processes/process, invertebrates/invertebrate, bodies/body, moves/move, held/hold.
        Кэш на уровне класса: 10 млн вызовов за 40 вопросов (65 % времени ответа) → словарь (profile 23.10)."""
        c = ExperimentalMind._FORMS_CACHE.get(w)
        if c is not None: return c
        f = self._forms_raw(w)
        if len(ExperimentalMind._FORMS_CACHE) < 400000: ExperimentalMind._FORMS_CACHE[w] = f
        return f
    def _forms_raw(self, w):
        f = {w}
        if w in self.IRREG: f.add(self.IRREG[w]); f.add(self.IRREG[w] + "s")
        if len(w) > 4 and w.endswith("ies"): f.add(w[:-3] + "y")
        if len(w) > 5 and w.endswith("es"): f.add(w[:-2]); f.add(w[:-1])
        elif len(w) > 4 and w.endswith("s"): f.add(w[:-1])
        if len(w) > 4 and w.endswith("e"): f.add(w[:-1])
        if len(w) > 5 and w.endswith("ing"): f.add(w[:-3]); f.add(w[:-3] + "e")
        if len(w) > 5 and w.endswith("ed"): f.add(w[:-2]); f.add(w[:-1])
        if len(w) > 7 and w.endswith("ize"): f.add(w[:-3]); f.add(w[:-3] + "is")          # photosynthesize → photosynthes(is)
        if len(w) > 7 and w.endswith("sis"): f.add(w[:-2]); f.add(w[:-3])                 # photosynthesis → photosynthe(s)
        if len(w) > 6 and w.endswith("tion"): f.add(w[:-4]); f.add(w[:-3])                # digestion → diges(t)
        if len(w) > 5 and w.endswith("ity"): f.add(w[:-3])                                # ability → abil
        return f
    # ВНИМАНИЕ (attention без градиента): какое слово вопроса ОБЯЗАТЕЛЬНО в факте, а какое — лишь подсказка.
    # Обязательны термины (существительные-предметы). Глаголы и связки (holds/held, made, associated, cannot, either, humans)
    # важны слабо: факт может сказать то же другим глаголом. Неправильные формы глаголов — таблица.
    SOFT = set("holds hold held made make makes making associated associate contains contain containing cannot can could either neither "
               "humans human people person things thing comes come came goes go went gives give gave takes take took gets get got "
               "found find finds occurs occur occurring happens happen happened produced produce produces producing allows allow "
               "helps help helped causes cause caused forms form formed moves move moved consists consist consisting involves involve "
               "describes describe refers refer appears appear becomes become remains remain begins begin starts start ends end "
               "approximately about roughly nearly almost mostly mainly usually often sometimes commonly typically generally very much many "
               "together apart inside outside within between among along across through during before after when while where".split())
    IRREG = {"held": "hold", "made": "make", "found": "find", "took": "take", "gave": "give", "came": "come", "went": "go", "got": "get",
             "built": "build", "kept": "keep", "left": "leave", "lost": "lose", "met": "meet", "sent": "send", "spent": "spend", "brought": "bring",
             "bought": "buy", "caught": "catch", "taught": "teach", "thought": "think", "began": "begin", "begun": "begin", "broke": "break",
             "broken": "break", "chose": "choose", "chosen": "choose", "drew": "draw", "drawn": "draw", "grew": "grow", "grown": "grow",
             "knew": "know", "known": "know", "saw": "see", "seen": "see", "ran": "run", "fed": "feed", "led": "lead", "bred": "breed",
             "wrote": "write", "written": "write", "sang": "sing", "sung": "sing", "rose": "rise", "risen": "rise", "fell": "fall", "fallen": "fall"}
    def _widf(self, w):
        if w in self.META: return min(self.kb.idf(w), 0.4)
        if self._soft(w): return min(self.kb.idf(w), 1.2)
        return self.kb.idf(w)
    def _soft(self, w):
        if w in self.META or w in self.SOFT or w in self.IRREG or (len(w) > 4 and (w.endswith("ing") or w.endswith("ed"))): return True
        G = getattr(self, "growth", None)
        return bool(G and G.is_soft_learned(w))                       # выученное внимание: слово было лишним в исправлениях
    def _one_step_soft(self, fr, known):
        """ПОНИМАНИЕ ПО СМЫСЛУ: вопрос перефразирован (plants↔plant, forests↔forest, слова переставлены, одно слово
        заменено синонимом) → факт, покрывающий ≥70 % веса слов вопроса с учётом форм и близости смысла (meaning),
        при этом редкие слова (idf высок) обязательны. Ответ — свободная часть факта."""
        idf = {w: self._widf(w) for w in known}; kidf = sum(idf.values())
        if kidf <= 0: return None
        cnt = collections.Counter()
        for w in known:
            if w in self.META: continue
            fw = self._forms(w)
            ws = {w} | {x for x in self.kb.by_word if len(x) > 3 and (self._forms(x) & fw)} if len(w) > 3 else {w}
            for x in list(ws)[:8]:
                for k in self.kb.by_word.get(x, []): cnt[k] += idf[w] * (1.0 if x == w else 0.9)
        for k in list(cnt):                                                          # + глагол факта, если он в вопросе
            rw = set(toks(self.kb.facts[k][1].replace("_", " "))) & known
            for w in rw: cnt[k] += idf[w]
        best = None
        for k, score in cnt.most_common(40):
            Xw, rel, Yw, X, Y, src = self.kb.facts[k]; allw = Xw | Yw | set(toks(rel.replace("_", " ")))
            stemmed = set().union(*[self._forms(x) for x in allw]) if allw else set()
            missing = [w for w in known if w not in allw and not (self._forms(w) & stemmed)]
            # пропущенное слово допустимо, только если оно частое (idf < 2.0) или по смыслу близко к какому-то слову факта
            M = getattr(self, "meaning", None); ok = True
            for w in missing:
                if idf[w] < 2.0 or self._soft(w): continue
                if M is not None and any((M.similar(w, x) or 0) > 0.35 for x in allw if len(x) > 3): continue
                ok = False; break
            if not ok or score / kidf < 0.7: continue
            part = Yw if fr["slot"] == "X" else Xw
            ans = X if fr["slot"] == "X" else Y
            if not ans or set(content_words(toks(ans))) <= known: continue       # ответ не должен быть повтором вопроса
            s = score / kidf * (1.2 if rel_ok(fr["rel"], rel) else 1.0) * (1.1 if known & part else 1.0)
            # РАНЖИР ПО СМЫСЛУ: код вопроса · код факта (meaning) — среди фактов с одинаковыми словами выбираем тот,
            # чей смысл ближе ко всему вопросу (а не к одному слову)
            if M is not None:
                ps = M.phrase_sim(known, allw)
                if ps is not None: s *= (0.7 + 0.6 * max(0.0, ps))
            if not best or s > best[0]: best = (round(s * 0.8, 2), ans, (X, rel, Y, src))
        if best: return best
        # СКЛЕЙКА: вопрос покрывает ДВА атомарных факта с общей сущностью («plant hormones be chemical signals» +
        # «chemical signals control processes») → ответ = свободный конец цепочки (X первого факта)
        for k, score in cnt.most_common(25):
            Xw, rel, Yw, X, Y, src = self.kb.facts[k]
            for k2 in set(self.kb.by_x.get(" ".join(toks(X)), [])) | {j for w in Xw for j in self.kb.by_word.get(w, [])[:200]}:
                if k2 == k: continue
                X2w, rel2, Y2w, X2, Y2, src2 = self.kb.facts[k2]
                if not (Xw and Xw <= (X2w | Y2w)) : continue                       # общая сущность = субъект k внутри k2
                allw = Xw | Yw | X2w | Y2w | set(toks(rel.replace("_", " "))) | set(toks(rel2.replace("_", " "))); stemmed = {self._stem(x) for x in allw}
                if any(w not in allw and self._stem(w) not in stemmed and idf[w] >= 2.0 for w in known): continue
                cover = sum(idf[w] for w in known if w in allw or self._stem(w) in stemmed) / kidf
                if cover < 0.85: continue
                free = X2 if Xw <= Y2w else Y2                                      # конец цепочки, которого нет в вопросе
                if not free or set(content_words(toks(free))) <= known: continue
                sc = round(cover * 0.75, 2)
                if not best or sc > best[0]: best = (sc, free, (X2, rel2, Y2, src2), (X, rel, Y, src))
        return best[:3] if best else None

    def _span(self, fr, known0, only_sents=None, relax=0.0):
        """ОТРЫВОК: читатель не разобрал предложение на факты (длинное, список, «X, Y and Z indicate…»), но
        предложение памяти покрывает ≥60 % веса слов вопроса → ответ = его свободный кусок: часть предложения
        (между знаками/союзами), где нет слов вопроса. Это не поиск по словам «на удачу»: обязательны редкие слова.
        relax>0 — ВТОРОЙ, СМЕЛЫЙ проход (случаи сказали: на таких вопросах молчание было ошибкой): порог ниже, одно редкое слово прощается."""
        known = {w for w in known0 if w not in self.META} | {w for w in content_words(toks((fr.get("rel") or "").replace("_", " "))) if w not in self.META and w not in ("prop", "called", "be", "do", "have")}
        # ТИП ОТВЕТА в вопросе («what ANIMAL GROUP», «what OBJECTS are», «what ORGAN in», «what is the RESULT», «which BODY SYSTEM»)
        # — это не слова, которые должны быть в факте; это то, ЧЕМ должен быть ответ. Убираем из обязательных.
        qt = re.search(r"\b(?:what|which)\s+(?:broad\s+|main\s+|primary\s+|basic\s+|large\s+|small\s+|general\s+)?([a-z]+(?:\s+[a-z]+)?)\s+(?:is|are|was|were|do|does|did|in|of|that|has|have|can|will|would)\b", self._cur_q.lower())
        atype_w = set(content_words(toks(qt.group(1)))) if qt else set()
        qt2 = re.search(r"\bwhat (?:is|are|was|were) the ([a-z]+)\s*\??$", self._cur_q.lower())
        if qt2: atype_w.add(qt2.group(1))
        atype_w |= {w for w in known if w in ("result", "outcome", "objects", "object", "body", "group", "organ", "material", "substance", "structure", "feature", "property", "process", "system", "element", "term", "type", "kind")}
        known = known - atype_w
        RF = getattr(self, "reform", None)
        if RF: known = known - RF.type_words(self._cur_q)            # выученное: «what PART of…» — part задаёт тип ответа, не искомое слово
        if not known: return None
        idf = {w: self._widf(w) for w in known}; forms = {w: self._forms(w) for w in known}
        # СЛУЧАИ: похожие исправленные вопросы говорят, какие слова были лишними → они мягкие здесь
        adv = self._advice if self._advice and self._advice.get("q") == self._cur_q else None
        case_soft = (adv or {}).get("soft", set())
        case_soft_cov = case_soft if os.environ.get("MINI_AI_CASES_SOFT_COV", "0") == "1" else set()   # измерено: мягкие слова из случаев прощаются только как «редкое слово» (rare_missing), в знаменателе покрытия они остаются (+0.003 vs в знаменателе)
        # покрытие считаем по ТВЁРДЫМ словам (терминам); мягкие (глаголы, связки) — только бонус, не знаменатель
        hard = {w for w in known if not self._soft(w) and w not in case_soft_cov} or known
        kidf = sum(idf[w] for w in hard) + 0.3 * sum(idf[w] for w in known - hard)
        qforms = set().union(*[self._forms(w) for w in (known0 | known)]) | {w for w in known0}   # все слова вопроса (и пустышки) — не ответ
        cnt = collections.Counter(); ctx_hit = collections.Counter(); self._neural_hit = {}; G = getattr(self, "growth", None)
        if only_sents is None and not relax: self._span_alts = []
        for w in known:
            wgt = idf[w] if w in hard else 0.3 * idf[w]
            alts = ({w} | {x for x in self.kb.sent_by_word if len(x) > 3 and (self._forms(x) & forms[w])} if len(w) > 3 else {w})
            if G: alts |= set(G.paraphrases(w))                                       # выученный перефраз: holds → held, make → produce
            if RF: alts |= set(RF.subs(w))                                            # выученная замена из выравнивания вопроса с предложением
            for x in alts:
                for si in self.kb.sent_by_word.get(x, []): cnt[si] += wgt * (1.0 if x == w else 0.9)
                for si in self.kb.sent_ctx_by_word.get(x, []):
                    # контекст абзаца для обычных предложений — слабый (0.3): только чтобы не потерять, не чтобы победить
                    cnt[si] += wgt * 0.3 * (1.0 if x == w else 0.9); ctx_hit[si] += 1
        best = None
        # СМЕЛОСТЬ ПО ОПЫТУ: если на таком типе вопроса «отрывок» попадал, а «не знаю» промахивался — порог покрытия ниже (0.45),
        # если отрывок ошибался — выше (0.7). Ядро само решает, где можно догадываться.
        G = getattr(self, "growth", None); q_ = " ".join(sorted(known0))
        bold = G.boldness(self._cur_q) if G else 0.5
        thr = 0.75 - 0.4 * bold                       # смелость 0.5 → порог 0.55; опыт «отрывок попадает, молчать — ошибка» → 0.35
        dr = G.diag_rates(self._cur_q) if G else {}
        if dr: thr -= 0.15 * dr.get("silent", 0) - 0.1 * dr.get("wrong-piece", 0)   # молчала зря → смелее; брала не тот кусок → строже
        # СМЕЛЫЙ ПРОХОД (из случаев): порог ниже на relax, одно лишнее редкое слово вопроса прощается
        thr = max(0.3, thr - relax); tol = 1 if relax > 0 else 0
        DEF_RX = re.compile(r"\b(?:is|are) (?:called|known as|termed|named|referred to as)\b|^(?:this|these|that|those|one such|such a|the process|the term|the name)\b", re.I)
        for si, score in list(cnt.most_common(12)):
            # предложение-определение («… is called X», «These are called X», «One such device is X») — именно в нём ответ;
            # покрытие ему добавляет сосед (контекст), а сам факт определения — бонус
            if si in self.kb.sent_glued or re.search(r"\b(?:is|are) (?:called|known as|termed|named|referred to as)\b", self.kb.sents[si][0], re.I): cnt[si] = score * 1.3
        if only_sents is not None:                                   # комбинация: предложения, найденные другим способом — покрытие не проверяем строго
            cnt = collections.Counter({si: max(cnt.get(si, 0), kidf * max(thr, 0.6)) for si in only_sents})
        elif getattr(self, "neural", None) is not None:
            # НЕЙРОННЫЙ ГЛАЗ: предложения, близкие к вопросу ПО СМЫСЛУ (другие слова) — добавляем как кандидатов с покрытием = сходство;
            # дальше они проходят те же проверки (редкие слова, куски), что и найденные по словам
            for si, sim in self.neural.search(self._cur_q, k=6, min_sim=0.45):
                cnt[si] = max(cnt.get(si, 0), kidf * sim); self._neural_hit[si] = sim
        # СЛЕДУЮЩЕЕ ПРЕДЛОЖЕНИЕ: если предложение совпало с вопросом по смыслу/словам, ответ часто в СЛЕДУЮЩЕМ
        # («Several devices measure radiation. ONE SUCH DEVICE IS THE GEIGER COUNTER.», «Mammals raise their hair… THEY CAUSE GOOSEBUMPS.»)
        # — добавляем сосед справа как кандидата с покрытием предыдущего × 0.9 (и помечаем, чтобы при выборе куска учесть)
        follow = {}
        for si, score in list(cnt.most_common(6)):
            nx = si + 1
            if nx < len(self.kb.sents) and self.kb.sents[nx][1] == self.kb.sents[si][1] and score / kidf >= thr:
                if re.match(r"^(?:this|these|that|those|one such|such a|such an|they|it|the process|the term|the result|this is|these are)\b", self.kb.sents[nx][0], re.I) or len(toks(self.kb.sents[nx][0])) <= 8:
                    if cnt.get(nx, 0) < score * 0.9: cnt[nx] = score * 0.9; follow[nx] = si
        dbg = dict(thr=round(thr, 2), top=[], picked=None); alts = []                        # след для cases.learn / отладки: что видела и почему молчала
        if only_sents is None and not relax: self._span_dbg = dbg                          # урок берём со строгого прохода
        for si, score in cnt.most_common(12):
            cov = score / kidf
            dbg["top"].append([si, round(cov, 2), None, None])
            sent, src = self.kb.sents[si]; sw = set(content_words(toks(sent)))
            if si in self.kb.sent_glued or si in follow:
                sw = sw | set(content_words(toks(self.kb.sents[si - 1][0])))        # определение с отсылкой / следующее предложение: слова соседа — тоже «в предложении»
            sforms = set().union(*[self._forms(x) for x in sw]) if sw else set()
            # ВЫУЧЕННЫЕ УТОЧНИТЕЛИ (reform): кусок вопроса, целиком отсутствующий в предложении и имеющий форму, которая в исправлениях
            # оказывалась лишней («within an incomplete digestive system», «such as the breakdown of…»), — не требуем; покрытие пересчитываем
            absent = {w for w in known if not (forms[w] & sforms)}; qual = set(); cov_raw = cov
            if RF and absent:
                qual = RF.qualifiers(self._cur_q, absent) & absent
                if qual:
                    drop = sum(idf[w] if w in hard else 0.3 * idf[w] for w in qual)
                    # остаток должен держать смысл: ≥1 твёрдое слово найдено, и остаётся ≥35 % веса (или ≥2 твёрдых найденных слова)
                    found_hard = [w for w in hard if w not in absent]
                    if found_hard and (kidf - drop >= 0.35 * kidf or len(found_hard) >= 2) and kidf - drop > 0: cov = score / (kidf - drop)
                    else: qual = set()
            passed = cov >= thr                                               # фильтры НЕ прерывают: предложение-«почти» идёт в альтернативы для solver'а
            if not passed and cov < thr - float(os.environ.get("MINI_AI_ALT_COV", "0.25")): continue
            hy = set(re.findall(r"\b([a-z]+)-(?:like|type|shaped|based|free|rich)\b", self._cur_q.lower()))          # «tube-like» — описание, не термин
            rare_missing = [w for w in known if idf[w] >= 2.5 and not self._soft(w) and w not in hy and w not in case_soft and w not in qual and not (forms[w] & sforms) and not (G and set(G.paraphrases(w)) & sw) and not (RF and set(RF.subs(w)) & sw)]
            if self._neural_hit.get(si, 0) >= 0.6: rare_missing = rare_missing[:1] if cov >= 0.5 else rare_missing[:2]   # смысл совпал — слова могут отличаться
            dbg["top"][-1][2] = rare_missing
            # одно пропущенное твёрдое слово допустимо, если покрытие всё равно высокое и оно не самое редкое в вопросе
            # (вопрос часто содержит лишнее уточнение: «in physics», «located below the soil», «during evolution»)
            if rare_missing:
                if len(rare_missing) > 2 + tol or cov < thr + 0.1 - 0.1 * tol: passed = False
                if len(rare_missing) == 2 and cov < 0.75 - 0.2 * tol: passed = False
                if not tol and idf[rare_missing[0]] >= max(idf[w] for w in hard) and len(hard) <= 3 and cov < 0.8: passed = False
                if len(rare_missing) > 3: continue
            # куски предложения; кандидат = кусок без слов вопроса, с ≥1 содержательным словом
            body = re.sub(r"\s*\(([^)]*)\)", r", \1, ", sent)                       # «(about 2 km)» — тоже кусок
            pieces = [p.strip(" ,;:.") for p in re.split(r",|;|:| — | - |\b(?:and|or|but|because|which|that|while|whereas|so that|such as|called|known as|is|are|was|were|by|through|into|from|than)\b", body)]
            cands = []
            for p in pieces:
                pw = set(content_words(toks(p))) - self.JUNK
                if not pw: continue
                if all(x.isdigit() for x in pw) and not re.search(r"\bhow (?:many|much|long|old|far)\b|\bwhat (?:year|number|percent|age)\b", self._cur_q, re.I): continue
                pf = set().union(*[self._forms(x) for x in pw])
                overlap = sum(1 for x in pw if (self._forms(x) & qforms) or x in self.META)        # слов куска, которые есть в вопросе
                if len(pw) <= 4 and re.search(r"\b(?:called|known as|termed|named|referred to as)\s+(?:a |an |the )?" + re.escape(p.split()[0]), body, re.I):
                    # «is called frequency modulation» — после «called» термин берётся целиком, даже если в нём есть слово вопроса
                    cands.append((max(self.kb.idf(x) for x in pw) + 1.5, p)); continue
                if overlap and len(pw) > overlap:
                    # кусок со словами вопроса: ответ — самый длинный пробел между ними («most living things use OXYGEN to make ATP»)
                    # ПОРЯДОК СЛОВ: пробел в вопросе (где стоит «what») окружён словами; в предложении ответ — тот пробел,
                    # у которого те же соседи. «what do X use ___ to make ATP» → сосед слева «use», справа «make».
                    qt_ = toks(self._cur_q); gap_l = gap_r = None
                    for i_, x in enumerate(qt_):
                        if x in ("what", "which") and 0 < i_ < len(qt_) - 1 and i_ > 2:
                            gap_l = next((y for y in reversed(qt_[:i_]) if y not in STOPW), None); gap_r = next((y for y in qt_[i_+1:] if y not in STOPW and y not in self.META), None); break
                    if gap_l is None and qt_ and qt_[-1] in ("what", "which"): gap_l = next((y for y in reversed(qt_[:-1]) if y not in STOPW), None)
                    if gap_l is None and len(qt_) > 3 and qt_[0] == "what" and qt_[1] in ("do", "does", "did"):
                        # «What do X USE ___ to make Y» — дополнение вынесено вперёд; пробел — сразу после глагола (первое не-стоп слово после подлежащего, не -ing)
                        subj_end = next((i_ for i_ in range(2, len(qt_)) if qt_[i_] not in STOPW and not qt_[i_].endswith("ing") and i_ >= 3 and qt_[i_-1] not in STOPW), None)
                        verb = next((qt_[i_] for i_ in range(2, len(qt_)) if qt_[i_] in ("use", "need", "require", "produce", "make", "form", "release", "eat", "have", "contain", "get", "take", "absorb", "store", "carry", "call")), None)
                        if verb: gap_l = verb; vi = qt_.index(verb); gap_r = next((y for y in qt_[vi+1:] if y not in STOPW and y not in self.META), None)
                    run, runs, prev = [], [], None; pt = toks(p)
                    for i_, x in enumerate(pt):
                        isq = (self._forms(x) & qforms) or x in self.META
                        if x in STOPW or x in self.JUNK:
                            if run: runs.append((run, prev, None)); run = []
                            continue
                        if isq:
                            if run: runs[-1] = (runs[-1][0], runs[-1][1], x) if runs and runs[-1][0] is run else runs.append((run, prev, x)) or runs[-1]; run = []
                            prev = x
                        else: run.append(x)
                    if run: runs.append((run, prev, None))
                    for r_, left, right in runs:
                        if not (1 <= len(r_) <= 4): continue
                        sc_ = max(self.kb.idf(x) for x in r_) * 0.9 - 0.05 * len(r_)
                        if gap_l and left and (self._forms(left) & self._forms(gap_l)): sc_ += 1.0        # тот же сосед слева, что у «what» в вопросе
                        if gap_r and right and (self._forms(right) & self._forms(gap_r)): sc_ += 0.6
                        for cl, cr, cs in (adv or {}).get("pos", []):                                     # соседи ответа из похожих случаев
                            if cl and left and (self._forms(left) & self._forms(cl)): sc_ += 0.5 * cs
                            if cr and right and (self._forms(right) & self._forms(cr)): sc_ += 0.3 * cs
                        cands.append((sc_, " ".join(r_)))
                    continue
                if overlap == 0 and len(pw) <= 6:
                    # лучший кусок — с самым редким словом (термин), не «made»/«used»/«found» (глагольные хвосты)
                    verbish = all(x.endswith(("ed", "ing")) or x in ("made", "used", "found", "done", "seen", "known") for x in pw)
                    sc_ = max(self.kb.idf(x) for x in pw) * (0.3 if verbish else 1.0) - 0.05 * len(pw)
                    if re.search(r"\b(?:called|known as|termed|named|referred to as)\s+" + re.escape(p.split()[0]), body, re.I): sc_ += 1.5   # «is called X» — X и есть ответ
                    cands.append((sc_, p))

            numq = bool(re.search(r"\bhow (?:many|much|long|old|far)\b|\bwhat (?:year|number|percent|age)\b", self._cur_q, re.I))
            cands = [(sc_, pc) for sc_, pc in cands if not (all(x.isdigit() for x in toks(pc)) and not numq)]      # «1», «2050» — не ответ на «what»
            if not cands: continue
            if G: cands = [(sc_ * G.pos_prior(self._cur_q, sent, pc), pc) for sc_, pc in cands]     # выученная позиция ответа
            if getattr(self, "neural", None) is not None and len(cands) > 1:
                # НЕЙРОННЫЙ ВЫБОР КУСКА: «вопрос + кусок» как утверждение против предложения — какой кусок делает утверждение
                # ближе к предложению-источнику? («What holds together A and T? hydrogen bonds» ≈ «…held together with two hydrogen bonds»)
                try:
                    qbase = re.sub(r"\b(?:what|which|who|where|when|how)\b", "", self._cur_q.rstrip("?"), flags=re.I).strip()
                    stmts = [f"{qbase} {pc}" for _, pc in cands]
                    from mini_ai.neural import _model
                    vs = _model().encode([sent] + stmts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
                    sims = vs[1:] @ vs[0]
                    top = max(sc_ for sc_, _ in cands) or 1.0
                    cands = [(sc_ * 0.5 + top * 0.5 * float(sm), pc) for (sc_, pc), sm in zip(cands, sims)]
                except Exception: pass
            ans = max(cands, key=lambda c: c[0])[1]
            _clean = lambda a: re.sub(r"^(?:the|a|an|its|their|also|about|only|all|most|many|some|very)\s+", "", a, flags=re.I).strip()
            ans = _clean(ans)
            if len(ans) < 2: continue
            dbg["top"][-1][3] = ans
            sc = round(cov_raw * 0.55, 2)                    # уверенность — по честному покрытию: уточнитель открывает дверь, но не добавляет веры
            # АЛЬТЕРНАТИВЫ ДЛЯ ОБУЧАЕМОГО АРБИТРА: лучший кусок КАЖДОГО предложения из топ-12 (в т.ч. не прошедшего фильтр) —
            # замер (5000): из 28 промахов отрывка у 17 верный ответ был в другом предложении топ-12, лишь у 2 — в другом куске того же
            alts.append(dict(sc=sc, ans=ans, sent=sent[:120], src=src, rank=len(dbg["top"]) - 1, cov=round(cov, 2), passed=passed,
                             n_rare=len(rare_missing), neural=round(self._neural_hit.get(si, 0), 2), top2=[_clean(pc) for _, pc in sorted(cands, key=lambda c: -c[0])[1:3]]))
            if not passed: continue
            if not best or sc > best[0]: best = (sc, ans, (sent[:120], "span", "", src))
        if only_sents is None and not relax: self._span_alts = [a for a in alts if not best or a["ans"].lower() != best[1].lower()][:8]
        return best

    def _arbiter(self, q, fr):
        """все способы → кандидаты (how, answer, confidence, facts); счёт = confidence × trust(how, форма вопроса).
        Подсказки-связки (growth.hints) добавляют +0.15 кандидату, чей ответ уже встречался на похожих вопросах."""
        G = getattr(self, "growth", None); cands = []; alt_cands = []
        r = self._one_step(fr)
        if r: s, ans, fact = r; cands.append(dict(how="1 шаг", answer=ans, confidence=round(min(1.0, s / 1.8), 2), facts=[f"{fact[0]} —{fact[1]}→ {fact[2]}"], raw=r))
        r2 = self._two_step(fr)
        if r2:
            s, ans, t, F1, F2 = r2
            cands.append(dict(how=f"2 шага через «{t}»", answer=ans, confidence=round(min(1.0, s / 2.0), 2), facts=[f"{F1[3]} —{F1[1]}→ {F1[4]}", f"{F2[3]} —{F2[1]}→ {F2[4]}"]))
        sp = self._span(fr, set(fr["Y"] if fr["slot"] == "X" else fr["X"]) | set(fr["Y"] if fr["slot"] == "Y" else fr["X"]))
        if sp:
            s, ans, (sent, _, _, src) = sp
            aw_ = [w for w in toks(ans) if w not in STOPW]
            vb_ = bool(aw_) and all(len(w) > 4 and w.endswith(("ed", "ing")) for w in aw_)             # «classified», «surrounding» — глагольная форма редко бывает ответом
            cands.append(dict(how="отрывок предложения", answer=ans, confidence=round(min(0.9, s / 0.6) * (0.8 if vb_ else 1.0), 2), facts=[f"{ans} —span→ {sent}"], text=f"From what I've read: {sent.rstrip('.')}. So I'd say: {ans}."))
            # варианты куска — на доску с пониженной уверенностью; выиграть могут только через обучаемый арбитр (solver), в согласие не входят
            if os.environ.get("MINI_AI_SPAN_ALTS", "1") != "0":
                seen_ = {str(ans).lower()}
                for a_ in getattr(self, "_span_alts", None) or []:
                    pc = a_["ans"]
                    if pc.lower() in seen_: continue
                    seen_.add(pc.lower())
                    alt_cands.append(dict(how="отрывок: другое предложение", answer=pc, confidence=round(min(0.9, a_["sc"] / 0.6) * (0.7 if a_["passed"] else 0.4), 2), facts=[f"{pc} —span→ {a_['sent']}"],
                                      alt_rank=a_["rank"], alt_rel=a_["cov"], alt_passed=a_["passed"], alt_rare=a_["n_rare"], alt_neural=a_["neural"],
                                      text=f"From what I've read: {a_['sent'].rstrip('.')}. So I'd say: {pc}."))
        elif not cands:
            # СМЕЛОСТЬ ИЗ СЛУЧАЕВ: на доске пусто, а похожие исправленные вопросы говорят «предложение с ответом было видно, я молчала зря»
            # → второй, смелый проход отрывка. Только когда иначе был бы «не знаю»: верный ответ другого способа он вытеснить не может.
            adv = self._advice if self._advice and self._advice.get("q") == q else None
            bold_c = float((adv or {}).get("bold", 0.0) or 0.0)
            if os.environ.get("MINI_AI_FORCE_BOLD"): bold_c = float(os.environ["MINI_AI_FORCE_BOLD"])                 # замер потолка (probe)
            # РЕШЕНИЕ «МОЛЧАТЬ ИЛИ РИСКНУТЬ» ОТДАНО SOLVER'У: смелый проход считаем всегда, когда иначе было бы «не знаю»;
            # принимаем, если так советуют случаи (bold_c) ИЛИ обученный арбитр даёт p ≥ порога. Отказ тоже запоминается
            # (self._last_cands) — исправление «no, it's X» после молчания размечает этого кандидата и учит порог.
            use_solver_bold = os.environ.get("MINI_AI_SOLVER_BOLD", "1") != "0" and getattr(self, "solver", None) is not None
            if bold_c >= float(os.environ.get("MINI_AI_BOLD_MIN", "0.3")) or use_solver_bold:
                sp = self._span(fr, set(fr["Y"] if fr["slot"] == "X" else fr["X"]) | set(fr["Y"] if fr["slot"] == "Y" else fr["X"]), relax=float(os.environ.get("MINI_AI_BOLD_RELAX", "0.2")))
                if sp:
                    s, ans, (sent, _, _, src) = sp
                    c_bold = dict(how="отрывок предложения (смело, по случаям)", answer=ans, confidence=round(min(0.7, s / 0.6), 2), facts=[f"{ans} —span→ {sent}"], text=f"From what I've read: {sent.rstrip('.')}. So I'd say: {ans}.")
                    take = bold_c >= float(os.environ.get("MINI_AI_BOLD_MIN", "0.3"))
                    if use_solver_bold:
                        try:
                            c_bold["score"] = c_bold["confidence"]; c_bold["trust"] = 1.0; c_bold["check"] = self._self_check(fr, ans)["note"]
                            fb = self.solver.features(q, [c_bold]); pb = self.solver.predict(fb)[0] if self.solver.ready else None
                            c_bold["solver"] = None if pb is None else round(pb, 2)
                            if pb is not None and pb >= float(os.environ.get("MINI_AI_SOLVER_PBOLD", "0.5")): take = True; c_bold["how"] = "отрывок предложения (смело, по solver)"
                            if pb is not None and pb < float(os.environ.get("MINI_AI_SOLVER_PVETO", "0.0")): take = False
                            if not take: self._last_cands = (q, [dict(c_bold)], fb)          # молчим, но запоминаем — исправление научит
                        except Exception: pass
                    if take: cands.append(c_bold)
        # УЧЁБА У ПРОИГРАВШИХ (growth.learn_from_board): если на этом классе способ A обычно находит нужное предложение,
        # но не ответ — берём его предложение и достаём ответ как отрывок. Правило выведено ею из исправлений.
        pf = G.place_finders(q) if G else []
        if pf:
            sis = []
            for c in cands:
                if G._howkey(c["how"]) in pf and c.get("facts"):
                    m_ = re.match(r"^(.*?) —(.*?)→ (.*)$", c["facts"][-1])
                    if not m_: continue
                    fx = set(content_words(toks(m_.group(1) + " " + m_.group(3))))
                    for w in list(fx)[:6]:
                        for si in self.kb.sent_by_word.get(w, [])[:50]:
                            if fx <= set(content_words(toks(self.kb.sents[si][0]))) | {w for w in fx if len(w) < 4}: sis.append(si)
            if sis:
                sp2 = self._span(fr, set(fr["Y"] if fr["slot"] == "X" else fr["X"]) | set(fr["Y"] if fr["slot"] == "Y" else fr["X"]), only_sents=list(dict.fromkeys(sis))[:5])
                if sp2 and not any(str(c["answer"]).lower() == str(sp2[1]).lower() for c in cands):
                    s_, ans, (sent, _, _, src) = sp2
                    cands.append(dict(how="комбинация (место от " + "/".join(pf) + " + отрывок)", answer=ans, confidence=round(min(1.0, s_ / 0.6), 2), facts=[f"{ans} —span→ {sent}"],
                                      text=f"From what I've read: {sent.rstrip('.')}. So I'd say: {ans}."))
        # НЕЙРОННЫЙ ЧИТАТЕЛЬ (neural.read_answer): ядро выбирает 2–4 предложения-места (по словам, по смыслу, по соседству),
        # читатель достаёт из них кусок. Это гибрид: где искать — символьная память и рост; что именно взять — обученная модель.
        if getattr(self, "neural", None) is not None and self.kb.sents:
            try:
                sis = []
                known_ = set(fr["Y"] if fr["slot"] == "X" else fr["X"]) | set(fr["Y"] if fr["slot"] == "Y" else fr["X"])
                for w in sorted(known_, key=lambda w: -self.kb.idf(w))[:4]:
                    for x in ({w} | {x for x in self.kb.sent_by_word if len(x) > 3 and (self._forms(x) & self._forms(w))} if len(w) > 3 else {w}):
                        sis += self.kb.sent_by_word.get(x, [])[:30]
                cnt_ = collections.Counter(sis)
                top = [si for si, _ in cnt_.most_common(3)]
                top += [si for si, _ in self.neural.search(q, k=3, min_sim=0.4) if si not in top]
                ctx_ids = []
                for si in top[:4]:
                    for j in (si - 1, si, si + 1):                                    # предложение с соседями
                        if 0 <= j < len(self.kb.sents) and j not in ctx_ids: ctx_ids.append(j)
                ctx_ids = sorted(ctx_ids)[:10]
                if ctx_ids:
                    from mini_ai.neural import read_answer
                    context = " ".join(self.kb.sents[j][0] for j in ctx_ids)
                    ans, sc = read_answer(q, context)
                    aw = set(content_words(toks(ans or "")))
                    if ans and sc >= 12.0 and aw and not (aw <= known_):
                        src_sent = next((self.kb.sents[j][0] for j in ctx_ids if ans.lower() in self.kb.sents[j][0].lower()), context[:120])
                        # калибровка по измерению (A/B на 300): score<22 → ~0.3 точности, ≥22 → ~0.6; уверенность = измеренная точность
                        conf = 0.3 if sc < 18 else (0.45 if sc < 22 else (0.6 if sc < 26 else 0.72))
                        # предложение-источник должно содержать хоть одно твёрдое слово вопроса — иначе читатель «угадал» из чужого контекста
                        ssw = set(content_words(toks(src_sent))); sf = set().union(*[self._forms(x) for x in ssw]) if ssw else set()
                        hardq = [w for w in known_ if not self._soft(w) and w not in self.META and len(w) > 3]
                        if hardq and not any(self._forms(w) & sf for w in hardq): conf *= 0.6
                        cands.append(dict(how="нейронный читатель", answer=ans, confidence=conf, facts=[f"{ans} —span→ {src_sent[:160]}"],
                                          text=f"From what I've read: {src_sent.rstrip('.')[:160]}. So I'd say: {ans}."))
            except Exception as e_: pass
        # ГЛУБОКОЕ РАССУЖДЕНИЕ (deep.py): до 7 шагов по графу фактов с откатом — когда прямые способы слабы или молчат
        try: dp = self.deep.reason(q)                                   # deep думает всегда: его сила — тип ответа и цепочки
        except Exception: dp = None
        if True:
            if dp and dp["steps"] >= 2:
                cands.append(dict(how=f"{dp['steps']} шагов (deep)", answer=dp["answer"], confidence=dp["confidence"], facts=dp["facts"], thoughts=dp["thoughts"],
                                  text=" ".join(dp["thoughts"])))
        # ЦЕПОЧКА ПО ДИСКУРСУ (discourse.py): вопрос совпал с одним предложением, ответ — в соседнем, связанном отсылкой
        # («…liquid would be left behind. PLASMA is this fluid part…», «…growth of algae. This is called an ALGAL BLOOM»)
        if os.environ.get("MINI_AI_DISCOURSE", "1") == "1":
            try: dc = self.discourse.reason(q, fr)
            except Exception: dc = None
            if dc:
                cands.append(dict(how="3 шага (по абзацу)", answer=dc["answer"], confidence=dc["confidence"], facts=dc["facts"], thoughts=dc["thoughts"],
                                  text=" ".join(dc["thoughts"])))
        if not cands: return None                                 # _last_cands при молчании уже содержит отвергнутого смелого кандидата
        # СОГЛАСИЕ: два разных способа дали один ответ — независимые свидетели → уверенность вверх; одиночка против согласных — вниз.
        # (измерено на SciQ: 16 из 130 промахов — верный ответ был на доске у двух способов, но проигрывал одиночному отрывку 0.9)
        def _aw(c): return {w for w in content_words(toks(str(c["answer"]))) if len(w) > 2}
        agree = {}
        for i, a in enumerate(cands):
            for j, b in enumerate(cands):
                if j <= i or a["how"][:6] == b["how"][:6]: continue
                A, B = _aw(a), _aw(b)
                if A and B and (A <= B or B <= A): agree.setdefault(i, set()).add(j); agree.setdefault(j, set()).add(i)
        if agree:
            for i, c in enumerate(cands):
                n_ = len(agree.get(i, ()))
                if n_: c["confidence"] = round(min(1.0, c["confidence"] + 0.2 + 0.1 * (n_ - 1)), 2); c["how"] = c["how"] + " +согласие"
                else: c["confidence"] = round(c["confidence"] * 0.75, 2)
        hints = set(G.hints(q)) if G else set(); prefs = G.slot_prefs(q) if G else {}
        if self._advice and self._advice.get("q") == q and self._advice.get("how"):
            for c in cands:                                                 # похожие случаи решались этим способом → бонус
                hk = re.sub(r"[«»].*", "", c["how"]).split(":")[0].strip()
                b = self._advice["how"].get(hk, 0)
                if b: c["confidence"] = round(min(1.0, c["confidence"] + 0.15 * min(1.0, b)), 2)
        dr = G.diag_rates(q) if G else {}
        for c in cands:                                            # диагноз → кого двигать: «не тот факт» → отрывку +, «не тот кусок» → факту +
            if dr.get("wrong-fact", 0) > 0.4 and c["how"] == "отрывок предложения": c["confidence"] = min(1.0, c["confidence"] + 0.1)
            if dr.get("wrong-piece", 0) > 0.4 and c["how"] == "1 шаг": c["confidence"] = min(1.0, c["confidence"] + 0.1)
        ptotal = sum(prefs.values()) if prefs else 0
        for c in cands:
            tr = G.trust(c["how"], q) if G else 1.0
            bonus = 0.15 if hints & set(content_words(toks(str(c["answer"])))) else 0.0
            if ptotal >= 2 and c["facts"]:
                # выученное «где в факте лежит ответ» для этого типа вопроса: совпало → +, противоречит → −
                m = re.match(r"^(.*?) —(.*?)→ (.*)$", c["facts"][-1])
                if m:
                    X, rel, Y = m.groups(); aw = set(content_words(toks(str(c["answer"]))))
                    slot = "X" if aw and aw <= set(content_words(toks(X))) else ("Y" if aw and aw <= set(content_words(toks(Y))) else None)
                    if slot:
                        agree = sum(n for (sl, r_, _), n in prefs.items() if sl == slot and r_ == rel) / ptotal
                        bonus += 0.25 * agree - 0.1 * (1 - agree)
            c["score"] = round(c["confidence"] * tr + bonus, 3); c["trust"] = tr
        for c in cands:
            # САМОПРОВЕРКА: ответ подставляем обратно в вопрос и смотрим по памяти — (а) тип ответа совпадает с
            # «what kind of CELL» / «which ORGAN» (ответ должен быть определён как такой тип или по смыслу близок);
            # (б) ответ не повторяет слова вопроса; (в) ответ не «мусор» (одно служебное слово). Несовпадение — штраф, не запрет.
            chk = self._self_check(fr, c["answer"]); c["check"] = chk["note"]; c["score"] = round(c["score"] * chk["mult"], 3)
        # ОБУЧАЕМЫЙ АРБИТР (solver.py): признаки каждого кандидата → вероятность «верен», выученная из исправлений решением уравнения
        try:
            feats = self.solver.features(q, cands); self.solver.apply(cands, feats)
            # ДРУГИЕ КУСКИ того же предложения: в доску и в согласие не входят; размечаются исправлением (учат «какой кусок»),
            # а заменить основной отрывок могут ТОЛЬКО по вердикту solver'а: p_alt > p_main + запас — иначе поведение прежнее
            if alt_cands:
                for c in alt_cands: c["score"] = c["confidence"]; c["trust"] = 1.0; c["check"] = self._self_check(fr, c["answer"])["note"]
                fa = self.solver.features(q, alt_cands); cands_all = list(cands) + list(alt_cands)       # порядок = порядок признаков
                if self.solver.ready:
                    main = next((c for c in cands if c["how"].startswith("отрывок предложения")), None)
                    if main is not None:
                        pm = main.get("solver", 0.5); pa = self.solver.predict(fa)
                        for c, p_ in zip(alt_cands, pa): c["solver"] = round(p_, 2)
                        j = max(range(len(pa)), key=lambda k: pa[k])
                        if pa[j] > pm + float(os.environ.get("MINI_AI_ALT_MARGIN", "0.25")) and pa[j] >= 0.6:
                            alt_cands[j]["score"] = round(main["score"] * 1.01, 3); alt_cands[j]["how"] = "отрывок: другое предложение (по solver)"; cands.append(alt_cands[j])
                self._last_cands = (q, [dict(c) for c in cands_all], feats + fa)
            else: self._last_cands = (q, [dict(c) for c in cands], feats)
        except Exception as e_: self._last_cands = None; print("SOLVER-ERR", repr(e_)) if os.environ.get("MINI_AI_DEBUG") else None
        cands.sort(key=lambda c: -c["score"])
        # САМОПРОВЕРКА КАК НАГРАДА: согласие независимых способов / провал проверки типа → псевдометки solver'у без учителя (вес 0.3)
        try:
            lc = self._last_cands
            if lc and lc[0] == q: self._self_reward_note = self.solver.self_reward(q, lc[1], lc[2], cands[0])
        except Exception: pass
        best = cands[0]; best["board"] = [(c["how"], str(c["answer"])[:40], c["confidence"], c["trust"], c["score"], c.get("check"), (c.get("facts") or [""])[-1][:160]) for c in cands]
        return best

    def _self_check(self, fr, answer):
        a = str(answer); aw = set(content_words(toks(a))); mult = 1.0; notes = []
        if not aw: return dict(mult=0.3, note="empty")
        known = set(fr.get("X") or []) | set(fr.get("Y") or [])
        if aw <= known: mult *= 0.5; notes.append("repeats question")
        atype = fr.get("atype")
        if not atype:
            try: atype = self.deep._goal(self._cur_q)
            except Exception: atype = None
        if atype in ("place", "person") or (atype and (self._soft(atype) or atype in self.deep.VERBISH_T)): atype = None
        if atype and atype not in self.META and len(atype) > 2:
            # ответ определён как atype? — через граф deep (be-рёбра, определения «a sugar called X») или по смыслу
            try: self.deep._build(); tsc = self.deep._is_type(self.deep._node(a) or a.lower(), atype)
            except Exception: tsc = 0.5
            ok = tsc >= 0.7
            if ok: mult *= 1.15; notes.append(f"type {atype} ✓")
            else:
                # тип назван («what SUGAR»), а ответ не определён как такой тип и не близок по смыслу → сильный штраф;
                # кусок с глаголом («contains dna», «pairs») под тип не подходит никогда
                verbish = any(w.endswith(("s", "ed", "ing")) and len(w) > 4 and w not in self.kb.by_x for w in aw) and len(aw) <= 2
                mult *= 0.35 if verbish else 0.55; notes.append(f"type {atype} ✗")
        return dict(mult=mult, note=", ".join(notes) or "ok")

    def _two_step(self, fr, max_f1=60, max_t=8):
        known = set(fr["Y"] if fr["slot"] == "X" else fr["X"])
        known = known - self.META
        if len(known) < 2: return None
        idf = {w: self.kb.idf(w) for w in known}; kidf = sum(idf.values())
        cnt = collections.Counter()
        for w in known:
            for k in self.kb.by_word.get(w, []): cnt[k] += idf[w]
        best = None
        for k1, c in cnt.most_common(max_f1):
            c1 = c / kidf
            if not (0.4 <= c1 < 0.99): continue
            F1 = self.kb.facts[k1]; w1 = F1[0] | F1[2]; rest = known - w1
            # мост: редкий термин F1, который ОПРЕДЕЛЁН в памяти (есть факт с X = термин) — не любое слово
            bridges = [t for t in w1 - known if self.kb.idf(t) > 2.5 and t in self.kb.by_x]
            for t in sorted(bridges, key=lambda t: -self.kb.idf(t))[:max_t]:
                for k2 in self.kb.by_word.get(t, []):
                    if k2 == k1: continue
                    F2 = self.kb.facts[k2]; w2 = F2[0] | F2[2]
                    if not rest <= w2: continue
                    for F in (F2, F1):
                        for ans, aw in ((F[3], F[0]), (F[4], F[2])):
                            if aw and not (aw & known) and ans.lower() not in ("it", "this", "they", "these"):
                                s = c1 + sum(idf[w] for w in rest & w2) / kidf
                                if not best or s > best[0]: best = (s, ans, t, F1, F2)
        return best

    def ask(self, q, internet=True):
        # ОДНА МАТЕРИЯ (analogy.py, exp69/70): пример/объяснение → случай; задача с фактами → перенос случая; «нет, … потому что …» → исправление.
        # Стоит ПЕРЕД всем остальным: это разговор с учителем, не вопрос к памяти фактов.
        A = getattr(self, "analogy", None)
        if A and re.search(r"[а-яё]", q.lower()):
            qs = q.strip(); ql = qs.lower()
            R = lambda how, text, kind="answer", conf=1.0, **kw: dict(question=q, frame=None, read=[], answer=text, kind=kind, how=how, facts=[], confidence=conf, text=text, note=None, thoughts=kw.get("thoughts", []))
            try:
                # 1. исправление: «нет, X, потому что …» / «нет, любит — не старше»
                if re.match(r"^(?:нет|неверно|неправильно)\b", ql) and getattr(self, "_last_analogy", None):
                    task, res = self._last_analogy; say = A.correct(task, res, re.sub(r"^(?:нет|неверно|неправильно)[,\s]*", "", qs, flags=re.I))
                    self._last_analogy = None; return R("analogy: correction", say, "learned")
                # 2. объяснение/пример: есть вывод (значит/поэтому/то) или связь слов («X — то же, что Y») — и это не вопрос
                is_expl = not qs.endswith("?") and (A.CONCL_RX.search(qs) or re.match(r"^(?:если|когда|смотри|например|вот|пример)\b", ql) or re.search(r"то же,? что|означает|не\s+(?:про|то же)|≠", ql))
                if is_expl:
                    case, say = A.tell(re.sub(r"^(?:объясняю|правило|запомни|пойми)\s*[:\-—]?\s*", "", qs, flags=re.I))
                    return R("analogy: case" if case else "analogy: told", say, "learned" if case else "clarify", 1.0 if case else 0.5)
                # 3. задача: факты + вопрос
                if qs.endswith("?") or re.search(r"\bсколько\b|\bкто\b|\bгде\b", ql):
                    res = A.solve(qs)
                    if res:
                        self._last_analogy = (qs, res); out = R(f"analogy: {len(res['derived'])} переносов", A.say_steps(res), "answer", res["confidence"], thoughts=res["steps"]); out["answer"] = str(res["answer"]); return out
                    # задача с фактами есть, а переноса нет → честно: чего не хватает (а не мусор из слоя речи)
                    facts = [t for c, _ in A.clauses(" ".join(p for p in re.split(r"(?<=[.!?])\s+", qs) if not p.endswith("?"))) for t in A.triples(c)]
                    if len(facts) >= 2 and A.question(qs.split(". ")[-1]):
                        rels = sorted({t[1] for t in facts if t[2] != "#"} | {t[1].split()[-1] for t in facts if t[2] == "#"})
                        self._last_analogy = (qs, None)
                        return R("analogy: no case", f"Вижу связи «{'», «'.join(rels[:3])}», но у меня нет примера, как их переносить. Покажи один раз: «… — значит …», и я применю.", "clarify", 0.3)
            except Exception as e:
                print("ANALOGY-ERR", repr(e))

        if not self.use_dialog:
            out = self._ask_core(self._resolve(q), internet); out["kind"] = "answer" if out.get("answer") else "unknown"; return out
        out = self._ask_dialog(q, internet)
        if out.get("kind") in ("answer", "clarify", "unknown") and not (out.get("how") or "").startswith("growth"):
            self._last_out = dict(out, question=out.get("question") or q, _span_dbg=getattr(self, "_span_dbg", None))   # след чтения — для урока случая
        return out

    def _ask_core(self, q, internet=True, fr=None):
        self._cur_q = q; self._span_dbg = None
        try: a = self.cases.advice(q); self._advice = dict(a, q=q) if a else None
        except Exception: self._advice = None
        fr = fr or parse_question(q)
        out = dict(question=q, frame=fr, read=[])
        if internet and fr.get("atype") == "place" and fr["X"]:
            # «Where is X» — лучший ответ в первом предложении статьи о самом X; читаем её один раз
            title = " ".join(w.capitalize() for w in fr["X"])
            if title not in self.kb.sources:
                n = self._read_wiki(title)
                if n: out["read"].append(f"{title}: {n} фактов")
        # РАССУЖДЕНИЕ ЦЕПОЧКОЙ (think.py): вопрос с типом ответа («in which country», «who», «where») →
        # путь по фактам до узла нужного типа; показываем шаги. Одношаговый ответ допускается только если тип совпал.
        # FEW-SHOT (fewshot.py): правило вывода из показанного примера — применяется к новым сущностям
        fsr = self.fewshot.answer(q)
        if fsr:
            out.update(answer=fsr["answer"], how=f"few-shot: {fsr['schema']}", facts=fsr["facts"], confidence=fsr["confidence"],
                       thoughts=[f"You showed me: {fsr['example'][0]} → {fsr['example'][1]}.", f"Rule: follow {fsr['schema'].replace('inv:', 'reverse ')}."] + [f"I know: {f}" for f in fsr["facts"]] + [f"So: {fsr['answer']}."])
            out["text"] = f"By the rule you showed me ({fsr['schema'].replace('inv:', 'reverse ').replace('→', ' then ')}): " + " ".join(self._nice_fact(f) for f in fsr["facts"][:3]) + f" So the answer is {fsr['answer']}."; out["kind"] = "answer"
            self.history.append(dict(q=q, a=fsr["answer"], entity=fsr["answer"])); return out
        # ОПЕРАТОРЫ рассуждения (ops.py): сравнение, да/нет, счёт, список, почему, общее — над общей памятью
        op = self.ops.run(q)
        if op:
            if op.get("answer") is None and not op.get("thoughts"):
                pass                                                  # оператор понял вопрос, но фактов нет → дальше обычный путь (может переспросить)
            else:
                out.update(answer=op.get("answer"), how=f"op:{op['op']}", facts=[t.replace("I know: ", "") for t in op["thoughts"] if t.startswith("I know")], confidence=op.get("confidence", 0.0), thoughts=op["thoughts"], text=op["text"], kind=op["kind"])
                self.history.append(dict(q=q, a=out["answer"], entity=None)); return out
        # CoT (cot.py): мысли пишутся в рабочую память и перечитываются; несколько цепочек → голосование
        cot = self.cot.reason(q, internet=internet) if re.search(r"\b(?:which|what) (?:country|city|continent)\b|\bthe (?:capital|currency|largest city) of\b", q.lower()) else None
        if cot and cot.get("answer"):
            out.update(answer=cot["answer"], how=f"{len(cot['support'])} chain(s), {len(cot['thoughts'])} thoughts", facts=cot["support"], confidence=cot["confidence"], thoughts=cot["thoughts"])
            steps = [t for t in cot["thoughts"] if t.startswith("I know") or t.startswith("So ")]
            out["text"] = self._cot_text(steps, cot["answer"]) if len(steps) > 1 else None; out["kind"] = "answer"
            if not out["text"]:
                out.pop("thoughts", None)                        # прямой факт — пусть диалоговый слой скажет своими словами
                if cot["support"] and "—" in cot["support"][0] and not cot["support"][0].startswith("I know"): out["facts"] = cot["support"][:1]
                else:
                    sup = next((s_ for s_ in cot["support"] if s_.startswith("I know")), None)
                    if sup: mm = re.match(r"I know: (.*?) —(.*?)→ (.*?)\.?$", sup); out["facts"] = [f"{mm.group(1)} —be→ a {mm.group(2).replace(' in', '')} in {mm.group(3)}" if mm and mm.group(2).endswith(" in") else (f"{mm.group(1)} —{mm.group(2).replace(' ', '_')}→ {mm.group(3)}" if mm else sup)]
            self.history.append(dict(q=q, a=cot["answer"], entity=cot["answer"])); return out
        th = self.thinker.think(q) if re.search(r"\b(?:which|what) (?:country|city|continent|river|company|language|currency)\b", q.lower()) else None
        typed_q = bool(re.search(r"\b(?:which|what) (?:country|city|continent|river|company|language|currency)\b", q.lower()))
        if th:
            out.update(answer=th["answer"], how=f"{len(th['steps'])} шага: " + " → ".join(th["steps"]), facts=th["steps"], confidence=th["score"])
            out["text"] = self._phrase(out, fr); out["kind"] = "answer"
            self.history.append(dict(q=q, a=th["answer"], entity=th["answer"])); return out
        if typed_q and not th and self.thinker.anchors(q):
            # тип ответа задан, цепочки нет → честно: знаю сущность, но связи до нужного типа не нашла (не подменять поиском по словам)
            a = self.thinker.anchors(q)[0]
            out.update(answer=None, how="нет цепочки", facts=[], confidence=0.0, kind="unknown",
                       text=f"I know {a.title()}, but I can't trace it to a {self.thinker.goal(q)} from what I've read. Tell me the link and I'll remember.")
            return out
        # АРБИТР (growth.py): все способы думают ВСЕГДА, каждый даёт ответ с уверенностью; уверенность × доверие
        # по опыту (этот способ на этой форме вопроса попадал/промахивался) → побеждает лучший, не первый сработавший
        pick = self._arbiter(q, fr)
        r = None
        if pick and pick["how"].startswith("1 шаг"): r = pick["raw"]; out["arbiter"] = pick["board"]
        elif pick and pick["how"].startswith("2 шага"):
            out.update(answer=pick["answer"], how=pick["how"], facts=pick["facts"], confidence=pick["confidence"], arbiter=pick["board"])
        elif pick and (pick["how"].startswith(("отрывок предложения", "нейронный читатель", "комбинация"))):
            out.update(answer=pick["answer"], how=pick["how"], facts=pick["facts"], confidence=pick["confidence"], text=pick["text"], arbiter=pick["board"])
            self.history.append(dict(q=q, a=pick["answer"], entity=pick["answer"])); return out
        elif pick and ("(deep)" in pick["how"] or "(по абзацу)" in pick["how"]):
            out.update(answer=pick["answer"], how=pick["how"], facts=pick["facts"], confidence=pick["confidence"], text=pick["text"], thoughts=pick["thoughts"], arbiter=pick["board"], kind="answer")
            self.history.append(dict(q=q, a=pick["answer"], entity=pick["answer"])); return out
        if not r and not pick:
            if internet:
                # НЕ ЗНАЮ → ИДУ В ИНТЕРНЕТ: ищу статьи Википедии по словам вопроса, читаю, пробую снова
                for title in self._wiki_titles(fr):
                    n = self._read_wiki(title)
                    if n: out["read"].append(f"{title} (+{n} фактов)")
                    r = self._one_step(fr)
                    if r: break
                if not r and out["read"]:
                    r2 = self._two_step(fr)
                    if r2:
                        s, ans, t, F1, F2 = r2
                        out.update(answer=ans, how=f"прочитала в интернете, 2 шага через «{t}»", facts=[f"{F1[3]} —{F1[1]}→ {F1[4]}", f"{F2[3]} —{F2[1]}→ {F2[4]}"], confidence=round(min(1.0, s / 2.0), 2))
        if r:
            s, ans, fact = r
            how = "1 шаг" if not out["read"] else "прочитала в интернете, 1 шаг"
            out.update(answer=ans, how=how, facts=[f"{fact[0]} —{fact[1]}→ {fact[2]}"], confidence=round(min(1.0, s / 1.8), 2))
        if "answer" not in out: out.update(answer=None, how="не знаю" + (" (в интернете тоже не нашла)" if out["read"] else ""), facts=[], confidence=0.0)
        out["text"] = self._phrase(out, fr)
        ent = out["answer"] if out.get("answer") else (" ".join(fr["X"][:2]) if fr["X"] else None)
        if not self.use_dialog: self.history.append(dict(q=q, a=out["answer"], entity=ent))
        return out

    # ---------- exp40: слой диалога ----------
    def _ask_dialog(self, q, internet=True):
        from mini_ai.dialog import _is_question, entity_phrase, strip_greeting, YES, NO
        from mini_ai.speak import small_talk
        d = self.dialog; conv = d.conv; q0 = q.strip(); notes = []
        pend = conv.pending()
        # -1. РОСТ МЫШЛЕНИЯ (growth.py): исправление «no, it's X» / «wrong, X» / «the answer is X» после ответа →
        #     способ теряет доверие на этой форме вопроса, путь вопрос→X становится схемой, связка слов запоминается
        last = self._last_out
        if re.match(r"^(?:how (?:did|have) you (?:grown|grow|learned|improved)|what have you learned|growth report|show growth|как ты выросла)\s*\??$", q0, re.I):
            return dict(question=q0, frame=None, read=[], answer=None, kind="chat", how="growth: report", facts=[], confidence=1.0, text=self.growth.report() + "\n" + self.growth.report_gates() + "\n" + self.cases.report() + "\n" + self.solver.report(), note=None)
        # ПОЧЕМУ ТЫ ТАК ОТВЕТИЛА — объяснение реального пути: способ, факты, доска арбитра (кто ещё что предлагал)
        if re.match(r"^(?:why|why (?:did you (?:say|answer|think) that|that|so|not)|how do you know|how did you get that|explain|почему)\s*\??$", q0, re.I) and last:
            how = last.get("how") or "?"; facts = last.get("facts") or []; board = last.get("arbiter") or []
            lines = [f"I answered «{last.get('answer')}» by: {how}."]
            if last.get("thoughts"): lines.append("My steps: " + " ".join(last["thoughts"][:6]))
            elif facts: lines.append("Because I had read: " + "; ".join(self._nice_fact(f) for f in facts[:3]))
            if len(board) > 1: lines.append("Other ways proposed: " + "; ".join(f"{b[0]} → «{b[1]}» (score {b[4]})" for b in board[1:4]) + ".")
            if last.get("confidence") is not None: lines.append(f"My confidence was {last['confidence']}." + (" I wasn't sure — tell me if it's wrong and I'll learn why." if last["confidence"] < 0.6 else ""))
            return dict(question=q0, frame=None, read=[], answer=None, kind="chat", how="explain", facts=facts, confidence=1.0, text=" ".join(lines), note=None)
        mc = re.match(r"^(?:no|nope|wrong|incorrect|not that|нет|неверно|неправильно|не так)[,.!\s]*(?:it'?s|it is|the answer is|answer is|actually|это|правильно|ответ)?\s*[:\-—]?\s*(?P<a>[A-Za-zА-Яа-яЁё0-9][^?!.]{0,60})$", q0, re.I) \
             or re.match(r"^(?:the answer is|answer:|correct answer is|правильно|правильный ответ)\s*[:\-—]?\s*(?P<a>[A-Za-zА-Яа-яЁё0-9][^?!.]{0,60})$", q0, re.I)
        if mc and re.search(r"[а-яё]", mc.group("a").lower()) and getattr(self, "russian", None):
            # русская поправка «нет, это кислород» → переводим ответ на английский (язык фактов); если слова нет в словаре — учим пару позже
            en_a, _, unk_a = self.russian.to_en(mc.group("a"))
            en_a = re.sub(r"^(?:what|which|who|is|are|the|a|an)\s+", "", en_a.rstrip("?")).strip()
            if en_a and not unk_a: mc = re.match(r"(?P<a>.+)", en_a)
            elif unk_a and last and not last.get("_graded"):
                last["_graded"] = True; conv.push(q=q0, frame=None, entity=None, answer=None, kind="chat", pending=None, key=None)
                return dict(question=q0, frame=None, read=[], answer=None, kind="chat", how="russian: unknown word", facts=[], confidence=1.0,
                            text="Хочу запомнить поправку, но не знаю слов: " + ", ".join(unk_a) + ". Скажи «по-русски <слово> значит <english>» — и повтори поправку.", note=None)
        if mc and last and last.get("question") and not last.get("_graded") and not (pend and pend.get("options")):
            last["_graded"] = True
            right = mc.group("a").strip(" .")
            if last.get("answer") and set(content_words(toks(right))) and set(content_words(toks(right))) <= set(content_words(toks(str(last["answer"])))):
                # «no, it's oxygen» — а она и сказала oxygen → это подтверждение, не исправление
                self.growth.learn_from_confirmation(last["question"], last)
                try:
                    lc = self._last_cands
                    if lc and lc[0] == last["question"]: self.solver.learn(lc[0], lc[1], right, lc[2])
                except Exception: pass
                conv.push(q=q0, frame=None, entity=None, answer=None, kind="chat", pending=None, key=None)
                return dict(question=q0, frame=None, read=[], answer=None, kind="chat", how="growth: confirmation", facts=[], confidence=1.0, text=f"That's what I said — {right}. Good, I'll trust that way more.", note=None)
            grew = self.growth.learn_from_correction(last["question"], last, right)
            try:
                rg = self.reform.learn(last["question"], right)
                if rg: grew.append("how to read it: " + "; ".join(rg[:3]))
            except Exception as e: grew.append(f"reform error {e}")
            try:
                lc = self._last_cands
                if lc and lc[0] == last["question"]: self.solver.learn(lc[0], lc[1], right, lc[2])
            except Exception as e: grew.append(f"solver error {e}")
            try:
                it = self.cases.learn(last["question"], right, last)
                if it: grew.append("case remembered: " + ", ".join(f"{w} {'needed' if r == 'needed' else 'extra'}" for w, r in list(it["roles"].items())[:4]))
            except Exception as e: grew.append(f"case error {e}")
            conv.push(q=q0, frame=None, entity=right, answer=right, kind="learned", pending=None, key=None)
            diag = next((g for g in grew if g.startswith("diagnosis")), "")
            why = {"diagnosis: silent": "I had it in memory but didn't dare to answer", "diagnosis: wrong-piece": "I found the right sentence but took the wrong part",
                   "diagnosis: wrong-fact": "I picked the wrong fact", "diagnosis: no-sentence": "I had never read that — now I know"}.get(diag, "")
            txt = f"Got it — {right}. " + (f"Why I was wrong: {why}. " if why else "") + ("What I learned: " + "; ".join(g for g in grew if not g.startswith("diagnosis"))[:400] + "." if grew else "I'll remember that.")
            return dict(question=q0, frame=None, read=[], answer=right, kind="learned", how="growth: correction", facts=[], confidence=1.0, text=txt, note=None)
        if last and last.get("answer") and not last.get("_graded") and (YES.match(q0) or re.match(r"^(?:right|correct|exactly|yes that'?s right|верно|правильно)[\s.!]*$", q0, re.I)) and not (pend and pend.get("options")):
            last["_graded"] = True
            self.growth.learn_from_confirmation(last["question"], last)
            try: self.cases.learn(last["question"], str(last["answer"]), last, hit=True)      # попадание — тоже случай: способ и позиция верны
            except Exception: pass
            try:
                lc = self._last_cands
                if lc and lc[0] == last["question"]: self.solver.learn(lc[0], lc[1], str(last["answer"]), lc[2])
            except Exception: pass
            try: self.reform.learn(last["question"], str(last["answer"]), confirmed=True)
            except Exception: pass
            conv.push(q=q0, frame=None, entity=None, answer=None, kind="chat", pending=None, key=None)
            return dict(question=q0, frame=None, read=[], answer=None, kind="chat", how="growth: confirmation", facts=[], confidence=1.0, text="Good — I'll trust that way of thinking more for questions like this.", note=None)
        # 0. реплика-собеседник (hi / thanks / how are you / who are you) — без поиска фактов
        st = small_talk(q0, conv.rng, len(self.kb.facts))
        is_ru = bool(re.search(r"[а-яё]", q0.lower()))
        # РУССКИЙ: «по-русски дом значит house» → пара в словарь; русский вопрос → перевод → ядро → ответ по-русски
        R = getattr(self, "russian", None)
        # АРИФМЕТИКА: «1+1», «сколько будет 7*8», «what is 12/4» → считаем, не ищем (любой язык)
        ma = re.search(r"(-?\d+(?:[.,]\d+)?(?:\s*[-+*/×÷:^]\s*-?\d+(?:[.,]\d+)?)+)", q0.replace("х", "*").replace("x", "*"))
        if ma and re.fullmatch(r"[\d\s.,+\-*/×÷:^()]+", ma.group(1)):
            expr = ma.group(1).replace(",", ".").replace("×", "*").replace("÷", "/").replace(":", "/").replace("^", "**")
            try:
                val = eval(expr, {"__builtins__": {}}, {})
                val = int(val) if isinstance(val, float) and val.is_integer() else (round(val, 6) if isinstance(val, float) else val)
                conv.push(q=q0, frame=None, entity=str(val), answer=str(val), kind="answer", pending=None, key=None)
                return dict(question=q0, frame=None, read=[], answer=str(val), kind="answer", how="арифметика", facts=[f"{expr.strip()} = {val}"], confidence=1.0,
                            text=(f"{expr.strip()} = {val}" if not is_ru else f"{expr.strip()} = {val}."), note=None)
            except Exception: pass
        if is_ru and R is not None:
            mt = re.match(r"^по-русски\s+([а-яё\-]+)\s+(?:значит|это|=|—|-)\s+([a-z\-]+)\s*\.?$", q0.strip().lower())
            if mt:
                R.learn_pair(mt.group(1), mt.group(2)); conv.push(q=q0, frame=None, entity=None, answer=None, kind="chat", pending=None, key=None)
                return dict(question=q0, frame=None, read=[], answer=None, kind="learned", how="russian: pair", facts=[], confidence=1.0, text=f"Запомнила: «{mt.group(1)}» = {mt.group(2)}.", note=None)
            # приветствие/спасибо — короткий честный ответ без фактов (закрытая таблица, не говорилка)
            if st and st[0] in ("greet", "thanks", "how_are_you", "bye"):
                conv.push(q=q0, frame=None, entity=None, answer=None, kind="chat", pending=None, key=None)
                ru_st = {"greet": "Привет. Спроси меня о чём-нибудь — я отвечаю по фактам, которые прочитала, и учусь на твоих поправках.",
                         "thanks": "Пожалуйста.", "how_are_you": f"Работаю. В памяти {len(self.kb.facts):,} фактов; учусь на каждом «нет, это…».".replace(",", " "), "bye": "До встречи."}
                return dict(question=q0, frame=None, read=[], answer=None, kind="chat", how="small talk: " + st[0], facts=[], confidence=1.0, text=ru_st.get(st[0], st[1]), note=None)
            # СОБЕСЕДНИК (converse.py): «лучше сказать так: …» → учится говорить; память о тебе; арифметика; схема рассуждения с шагами
            C = getattr(self, "converse", None)
            if C is not None:
                mb = re.match(r"^(?:лучше\s+(?:сказать\s+)?(?:так|ответить так)|скажи так|лучше так|правильнее так)\s*[:\-—]\s*(?P<b>.{3,})$", q0.strip(), re.I)
                if mb and getattr(self, "_last_ru", None):
                    inp0, out0, cid = self._last_ru
                    C.correct(inp0, out0, mb.group("b").strip(), cid); C.save()
                    conv.push(q=q0, frame=None, entity=None, answer=None, kind="chat", pending=None, key=None)
                    return dict(question=q0, frame=None, read=[], answer=None, kind="learned", how="речь: исправление", facts=[], confidence=1.0, text="Запомнила, в следующий раз скажу так.", note=None)
                cr = C.respond(q0)
                if cr and (cr["how"].startswith(("память", "арифметика", "схема")) or not (_is_question(q0) and re.search(r"\b(?:что|кто|где|когда|почему|как|какой|какая|какое|какие|сколько|чем|кем|кого|зачем)\b", q0.lower()))):
                    # фактические вопросы («что такое…», «какая столица…») — ядру; остальное (разговор, о себе, задачи) — собеседнику
                    self._last_ru = (q0, cr["text"], cr.get("case_id")); C.save() if C.stats.get("cases_learned", 0) % 10 == 0 else None
                    conv.push(q=q0, frame=None, entity=None, answer=cr["text"], kind="chat", pending=None, key=None)
                    return dict(question=q0, frame=None, read=[], answer=None, kind="chat", how=cr["how"], facts=[], confidence=cr.get("confidence", 0.5), text=cr["text"], thoughts=cr.get("thoughts"), note=None)
            # ВСЁ ОСТАЛЬНОЕ ПО-РУССКИ → перевод → ядро. Никакой говорилки: если ядро не знает — говорит честно, что поняла и чего не нашла.
            en_q, rep, unknown = R.to_en(q0)
            core = self._ask_core(en_q, internet) if len(toks(en_q)) >= 2 else dict(answer=None, how="не знаю", facts=[])
            conv.push(q=q0, frame=None, entity=core.get("answer"), answer=core.get("answer"), kind="answer" if core.get("answer") else "unknown", pending=None, key=None)
            out_ru = dict(question=q0, frame=core.get("frame"), read=[], answer=core.get("answer"), kind="answer" if core.get("answer") else "unknown",
                          how="russian→en: " + (core.get("how") or "?"), facts=core.get("facts") or [], confidence=core.get("confidence", 0.0), text=R.say(core, en_q, rep, unknown),
                          thoughts=core.get("thoughts"), arbiter=core.get("arbiter"), note="translated: " + en_q)
            self._last_out = dict(out_ru, question=en_q, _span_dbg=getattr(self, "_span_dbg", None))     # чтобы «нет, это X» после русского вопроса училось так же
            if C is not None and not core.get("answer"):
                # ядро не нашло — пусть ответит собеседник (случай речи), а не сухое «не нашла»
                cr = C.respond(q0)
                if cr and cr.get("confidence", 0) >= 0.5:
                    self._last_ru = (q0, cr["text"], cr.get("case_id")); out_ru.update(text=cr["text"], how=cr["how"], kind="chat")
            return out_ru
        # АНГЛИЙСКИЙ СОБЕСЕДНИК: то же, что по-русски — «better say: …» учит речь; о себе; арифметика; реплики без фактического вопроса
        C = getattr(self, "converse", None)
        if C is not None and not is_ru and not (pend and (pend.get("options") or YES.match(q0) or NO.match(q0))):
            mb = re.match(r"^(?:better(?: to)? say|say it like this|better like this|rather say)\s*[:\-—]\s*(?P<b>.{3,})$", q0.strip(), re.I)
            if mb and getattr(self, "_last_ru", None):
                inp0, out0, cid = self._last_ru
                C.correct(inp0, out0, mb.group("b").strip(), cid); C.save()
                conv.push(q=q0, frame=None, entity=None, answer=None, kind="chat", pending=None, key=None)
                return dict(question=q0, frame=None, read=[], answer=None, kind="learned", how="речь: исправление", facts=[], confidence=1.0, text="Got it — I'll say it that way next time.", note=None)
            factual = _is_question(q0) and re.search(r"\b(?:what|which|who|where|when|why|how many|how much|how does|how do|how did|name)\b", q0.lower()) and not re.search(r"\b(?:my|me|i|am i|do i|you|your)\b", q0.lower())
            if not factual and not re.match(r"^(?:example:|learn |выучи|/)", q0.lower()):
                cr = C.respond(q0)
                if cr and (cr["how"].startswith(("память", "арифметика", "схема")) or cr.get("confidence", 0) >= 0.45 or st):
                    self._last_ru = (q0, cr["text"], cr.get("case_id"))
                    conv.push(q=q0, frame=None, entity=None, answer=cr["text"], kind="chat", pending=None, key=None)
                    return dict(question=q0, frame=None, read=[], answer=None, kind="chat", how=cr["how"], facts=[], confidence=cr.get("confidence", 0.5), text=cr["text"], thoughts=cr.get("thoughts"), note=None)
        if st and not (pend and YES.match(q0)):
            conv.push(q=q0, frame=None, entity=None, answer=None, kind="chat", pending=None, key=None)
            return dict(question=q0, frame=None, read=[], answer=None, kind="chat", how="small talk: " + st[0], facts=[], confidence=1.0, text=st[1], note=None)
        q0 = strip_greeting(q0)
        # 1. реплика после переспроса: yes/no/выбор варианта/утверждение-обучение
        if pend:
            fact, note = d.answer_pending(q0, pend)
            if fact:
                X, rel, Y, src = fact
                return self._reply_fact(q0, X, rel, Y, src, 1.0, how="из переспроса: " + note, facts=[f"{X} —{rel}→ {Y}"], frame=None)
            if note in ("отказ от вариантов", "класс сущности", "подтверждение, но вариантов нет"):
                name = pend.get("name") or "it"
                text = (f"Then tell me one sentence about {name} and I'll remember it." if note != "класс сущности"
                        else f"Okay. Tell me one sentence about {name} and I'll remember it.")
                conv.push(q=q0, frame=None, entity=pend.get("entity"), answer=None, kind="clarify", pending=dict(pend, options=[]), key=None)
                return dict(question=q0, frame=None, read=[], answer=None, kind="clarify", how=note, facts=[], confidence=0.0, text=text, note=None)
        if YES.match(q0) or NO.match(q0) or len(toks(q0)) == 0:
            conv.push(q=q0, frame=None, entity=None, answer=None, kind="chat", pending=None, key=None)
            return dict(question=q0, frame=None, read=[], answer=None, kind="chat", how="реплика без вопроса", facts=[], confidence=1.0, text="Okay. What would you like to ask?", note=None)
        # 2. утверждение (не вопрос) → выучить и сразу ответить тем, что поняла
        if not _is_question(q0) and len(q0.split()) >= 3:
            facts = [f for f in self.kb.reader.read(q0, "выучено")[0] if "@" not in f[1]]
            if facts:
                self.learn(q0)
                X, rel, Y = facts[0]
                r = self._reply_fact(q0, X, rel, Y, "выучено", 1.0, how=f"выучила {len(facts)} факт(а)", facts=[f"{X} —{rel}→ {Y}"], frame=None, learned=True)
                r["learned"] = len(facts); return r
        # 2b. одно имя без вопроса («Tashkent?») → что именно интересует
        if d.bare_entity(q0):
            c = d.clarify(q0, None)
            if c["kind"] == "clarify":
                conv.push(q=q0, frame=None, entity=c["pending"]["name"], answer=None, kind="clarify", pending=c["pending"], key=None)
                return dict(question=q0, frame=None, read=[], answer=None, kind="clarify", how="имя без вопроса → переспрос", facts=[], confidence=0.0, text=c["text"], note=None)
        # 3. нить: эллипсис и местоимения — кандидаты по очереди, берётся первый с ответом
        fr = parse_question(q0); out = None
        cands_ = list(d.resolve(q0, fr))
        if cands_ and cands_[-1][0] != q0 and cands_[-1][2]: cands_.append((q0, fr, None))      # подстановка не помогла → исходный вопрос как есть
        for q1, fr1, note in cands_:
            if fr1 is None and q1 != q0: fr1 = parse_question(q1)
            if fr1 is None: fr1 = fr
            out = self._ask_core(q1, internet, fr=fr1)
            if out.get("answer"): break
        if note: notes.append(note)
        if out.get("answer") and out["how"].startswith("2 шага") and out["confidence"] < 0.6:
            g = d.guess(q1, out)
            if g:
                conv.push(q=q1, frame=fr1, entity=g["pending"]["name"], answer=None, kind="clarify", pending=g["pending"], key=None)
                return dict(question=q1, frame=fr1, read=out["read"], answer=None, kind="clarify", how="слабая цепочка → догадка-переспрос", facts=out["facts"], confidence=out["confidence"], text=g["text"], note=", ".join(notes) or None)
            out["answer"] = None                                            # мост через случайное слово — не ответ
        if out.get("thoughts") is not None and (out.get("how", "").startswith("op:") or out.get("how", "").startswith("few-shot")) and out.get("text"):
            conv.push(q=q1, frame=fr1, entity=None, answer=out.get("answer"), kind=out["kind"], pending=None, key=None)
            out["question"] = q1; out["note"] = ", ".join(notes) or None; return out
        if out.get("answer") and (out.get("how") or "").startswith(("отрывок предложения", "нейронный читатель", "комбинация")):
            conv.push(q=q1, frame=fr1, entity=out["answer"], answer=out["answer"], kind="answer", pending=None, key=None)
            out["question"] = q1; out["kind"] = "answer"; out["note"] = ", ".join(notes) or None; return out
        if out.get("answer") and out.get("thoughts"):
            # ответ получен рассуждением (CoT): текст — мысли как есть, диалоговый слой не перефразирует
            conv.push(q=q1, frame=fr1, entity=out["answer"], answer=out["answer"], kind="answer", pending=None, key=None)
            out["question"] = q1; out["note"] = ", ".join(notes) or None; return out
        if out.get("answer"):
            X, rel, Y, src = self._fact_of(out)
            r = self._reply_fact(q1, X, rel, Y, src, out["confidence"], how=out["how"], facts=out["facts"], frame=fr1, read=out["read"], two_step=out if len(out["facts"]) > 1 else None)
            r["question"] = q1; r["note"] = ", ".join(notes) or None; return r
        # 4. не знаю → переспрос из того, что понято
        c = d.clarify(q1, fr1)
        ents = d.entities_in(q1)
        conv.push(q=q1, frame=fr1, entity=(c["pending"] or {}).get("name"), answer=None, kind=c["kind"], pending=c["pending"], key=None)
        return dict(question=q1, frame=fr1, read=out["read"], answer=None, kind=c["kind"], how="не знаю → " + ("переспрос" if c["kind"] == "clarify" else "уточнение"),
                    facts=[], confidence=0.0, text=c["text"], note=", ".join(notes) or None)

    def _nice_fact(self, f):
        m = re.match(r"^(.*?) —(.*?)→ (.*)$", f)
        if not m: return f
        X, rel, Y = m.groups(); v = {"be": "is", "have": "has"}.get(rel, rel.replace("_", " "))
        return f"{X} {v} {Y.rstrip('.')}."

    def show_example(self, q, a):
        """«Example: Who is the grand-teacher of Carl? → Ann»: выучить правило вывода из одного примера"""
        r = self.fewshot.show(q, a)
        if not r: return "I can't find a path from the question to that answer in what I know — tell me the facts first."
        return f"Got the rule: for «{self.fewshot.qkey(q)}» follow {'→'.join(r[0]).replace('inv:', 'reverse ')}. I'll apply it to new names."

    def _cot_text(self, steps, answer):
        """мысли → речь: «X —rel→ Y» → «X is in Y», без служебных стрелок; последняя цепочка + вывод"""
        def nice(t):
            t = re.sub(r"\s*\(because.*\)\.?$", "", t)
            t = re.sub(r"—born in country→|—born in continent→", "was born in", t); t = re.sub(r"—company in country→|—company in continent→|—company in→", "is based in", t)
            t = re.sub(r"—capital in continent→|—city in continent→", "is in", t); t = re.sub(r"—born in→", "was born in", t)
            t = re.sub(r"—capital of→", "is the capital of", t); t = re.sub(r"—city in→", "is a city in", t); t = re.sub(r"—country in→", "is a country in", t)
            t = re.sub(r"—largest city of→", "is the largest city of", t); t = re.sub(r"—currency of→", "is the currency of", t)
            t = re.sub(r"—(\w[\w ]*)→", lambda m: m.group(1).replace("_", " "), t); return t.rstrip(".") + "."
        facts = [nice(t)[len("I know: "):] for t in steps if t.startswith("I know")][:3]
        concl = [nice(t) for t in steps if t.startswith("So ") and answer.lower() in t.lower()]
        return "Let me think. " + " ".join(facts) + (" " + concl[-1] if concl else "") + f" So: {answer}."

    def _fact_of(self, out):
        m = re.match(r"^(.*?) —(.*?)→ (.*)$", out["facts"][-1] if len(out["facts"]) > 1 else out["facts"][0])
        X, rel, Y = m.groups() if m else (out["answer"], "be", "")
        src = None
        for k in self.kb.by_x.get(" ".join(toks(X)), []):
            f = self.kb.facts[k]
            if f[1] == rel and f[4] == Y: src = f[5]; break
        return X, rel, Y, src

    def _reply_fact(self, q, X, rel, Y, src, confidence, how, facts, frame, read=None, two_step=None, learned=False):
        """факт → реплика: форма по контексту (повтор → другая + «As I said»), инициатива — ещё факт/встречный вопрос"""
        d = self.dialog; conv = d.conv
        key = f"{X.lower()}|{rel}|{Y.lower()}"; qkey = " ".join(toks(q))
        pron, etype = d.persona(X)
        tense = "past" if (etype == "person" and rel not in ("be", "have")) else "present"
        if two_step: text, style = two_step["text"], "two-step"
        else:
            if learned: text, style = "Got it. " + __import__("mini_ai.speak", fromlist=["phrase"]).phrase(X, rel, Y, "direct", tense), "learned"
            else: text, style = d.say(X, rel, Y, src, key, qkey, confidence, tense)
        answer = Y if (frame and frame.get("slot") == "Y") or (frame is None and conv.last and conv.last.get("pending") and conv.last["pending"].get("want")) else X
        if frame and frame.get("slot") == "X": answer = X
        elif frame and frame.get("slot") == "Y": answer = Y
        elif learned: answer = X
        ini = None if learned else d.initiative(X, rel, Y, set(toks(answer)))
        ini_pend = None
        if ini: text = text + " " + ini[0]; ini_pend = ini[1]; ini = ini[0]
        ent = X if X.lower() not in ("it", "this", "they") else conv.last_entity()
        from mini_ai.speak import split_prop
        sp = split_prop(Y) if rel == "be" else None
        topic = sp[1] if sp and frame and frame.get("slot") == "X" else None            # «capital of Kazakhstan» → тема Kazakhstan
        conv.push(q=q, frame=frame, entity=ent, topic=topic, answer=answer, kind="answer", style=style, pending=ini_pend, key=key, qkey=qkey)
        self.history.append(dict(q=q, a=answer, entity=ent))
        return dict(question=q, frame=frame, read=read or [], answer=answer, kind="answer", how=how, facts=facts, confidence=confidence,
                    text=text, style=style, initiative=ini, note=None)

    def _phrase(self, out, fr):
        """ответ СВОИМИ СЛОВАМИ: из факта (X, rel, Y) собирается предложение по грамматике связи (speak.py),
        а не копируется исходный текст. Для 2 шагов — два предложения: мост и вывод."""
        from mini_ai.speak import verb, clean_y
        if not out.get("answer"): return None
        facts = []
        for f in out["facts"]:
            m = re.match(r"^(.*?) —(.*?)→ (.*)$", f)
            if m: facts.append(m.groups())
        sents = []
        for X, rel, Y in facts:
            y = clean_y(Y) or Y.strip(" ,.")
            if rel.startswith("be_called"): s = f"{X} is called {y}"
            elif "_" in rel and not rel.startswith("be_"): s = f"The {rel.replace('_', ' ')} of {X} is {y}"
            else:
                v = verb(rel, "present") or rel
                last = X.split()[-1].lower()
                plural = (last.endswith("s") and not last.endswith(("ss", "us", "is")) and (len(X.split()) > 1 or X[:1].islower())) or last in ("people", "children", "men", "women")
                if plural: v = {"is": "are", "has": "have"}.get(v, rel)
                s = f"{X} {v} {y}"
            sents.append(s[0].upper() + s[1:].rstrip(".") + ".")
        ans = out["answer"]
        if len(sents) == 1: return sents[0]
        return " ".join(sents[:2]) + f" So the answer is {ans}."

    # ---------- интернет: Википедия по словам вопроса ----------
    def _wiki_titles(self, fr, k=2):
        """кандидаты-статьи: имена с заглавной из вопроса → поиск Википедии по содержательным словам"""
        try:
            from tools.exp24_wiki_crawler import api
        except Exception: return []
        words = (fr["Y"] or []) + (fr["X"] or [])
        query = " ".join(w for w in words if w not in ("capital", "largest", "biggest", "called", "name", "type", "kind"))[:80]
        if not query.strip(): return []
        try:
            d = api(dict(action="query", list="search", srsearch=query, srlimit=k))
            return [h["title"] for h in d["query"]["search"]]
        except Exception: return []

    def _read_wiki(self, title):
        if title in self.kb.sources: return 0
        try:
            from tools.exp24_wiki_crawler import fetch_article, clean_extract
            real, text, _ = fetch_article(title)
            if real in self.kb.sources: return 0
            n = self.kb.read_text(clean_extract(text)[:20000], real); self.meaning.read(clean_extract(text)[:20000]); self.meaning.save()
            with open(os.path.join(self.state_dir, "learned.txt"), "a") as f: f.write(f"\n# wiki: {real}\n" + clean_extract(text)[:20000] + "\n")
            return n
        except Exception: return 0

    # ---------- речь своими словами ----------
    def tell(self, X):
        key = " ".join(toks(X)); ks = self.kb.by_x.get(key, [])
        if not ks:                                                    # частичное совпадение по первому слову
            ks = [k for k in self.kb.by_word.get(toks(X)[0], []) if self.kb.facts[k][0] and toks(X)[0] in self.kb.facts[k][0]][:20] if toks(X) else []
        fs = [(self.kb.facts[k][3], self.kb.facts[k][1], self.kb.facts[k][4]) for k in ks]
        read = []
        if len(fs) < 3:                                              # мало знаю → читаю статью в интернете
            n = self._read_wiki(X)
            if n: read.append(f"{X} (+{n} фактов)"); ks = self.kb.by_x.get(key, []); fs = [(self.kb.facts[k][3], self.kb.facts[k][1], self.kb.facts[k][4]) for k in ks]
        if not fs: return dict(text=None, facts=0, how="не знаю", read=read)
        self.history.append(dict(q=f"tell {X}", a=None, entity=X))
        return dict(text=compose(X, fs), facts=len(fs), how="из фактов" + (" (прочитала в интернете)" if read else ""), read=read)

    # ---------- обучение ----------
    def learn(self, text, src="выучено"):
        n = self.kb.read_text(text, src); self.meaning.read(text); self.meaning.save()
        with open(os.path.join(self.state_dir, "learned.txt"), "a") as f: f.write(text.strip() + "\n")
        return dict(new_facts=n, total=len(self.kb.facts))

    def word(self, w):
        """что она сама думает о слове — из контекстов, без словаря"""
        return self.meaning.explain(w)

    # ---------- действия и код ----------
    def act(self, request, check=None):
        if self._agent is None:
            from mini_ai.agent import Agent; self._agent = Agent(os.path.join(self.state_dir, "agent"))
        if check is None: check = lambda out: bool(out and str(out).strip() and "rror" not in str(out)[:40])   # без проверки: непустой вывод без ошибки
        out, how = self._agent.run(request, check); self._agent.save()
        return dict(output=out, how=how)

    def code(self, task, fname, tests):
        if self._coder is None:
            from mini_ai.coder import CodeMemory, Coder
            cm = CodeMemory(os.path.join(self.state_dir, "code_memory.json"))
            tb = os.path.join(ROOT, "data/code/textbook.py")
            if os.path.exists(tb) and not cm.items: cm.learn_from_code(open(tb).read(), "textbook")
            self._coder = Coder(cm)
        r = self._coder.solve(task, fname, tests); self._coder.mem.save(); return r
