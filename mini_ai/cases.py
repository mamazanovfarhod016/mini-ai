"""cases.py — РОСТ ЧЕРЕЗ СЛУЧАИ (case-based reasoning), без градиента.

Абляция показала: счётчики доверия/смелости/внимания на 100–200 исправлениях — шум (±1 %). Они меняют «кому верить»,
а ошибки — в «где искать и что взять». Здесь каждое исправление (и подтверждение) запоминается как СЛУЧАЙ:
    вопрос  →  предложение памяти с ответом  →  как ответ стоит в нём (позиция, соседи, тип разбора)
                                           →  УРОК: почему промахнулась (след чтения _span в момент ответа)
Уроки (поле lesson):
    silent-thr   — предложение с ответом БЫЛО в топ-12 чтения, но покрытие ниже порога → молчала зря
    silent-rare  — оно было в топе, порог прошло, но «не хватало редкого слова вопроса» → молчала зря
    wrong-piece  — предложение то, кусок не тот
    wrong-sent   — предложение с ответом было видно, но выбрано другое
    no-sent      — предложения с ответом в памяти нет (учиться тут нечему — только запомнить факт)
    hit          — подтверждение «yes»: способ и позиция верные
Новый вопрос ищет ближайшие случаи — БЕЗ нейросети: скелет вопроса, грубая форма (qcoarse), смысл слов
(meaning.phrase_sim по 68k векторам), пересечение слов — и ПОВТОРЯЕТ их способ чтения:
  1. «смелость»: среди похожих случаев молчание было ошибкой (silent-*) → порог покрытия ниже, одно редкое слово прощается;
     среди похожих брала не тот кусок / не то предложение → строже;
  2. «где искать»: какие слова вопроса были в ответном предложении, а какие лишние → слова той же роли мягкие;
  3. «что взять»: соседи ответа слева/справа; 4. «каким способом»: бонус способу, которым решались похожие.
Это не схема «вопрос-форма → слот» (их 90 на класс и они не переносятся), а перенос по СМЫСЛУ вопроса целиком.
"""
import re, json, os, time, collections
import numpy as np
from mini_ai.memory_v2 import toks, content_words

STOP = set("the a an of in on at to for by with from and or is are was were be been do does did what which who whom whose how why when where that this these those it its their there as into than then so such".split())
SHAPE = ("called", "known", "term", "type", "kind", "name", "part", "process", "example")

def skeleton(q):
    """скелет вопроса: содержательные слова → «_», служебные остаются. «What do most living things use to make ATP?» → «what do _ _ _ use to make _»"""
    t = toks(q); out = []
    for w in t:
        out.append(w if w in STOP or w in SHAPE else "_")
    return " ".join(out)

def coarse(q):
    """грубая форма — как growth.qcoarse, но локально (чтобы cases не зависел от growth): первые 2–3 слова скелета / «_ called what»"""
    f = skeleton(q).split()
    if not f: return "?"
    if f[-1] == "what" and "called" in f: return "_ called what"
    if "called" in f: return "what _ called"
    head = f[:3] if f[0] in ("what", "which", "who", "where", "when", "how", "why", "in", "on", "the", "_") else f[:2]
    return " ".join(head)

class Cases:
    SILENT = ("silent-thr", "silent-rare")                      # молчала, а предложение с ответом было в топе чтения → смелость помогла бы
    SILENT_ANY = ("silent-thr", "silent-rare", "silent-far", "no-sent-silent")   # все случаи молчания (знаменатель)
    STRICT = ("wrong-piece", "wrong-sent")
    def __init__(self, mind, state_dir=None):
        self.mind = mind; self.path = os.path.join(state_dir, "cases.json") if state_dir else None
        self.items = []            # dict(q, right, sent, skel, coarse, roles{word: needed|extra}, pos, how, lesson, gcov, vec(list)|None)
        self._vecs = None; self._stamp = -1
        self.load()

    # ---------- память случаев ----------
    def load(self):
        if not self.path or not os.path.exists(self.path): return
        try: self.items = json.load(open(self.path))[-2000:]
        except Exception: self.items = []
        vp = self.path.replace(".json", "_vecs.npy")
        if os.path.exists(vp):                                                    # смысл-векторы отдельно (npy), в json их нет
            try:
                V = np.load(vp); j = 0
                for it in self.items:
                    if it.get("vec") == "npy" and j < len(V): it["vec"] = V[j].tolist(); j += 1
            except Exception: pass
        self._since_save = 0
    def save(self, force=False):
        """Профиль школы (23.10): json.dump cases.json с 4096-мерными векторами = 1.8 с на каждое исправление (60 % шага учёбы).
        Теперь: векторы — в отдельный .npy, json — без них; писать раз в 10 уроков (и при force)."""
        if not self.path: return
        self._since_save = getattr(self, "_since_save", 0) + 1
        if not force and self._since_save < 10: return
        items = self.items[-2000:]; vecs = [it["vec"] for it in items if isinstance(it.get("vec"), list)]
        light = [dict(it, vec=("npy" if isinstance(it.get("vec"), list) else it.get("vec"))) for it in items]
        json.dump(light, open(self.path, "w"), ensure_ascii=False)
        if vecs: np.save(self.path.replace(".json", "_vecs.npy"), np.array(vecs, dtype=np.float16))
        self._since_save = 0

    def _encode(self, texts):
        if os.environ.get("MINI_AI_NEURAL", "1") == "0": return None
        try:
            from mini_ai.neural import _model
            return _model().encode(texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        except Exception: return None

    def _qwords(self, q): return [w for w in content_words(toks(q)) if w not in STOP]

    def _find_sent(self, q, right):
        """предложение памяти с RIGHT, покрывающее больше всего слов вопроса → (si, sent, sforms, cov) | None"""
        kb = self.mind.kb; rl = right.lower(); qw = self._qwords(q); best = None
        for si in range(len(kb.sents) - 1, max(-1, len(kb.sents) - 40000), -1):
            sent = kb.sents[si][0]
            if rl not in sent.lower(): continue
            sw = set(content_words(toks(sent))); sf = set().union(*[self.mind._forms(x) for x in sw]) if sw else set()
            cov = sum(1 for w in qw if self.mind._forms(w) & sf)
            if not best or cov > best[3]: best = (si, sent, sf, cov)
        return best

    def _lesson(self, si, right, out):
        """урок из следа чтения (_span_dbg): где стояло предложение с ответом в момент ответа и почему не взято"""
        dbg = (out or {}).get("_span_dbg") or getattr(self.mind, "_span_dbg", None) or {}; top = dbg.get("top") or []; thr = dbg.get("thr")
        how = (out or {}).get("how") or ""; silent = how.startswith("не знаю") or not (out or {}).get("answer")
        row = next((t for t in top if t[0] == si), None)
        if row is None: return ("silent-far" if silent else "wrong-sent"), None            # предложение есть в памяти, но не в топ-12 чтения
        _, cov, rare, ans = row
        if thr is not None and cov < thr: return "silent-thr", cov
        if ans is None and rare: return "silent-rare", cov
        if ans is not None and right.lower() in ans.lower(): return "wrong-sent", cov        # верный кусок был, арбитр выбрал другого
        if silent: return "silent-rare", cov
        return "wrong-piece", cov

    def learn(self, q, right, out=None, hit=False):
        """исправление «no, it's RIGHT» (или подтверждение hit=True) → случай."""
        rw = set(content_words(toks(right)))
        if not rw: return None
        qw = self._qwords(q)
        f = self._find_sent(q, right)
        if not f or f[3] == 0:
            how = (out or {}).get("how") or ""; silent = how.startswith("не знаю") or not (out or {}).get("answer")
            item = dict(q=q, right=right, sent=None, skel=skeleton(q), coarse=coarse(q), roles={}, pos=None, how=how, lesson=("no-sent-silent" if silent else "no-sent"), gcov=None, t=time.strftime("%m-%d %H:%M"))
            self.items.append(item); self.save(); self._stamp = -1; return item
        si, sent, sf, cov = f
        roles = {w: ("needed" if self.mind._forms(w) & sf else "extra") for w in qw}
        # позиция ответа: соседи слева/справа в предложении
        st = toks(sent); rt = toks(right); pos = None
        for i in range(len(st) - len(rt) + 1):
            if st[i:i + len(rt)] == rt:
                left = next((x for x in reversed(st[:i]) if x not in STOP), None); right_ = next((x for x in st[i + len(rt):] if x not in STOP), None)
                lstop = st[i - 1] if i > 0 else None
                pos = dict(left=left, right=right_, lstop=lstop, idx=i / max(1, len(st))); break
        how = (out or {}).get("how") if out else None
        lesson, gcov = ("hit", None) if hit else self._lesson(si, right, out)
        item = dict(q=q, right=right, sent=sent, skel=skeleton(q), coarse=coarse(q), roles=roles, pos=pos, how=how, lesson=lesson, gcov=gcov, t=time.strftime("%m-%d %H:%M"))
        v = self._encode([q])
        item["vec"] = v[0].tolist() if v is not None else None
        self.items.append(item); self.save(); self._stamp = -1
        return item

    # ---------- поиск похожих случаев ----------
    def _sim(self, q, qw, qf, sk, co, it):
        """сходство вопроса со случаем без нейросети: скелет 0.7 / грубая форма 0.5 / смысл слов (meaning) / пересечение слов"""
        s = 0.0
        if it["skel"] == sk: s = 0.7
        elif it.get("coarse") == co: s = 0.5
        iw = self._qwords(it["q"])
        if iw and qw:
            jf = set().union(*[self.mind._forms(x) for x in iw]); ov = sum(1 for w in qw if self.mind._forms(w) & jf)
            s = max(s, 0.9 * ov / max(len(qw), len(iw)))                                   # те же термины → почти тот же вопрос
            M = getattr(self.mind, "meaning", None)
            if M is not None:
                ps = M.phrase_sim(qw, iw)
                if ps is not None and ps > 0.45: s = max(s, min(0.65, ps))                  # смысл слов близок (68k векторов)
        return s

    def similar(self, q, k=6, min_sim=0.45):
        if not self.items: return []
        res = []
        if self._stamp != len(self.items):
            vs = [it["vec"] for it in self.items if it.get("vec")]
            self._vecs = np.array(vs, dtype=np.float32) if vs else None; self._idx = [i for i, it in enumerate(self.items) if it.get("vec")]; self._stamp = len(self.items)
        if self._vecs is not None and len(self._vecs):
            qv = self._encode([q])
            if qv is not None:
                sims = self._vecs @ qv[0]
                for j in np.argsort(-sims)[:k * 2]:
                    if sims[j] >= 0.55: res.append((float(sims[j]), self.items[self._idx[j]]))
        qw = self._qwords(q); sk = skeleton(q); co = coarse(q); qf = None
        for it in self.items[-800:]:
            if any(x is it for _, x in res): continue
            s = self._sim(q, qw, qf, sk, co, it)
            if s >= min_sim: res.append((s, it))
        res.sort(key=lambda x: -x[0]); return res[:k]

    # ---------- применение: подсказки чтению ----------
    def advice(self, q):
        """→ dict(soft=set слов вопроса, которые по похожим случаям были лишними; need=set нужных; pos=[(left,right,w)]; how=Counter;
        bold ∈[0,1] — доля (по весу) похожих случаев, где молчание было ошибкой; strict — доля, где брала не тот кусок/предложение)"""
        sims = self.similar(q)
        gb = gt = 0
        for it in self.items:
            L = it.get("lesson")
            if L in self.SILENT_ANY: gt += 1; gb += L in self.SILENT
        prior = gb / gt if gt >= 5 else 0.0; k = 2.0 if gt >= 5 else 0.0
        if not sims: return dict(soft=set(), need=set(), pos=[], how=collections.Counter(), n=0, top=0.0, bold=round(prior, 2), strict=0.0) if gt >= 5 else None
        qw = self._qwords(q)
        soft, need, pos, how = collections.Counter(), collections.Counter(), [], collections.Counter()
        M = getattr(self.mind, "meaning", None); wb = ws = wt = 0.0; sb = st = 0.0
        soft_mode = os.environ.get("MINI_AI_CASES_SOFT", "exact")                           # exact | meaning (абляция)
        soft_min = float(os.environ.get("MINI_AI_CASES_SOFT_MIN", "0.65"))                # роль переносится только из ОЧЕНЬ похожих случаев (тот же скелет / те же термины); измерено: 0.0 → +0.044, 0.65 → +0.053
        for s, it in sims:
            for w, role in it["roles"].items():
                if s < soft_min: break
                # перенос роли на слова нового вопроса: то же слово, или близкое по смыслу
                for x in qw:
                    same = (x == w) or (self.mind._forms(x) & self.mind._forms(w)) or (soft_mode == "meaning" and M is not None and len(x) > 3 and len(w) > 3 and (M.similar(x, w) or 0) > 0.5)
                    if same: (soft if role == "extra" else need)[x] += s
            if it.get("pos"): pos.append((it["pos"].get("left"), it["pos"].get("right"), s))
            if it.get("how") and it.get("lesson") == "hit": how[re.sub(r"[«»].*", "", it["how"]).split(":")[0].strip()] += s
            L = it.get("lesson")
            if L in self.SILENT_ANY: st += s; sb += s * (L in self.SILENT)
            if L in self.STRICT or L == "hit": wt += s; ws += s * (L in self.STRICT)
        # СМЕЛОСТЬ = P(предложение с ответом было видно | я молчала, вопрос похож). Похожих случаев мало → сглаживаем
        # общим опытом молчания по ВСЕМ случаям (тоже выучен из исправлений: при 0 случаев смелость 0 → поведение холодной).
        bold = (sb + k * prior) / (st + k) if (st + k) > 0 else 0.0
        soft_set = {w for w, v in soft.items() if v > need.get(w, 0) and v >= 0.6}
        off = set(os.environ.get("MINI_AI_CASES_OFF", "").split(","))                      # абляция: soft,pos,how,bold
        if "soft" in off: soft_set = set()
        if "pos" in off: pos = []
        if "how" in off: how = collections.Counter()
        if "bold" in off: bold = 0.0
        return dict(soft=soft_set, need={w for w, v in need.items() if v >= 0.6}, pos=pos, how=how, n=len(sims), top=sims[0][0],
                    bold=round(bold, 2), strict=(ws / wt if wt else 0.0))

    def report(self):
        if not self.items: return "no cases yet"
        c = collections.Counter(it["skel"] for it in self.items); L = collections.Counter(it.get("lesson") for it in self.items)
        return (f"cases remembered: {len(self.items)}; lessons: " + ", ".join(f"{k} ×{n}" for k, n in L.most_common()) +
                "; most common question shapes: " + "; ".join(f"«{k}» ×{n}" for k, n in c.most_common(3)))
