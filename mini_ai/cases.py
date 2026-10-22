"""cases.py — РОСТ ЧЕРЕЗ СЛУЧАИ (case-based reasoning), без градиента.

Абляция показала: счётчики доверия/смелости/внимания на 100–200 исправлениях — шум (±1 %). Они меняют «кому верить»,
а ошибки — в «где искать и что взять». Здесь каждое исправление запоминается как СЛУЧАЙ:
    вопрос  →  предложение памяти с ответом  →  как ответ стоит в нём (позиция, соседи, тип разбора)
Новый вопрос ищет ближайшие случаи по смыслу (MiniLM) и по форме (скелет вопроса) и ПОВТОРЯЕТ их способ чтения:
  1. «где искать»: случай говорит, какие слова вопроса были в ответном предложении, а какие лишние →
     для нового вопроса слова той же роли (по смыслу) взвешиваются так же;
  2. «что взять»: шаблон позиции ответа относительно слов вопроса («после called», «перед is», «между use и make»);
  3. «каким способом»: если похожие случаи решались deep / 1 шаг / отрывком — этот способ получает бонус.
Это не схема «вопрос-форма → слот» (их 90 на класс и они не переносятся), а перенос по СМЫСЛУ вопроса целиком.
"""
import re, json, os, time, collections
import numpy as np
from mini_ai.memory_v2 import toks, content_words

STOP = set("the a an of in on at to for by with from and or is are was were be been do does did what which who whom whose how why when where that this these those it its their there as into than then so such".split())

def skeleton(q):
    """скелет вопроса: содержательные слова → «_», служебные остаются. «What do most living things use to make ATP?» → «what do _ _ _ use to make _»"""
    t = toks(q); out = []
    for w in t:
        out.append(w if w in STOP or w in ("called", "known", "term", "type", "kind", "name", "part", "process", "example") else "_")
    return " ".join(out)

class Cases:
    def __init__(self, mind, state_dir=None):
        self.mind = mind; self.path = os.path.join(state_dir, "cases.json") if state_dir else None
        self.items = []            # dict(q, right, sent, skel, roles{word: needed|extra}, pos, how, vec(list)|None)
        self._vecs = None; self._stamp = -1
        self.load()

    # ---------- память случаев ----------
    def load(self):
        if not self.path or not os.path.exists(self.path): return
        try: self.items = json.load(open(self.path))[-2000:]
        except Exception: self.items = []
    def save(self):
        if self.path: json.dump(self.items[-2000:], open(self.path, "w"), ensure_ascii=False)

    def _encode(self, texts):
        try:
            from mini_ai.neural import _model
            return _model().encode(texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        except Exception: return None

    def learn(self, q, right, out=None):
        """исправление «no, it's RIGHT» → случай. Находим предложение памяти с RIGHT, покрывающее больше всего слов вопроса."""
        kb = self.mind.kb; rl = right.lower(); rw = set(content_words(toks(right)))
        if not rw: return None
        qw = [w for w in content_words(toks(q)) if w not in STOP]
        best = None
        for si in range(len(kb.sents) - 1, max(-1, len(kb.sents) - 40000), -1):
            sent = kb.sents[si][0]; sl = sent.lower()
            if rl not in sl: continue
            sw = set(content_words(toks(sent))); sf = set().union(*[self.mind._forms(x) for x in sw]) if sw else set()
            cov = sum(1 for w in qw if self.mind._forms(w) & sf)
            if not best or cov > best[0]: best = (cov, si, sent, sf)
        if not best or best[0] == 0: return None
        cov, si, sent, sf = best
        roles = {w: ("needed" if self.mind._forms(w) & sf else "extra") for w in qw}
        # позиция ответа: соседи слева/справа в предложении
        st = toks(sent); rt = toks(right); pos = None
        for i in range(len(st) - len(rt) + 1):
            if st[i:i + len(rt)] == rt:
                left = next((x for x in reversed(st[:i]) if x not in STOP), None); right_ = next((x for x in st[i + len(rt):] if x not in STOP), None)
                lstop = st[i - 1] if i > 0 else None
                pos = dict(left=left, right=right_, lstop=lstop, idx=i / max(1, len(st))); break
        how = (out or {}).get("how") if out else None
        item = dict(q=q, right=right, sent=sent, skel=skeleton(q), roles=roles, pos=pos, how=how, t=time.strftime("%m-%d %H:%M"))
        v = self._encode([q])
        item["vec"] = v[0].tolist() if v is not None else None
        self.items.append(item); self.save(); self._stamp = -1
        return item

    # ---------- поиск похожих случаев ----------
    def similar(self, q, k=5, min_sim=0.55):
        if not self.items: return []
        if self._stamp != len(self.items):
            vs = [it["vec"] for it in self.items if it.get("vec")]
            self._vecs = np.array(vs, dtype=np.float32) if vs else None; self._idx = [i for i, it in enumerate(self.items) if it.get("vec")]; self._stamp = len(self.items)
        res = []
        if self._vecs is not None and len(self._vecs):
            qv = self._encode([q])
            if qv is not None:
                sims = self._vecs @ qv[0]
                for j in np.argsort(-sims)[:k * 2]:
                    if sims[j] >= min_sim: res.append((float(sims[j]), self.items[self._idx[j]]))
        sk = skeleton(q)
        for it in self.items[-500:]:
            if it["skel"] == sk and not any(x is it for _, x in res): res.append((0.6, it))
        res.sort(key=lambda x: -x[0]); return res[:k]

    # ---------- применение: подсказки чтению ----------
    def advice(self, q):
        """→ dict(soft=set слов вопроса, которые по похожим случаям были лишними; need=set нужных; pos=[(left,right,w)]; how=Counter)"""
        sims = self.similar(q)
        if not sims: return None
        qw = [w for w in content_words(toks(q)) if w not in STOP]
        soft, need, pos, how = collections.Counter(), collections.Counter(), [], collections.Counter()
        M = getattr(self.mind, "meaning", None)
        for s, it in sims:
            for w, role in it["roles"].items():
                # перенос роли на слова нового вопроса: то же слово, или близкое по смыслу
                for x in qw:
                    same = (x == w) or (self.mind._forms(x) & self.mind._forms(w)) or (M is not None and len(x) > 3 and len(w) > 3 and (M.similar(x, w) or 0) > 0.5)
                    if same: (soft if role == "extra" else need)[x] += s
            if it.get("pos"): pos.append((it["pos"].get("left"), it["pos"].get("right"), s))
            if it.get("how"): how[re.sub(r"[«»].*", "", it["how"]).split(":")[0].strip()] += s
        soft_set = {w for w, v in soft.items() if v > need.get(w, 0) and v >= 0.6}
        return dict(soft=soft_set, need={w for w, v in need.items() if v >= 0.6}, pos=pos, how=how, n=len(sims), top=sims[0][0])

    def report(self):
        if not self.items: return "no cases yet"
        c = collections.Counter(it["skel"] for it in self.items)
        return f"cases remembered: {len(self.items)}; most common question shapes: " + "; ".join(f"«{k}» ×{n}" for k, n in c.most_common(3))
