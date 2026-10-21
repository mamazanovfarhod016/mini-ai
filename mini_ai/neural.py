"""neural.py — нейронный «глаз» для чтения формулировок: кодировщик предложений (MiniLM, 22M, CPU).
Ядро остаётся символьным (факты, граф, правила, рост); нейросеть делает одно — говорит, какое предложение памяти
ближе всего по смыслу к вопросу, даже если слова другие («What holds together adenine and thymine?» ↔
«… held together with two hydrogen bonds»). Это то, в чём правила слепы, а градиентная модель — зрячая.
Индекс: предложения памяти → векторы (384, нормированные), считаются лениво батчами; поиск = скалярное произведение.
Предельно честно: модель предобучена кем-то (sentence-transformers/all-MiniLM-L6-v2), не нами; мы её не дообучаем."""
import os, numpy as np, threading

_MODEL = None; _LOCK = threading.Lock()
def _model():
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            from sentence_transformers import SentenceTransformer
            _MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        return _MODEL

class SentIndex:
    """ленивый индекс: векторы считаются только для предложений-кандидатов (найденных ядром по словам), кэшируются.
    Полный учебник (50k предложений) НЕ кодируется — это минуты CPU и гигабайты; нейросеть — уточнитель, не поисковик."""
    def __init__(self, kb, dim=384, max_cache=60000):
        self.kb = kb; self.dim = dim; self.cache = {}; self.ok = True; self.max_cache = max_cache
    def _vecs(self, sis):
        todo = [si for si in sis if si not in self.cache]
        if todo:
            try:
                new = _model().encode([self.kb.sents[si][0] for si in todo], batch_size=32, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
                for si, v in zip(todo, new):
                    if len(self.cache) < self.max_cache: self.cache[si] = v
            except Exception as e:
                self.ok = False; print("neural index off:", e); return None
        return np.stack([self.cache[si] for si in sis if si in self.cache]) if sis else None
    def search(self, q, k=8, min_sim=0.35, among=None):
        """→ [(si, sim)] среди предложений-кандидатов `among` (если None — по словам вопроса из kb.sent_by_word, до 200)"""
        if not self.ok: return []
        if among is None:
            from mini_ai.memory_v2 import toks, content_words
            import collections
            c = collections.Counter()
            for w in content_words(toks(q)):
                for si in self.kb.sent_by_word.get(w, [])[:80]: c[si] += 1
            among = [si for si, _ in c.most_common(200)]
        among = [si for si in among if 0 <= si < len(self.kb.sents)]
        if not among: return []
        V = self._vecs(among)
        if V is None or not len(V): return []
        try: qv = _model().encode([q], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)[0].astype(np.float32)
        except Exception: return []
        sims = V @ qv; order = np.argsort(-sims)[:k]
        return [(among[i], float(sims[i])) for i in order if sims[i] >= min_sim]
    def sim(self, a, b):
        try:
            v = _model().encode([a, b], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True); return float(v[0] @ v[1])
        except Exception: return 0.0


# ---------- нейронный ЧИТАТЕЛЬ: вопрос + контекст (2–4 предложения) → кусок-ответ ----------
_QA = None
def _qa():
    global _QA
    with _LOCK:
        if _QA is None:
            import torch
            from transformers import AutoTokenizer, AutoModelForQuestionAnswering
            name = "distilbert-base-cased-distilled-squad"
            tok = AutoTokenizer.from_pretrained(name); mdl = AutoModelForQuestionAnswering.from_pretrained(name); mdl.eval()
            torch.set_num_threads(1); _QA = (tok, mdl)
        return _QA

def read_answer(question, context, max_len=384):
    """→ (answer_text, score) ; score = лучший span минус «нет ответа». Контекст — предложения, которые НАШЛО ядро,
    не весь учебник: читатель только достаёт кусок, выбор места остаётся за символьной памятью."""
    try:
        import torch
        tok, mdl = _qa()
        enc = tok(question, context, return_tensors="pt", truncation="only_second", max_length=max_len)
        with torch.no_grad(): out = mdl(**enc)
        s, e = out.start_logits[0], out.end_logits[0]; n = len(s)
        seq = enc.sequence_ids(0); ctx_idx = [i for i in range(n) if seq[i] == 1]
        if not ctx_idx: return None, 0.0
        lo, hi = ctx_idx[0], ctx_idx[-1]
        best = (-1e9, lo, lo)
        for i in range(lo, hi + 1):
            si = float(s[i])
            for j in range(i, min(i + 10, hi + 1)):
                sc = si + float(e[j])
                if sc > best[0]: best = (sc, i, j)
        ans = tok.decode(enc["input_ids"][0][best[1]:best[2] + 1], skip_special_tokens=True).strip()
        null = float(s[0] + e[0])
        return (ans or None), round(best[0] - null, 2)
    except Exception as ex:
        return None, 0.0
