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
    def __init__(self, kb, dim=384):
        self.kb = kb; self.dim = dim; self.vecs = np.zeros((0, dim), np.float32); self.built = 0; self.ok = True
    def _sync(self):
        n = len(self.kb.sents)
        if n <= self.built: return
        try:
            m = _model(); texts = [s for s, _ in self.kb.sents[self.built:n]]
            new = m.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
            self.vecs = np.vstack([self.vecs, new]) if len(self.vecs) else new; self.built = n
        except Exception as e:
            self.ok = False; print("neural index off:", e)
    def search(self, q, k=8, min_sim=0.35):
        """→ [(si, sim)] предложения памяти по смыслу"""
        if not self.ok: return []
        self._sync()
        if not self.built: return []
        try: qv = _model().encode([q], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)[0].astype(np.float32)
        except Exception: return []
        sims = self.vecs @ qv; idx = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        return sorted([(int(i), float(sims[i])) for i in idx if sims[i] >= min_sim], key=lambda t: -t[1])
    def sim(self, a, b):
        try:
            v = _model().encode([a, b], normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True); return float(v[0] @ v[1])
        except Exception: return 0.0
