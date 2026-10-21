"""Бинарная память (exp32): та же архитектура (ключ-вектор → текст), другая арифметика.
Вектор float D → D бит (знак). Сходство = D − 2·hamming = скалярное произведение ±1.
Popcount — SWAR на uint64 по кускам: 1 млн записей × 2048 бит = 256 МБ, запрос 0.22 с на 2 CPU (без FFT, без float).
"""
import os, json
import numpy as np

D = 2048
_M1, _M2, _M4, _H01 = (np.uint64(0x5555555555555555), np.uint64(0x3333333333333333),
                        np.uint64(0x0f0f0f0f0f0f0f0f), np.uint64(0x0101010101010101))


def pack(M):
    """float (N,D) или (D,) → uint8 (N, D/8)."""
    M = np.atleast_2d(M); return np.packbits((M > 0).astype(np.uint8), axis=-1)


def _pop64(x):
    x = x - ((x >> np.uint64(1)) & _M1)
    x = (x & _M2) + ((x >> np.uint64(2)) & _M2)
    x = (x + (x >> np.uint64(4))) & _M4
    return (x * _H01) >> np.uint64(56)


def hamming(B, q, chunk=50000):
    """B: (N, D/8) uint8, q: (D/8,) uint8 → (N,) расстояния Хэмминга."""
    B64, q64 = B.view(np.uint64), q.view(np.uint64); out = np.empty(len(B), np.uint16)
    for i in range(0, len(B), chunk):
        out[i:i + chunk] = _pop64(np.bitwise_xor(B64[i:i + chunk], q64)).sum(axis=1).astype(np.uint16)
    return out


class BinaryStore:
    """Список: бинарный ключ → текст + мета. Только растёт (не забывает). На диске: .npy + .jsonl."""
    def __init__(self, path=None, d=D):
        self.d = d; self.path = path; self.B = np.zeros((0, d // 8), np.uint8); self.text, self.meta = [], []
        self._buf = []
        if path and os.path.exists(os.path.join(path, "keys.npy")): self.load(path)

    def __len__(self): return len(self.text)

    def add(self, key_float, text, meta=None):
        self._buf.append(pack(key_float)[0]); self.text.append(text); self.meta.append(meta or {})
        if len(self._buf) >= 20000: self._flush()      # блоками, чтобы не держать сотни тысяч мелких массивов

    def add_packed(self, key_u8, text, meta=None):
        self._buf.append(key_u8); self.text.append(text); self.meta.append(meta or {})

    def _flush(self):
        if self._buf: self.B = np.vstack([self.B, np.stack(self._buf)]); self._buf = []

    def search(self, q_float, topk=3):
        self._flush()
        if not len(self.B): return []
        h = hamming(self.B, pack(q_float)[0]); order = np.argsort(h)[:topk]
        return [(self.text[i], int(self.d - 2 * int(h[i])), self.meta[i]) for i in order]

    def save(self, path=None):
        path = path or self.path
        if not path: raise ValueError("нет пути")
        os.makedirs(path, exist_ok=True); self._flush()
        np.save(os.path.join(path, "keys.npy"), self.B)
        with open(os.path.join(path, "text.jsonl"), "w") as f:
            for t, m in zip(self.text, self.meta): f.write(json.dumps([t, m], ensure_ascii=False) + "\n")

    def load(self, path):
        self.B = np.load(os.path.join(path, "keys.npy")); self.text, self.meta = [], []
        with open(os.path.join(path, "text.jsonl")) as f:
            for line in f:
                t, m = json.loads(line); self.text.append(t); self.meta.append(m)
