"""Проверка пути «видеокарта» без видеокарты: подменяем cupy строгим двойником,
который, как настоящий cupy, не принимает numpy-массивы. Ловит ошибку, упавшую на Kaggle 18.10
(KVModel на numpy вызывал bk.xp.sqrt → cupy)."""
import types

import numpy as np

from mini_ai import backend as bk
from mini_ai import transformer as T


class Dev(np.ndarray):
    pass


def _strict(fn):
    def f(*a, **k):
        for x in a:
            if isinstance(x, np.ndarray) and not isinstance(x, Dev):
                raise TypeError("numpy array passed to fake cupy")
        r = fn(*a, **k)
        return r.view(Dev) if isinstance(r, np.ndarray) else r
    return f


def test_kvmodel_works_in_gpu_mode(monkeypatch):
    fake = types.SimpleNamespace(**{n: getattr(np, n) for n in dir(np) if not n.startswith("_")})
    for n in ("sqrt", "tanh", "exp", "log"):
        setattr(fake, n, _strict(getattr(np, n)))
    monkeypatch.setattr(bk, "xp", fake)
    monkeypatch.setattr(bk, "GPU", True)
    monkeypatch.setattr(bk, "to_np", lambda a: np.asarray(a).view(np.ndarray))
    p, cfg = T.init_params(30, d=16, n_layer=2, n_head=2, block=12)
    kv = T.KVModel(p, cfg)
    out, _ = kv.generate([1, 5, 6, 2], max_new=4)
    assert len(out) <= 4
