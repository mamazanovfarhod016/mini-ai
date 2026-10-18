"""PyTorch-движок обязан совпадать с нашим numpy-кодом (эталоном): loss, все градиенты, шаги обучения."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mini_ai import transformer as T  # noqa: E402
from mini_ai.torch_engine import TorchEngine  # noqa: E402


def _batch(V, B=3, Tn=14, seed=0):
    r = np.random.default_rng(seed)
    idx = r.integers(4, V, (B, Tn))
    tg = r.integers(4, V, (B, Tn))
    mask = r.random((B, Tn)) < 0.6
    return idx, tg, mask


def test_loss_and_all_grads_match_numpy():
    p, cfg = T.init_params(60, d=32, n_layer=3, n_head=4, block=20, seed=1)
    p64 = {k: v.astype(np.float64) for k, v in p.items()}
    idx, tg, mask = _batch(60)
    loss_np, g_np = T.loss_and_grads(p64, cfg, idx, tg, mask)
    eng = TorchEngine(p, cfg, device="cpu")
    loss_t = eng.loss_and_grads(idx, tg, mask)
    assert abs(float(loss_t.detach()) - loss_np) < 1e-5
    for k, g in g_np.items():
        gt = eng.p[k].grad.detach().numpy()
        assert np.abs(gt - g).max() <= 1e-4 * (np.abs(g).max() + 1e-6), k


def test_training_steps_match_numpy():
    p, cfg = T.init_params(60, d=32, n_layer=2, n_head=4, block=20, seed=2)
    p_np = {k: v.astype(np.float64) for k, v in p.items()}
    opt = T.AdamW(p_np, lr=1e-3, wd=0.01)
    eng = TorchEngine(p, cfg, device="cpu")
    for s in range(5):
        idx, tg, mask = _batch(60, seed=10 + s)
        lr = T.lr_schedule(s, 5, 1e-3, 2)
        _, g = T.loss_and_grads(p_np, cfg, idx, tg, mask)
        T.clip_grads(g, 1.0)
        opt.step(p_np, g, lr)
        eng.step(idx, tg, mask, lr)
    pt = eng.params_numpy()
    for k in p_np:
        assert np.abs(pt[k] - p_np[k]).max() < 1e-4, k


# ======================================================================================
# RoPE: torch_engine обязан совпадать с numpy (pos='rope') — loss, все градиенты, шаги.
# ======================================================================================
def test_loss_and_all_grads_match_numpy_rope():
    p, cfg = T.init_params(60, d=32, n_layer=3, n_head=4, block=20, seed=1, pos="rope")
    p64 = {k: v.astype(np.float64) for k, v in p.items()}
    idx, tg, mask = _batch(60)
    loss_np, g_np = T.loss_and_grads(p64, cfg, idx, tg, mask)
    eng = TorchEngine(p, cfg, device="cpu")
    loss_t = eng.loss_and_grads(idx, tg, mask)
    assert abs(float(loss_t.detach()) - loss_np) < 1e-5
    for k, g in g_np.items():
        gt = eng.p[k].grad.detach().numpy()
        assert np.abs(gt - g).max() <= 1e-4 * (np.abs(g).max() + 1e-6), k


def test_training_steps_match_numpy_rope():
    p, cfg = T.init_params(60, d=32, n_layer=2, n_head=4, block=20, seed=2, pos="rope")
    p_np = {k: v.astype(np.float64) for k, v in p.items()}
    opt = T.AdamW(p_np, lr=1e-3, wd=0.01)
    eng = TorchEngine(p, cfg, device="cpu")
    for s in range(5):
        idx, tg, mask = _batch(60, seed=10 + s)
        lr = T.lr_schedule(s, 5, 1e-3, 2)
        _, g = T.loss_and_grads(p_np, cfg, idx, tg, mask)
        T.clip_grads(g, 1.0)
        opt.step(p_np, g, lr)
        eng.step(idx, tg, mask, lr)
    pt = eng.params_numpy()
    for k in p_np:
        assert np.abs(pt[k] - p_np[k]).max() < 1e-4, k
