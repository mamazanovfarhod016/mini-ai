"""
RoPE (вращающиеся позиционные эмбеддинги) — тесты.

Проверяем:
  1. gradient check для apply_rope (backward = поворот на -угол);
  2. gradient check для attention_forward/backward с rope (целиком);
  3. causal-маска всё ещё блокирует будущее у rope-модели;
  4. KV-cache == полный forward для rope, в том числе НА ПОЗИЦИЯХ > 200
     (длинный контекст — то, ради чего всё затевалось);
  5. Python ↔ JavaScript парность для rope-модели (логиты и greedy-ответы);
  6. torch_engine совпадает с numpy для rope (loss, все градиенты, 5 шагов обучения).
"""
import json
import subprocess
import sys
import shutil
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mini_ai.transformer import (  # noqa: E402
    attention_backward, attention_forward, apply_rope, apply_rope_backward,
    backward, cross_entropy, forward, grow_params, init_params, KVModel,
    rope_cos_sin, to_f16_b64,
)


def _numeric_grad(f, w, eps=1e-4):
    grad = np.zeros_like(w)
    it = np.nditer(w, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = w[i]
        w[i] = old + eps
        lp = f()
        w[i] = old - eps
        lm = f()
        w[i] = old
        grad[i] = (lp - lm) / (2 * eps)
    return grad


def _rel_err(a, b):
    return float(np.linalg.norm(a - b) / (np.linalg.norm(a) + np.linalg.norm(b) + 1e-12))


# ======================================================================================
# 1. Gradient check: сам поворот apply_rope
# ======================================================================================
def test_gradient_check_apply_rope():
    rng = np.random.default_rng(10)
    H, T, dh = 2, 5, 8
    x = rng.normal(size=(H, T, dh))
    cos, sin = rope_cos_sin(np.arange(T), dh, 10000.0)
    dy = rng.normal(size=(H, T, dh))

    def loss_fn(xx):
        y = apply_rope(xx, cos, sin)
        return float((y * dy).sum())

    y = apply_rope(x, cos, sin)
    dx = apply_rope_backward(dy, cos, sin)
    num = _numeric_grad(lambda: loss_fn(x), x)
    rel = _rel_err(num, dx)
    assert rel < 1e-5, rel
    # Поворот сохраняет длину вектора (ортогональное преобразование)
    assert np.allclose(np.linalg.norm(x, axis=-1), np.linalg.norm(y, axis=-1), atol=1e-8)


# ======================================================================================
# 2. Gradient check: attention с RoPE целиком
# ======================================================================================
def test_gradient_check_attention_rope():
    rng = np.random.default_rng(11)
    B, T, d, H = 2, 6, 8, 2
    dh = d // H
    h = rng.normal(0, 0.1, size=(B, T, d))
    wqkv = rng.normal(0, 0.1, size=(d, 3 * d))
    bqkv = rng.normal(0, 0.1, size=3 * d)
    wo = rng.normal(0, 0.1, size=(d, d))
    bo = rng.normal(0, 0.1, size=d)
    dy = rng.normal(size=(B, T, d))
    rope = rope_cos_sin(np.arange(T), dh, 10000.0)

    def loss_fn():
        y, _ = attention_forward(h, wqkv, bqkv, wo, bo, H, rope)
        return float((y * dy).sum())

    out, cache = attention_forward(h, wqkv, bqkv, wo, bo, H, rope)
    dh_, dwqkv, dbqkv, dwo, dbo = attention_backward(dy, cache)

    for name, w, dw in [("h", h, dh_), ("wqkv", wqkv, dwqkv), ("bqkv", bqkv, dbqkv),
                         ("wo", wo, dwo), ("bo", bo, dbo)]:
        num = _numeric_grad(loss_fn, w, eps=1e-4)
        rel = _rel_err(num, dw)
        assert rel < 2e-4, f"{name}: rel err {rel:.2e}"


def test_gradient_check_full_model_rope():
    """Полная сверка backward() для rope-модели (как test_gradient_check_block, но pos='rope')."""
    rng = np.random.default_rng(12)
    V, d, n_layer, n_head = 12, 8, 1, 2
    p, cfg = init_params(V, d=d, n_layer=n_layer, n_head=n_head, block=50, seed=3,
                          dtype=np.float64, pos="rope")
    B, T = 2, 5
    idx = rng.integers(0, V, size=(B, T))
    targets = rng.integers(0, V, size=(B, T))
    mask = np.ones((B, T), dtype=bool)

    def loss_fn():
        logits, cache = forward(p, cfg, idx, drop=0.0)
        loss, _ = cross_entropy(logits, targets, mask)
        return float(loss)

    logits, cache = forward(p, cfg, idx, drop=0.0)
    loss, dlogits = cross_entropy(logits, targets, mask)
    grads = backward(p, cfg, dlogits, cache)

    assert "wpe" not in p and "wpe" not in grads   # у rope-модели нет таблицы позиций
    for name in ["wte", "0.ln1.g", "0.attn.wqkv", "0.mlp.w1", "lnf.g"]:
        num = _numeric_grad(loss_fn, p[name], eps=1e-4)
        rel = _rel_err(num, grads[name])
        assert rel < 2e-3, f"{name}: rel err {rel:.2e}"


# ======================================================================================
# 3. Causal-маска всё ещё блокирует будущее у rope-модели
# ======================================================================================
def test_causal_mask_blocks_future_rope():
    rng = np.random.default_rng(13)
    V, d, n_layer, n_head = 15, 8, 2, 2
    p, cfg = init_params(V, d=d, n_layer=n_layer, n_head=n_head, block=50, seed=4, pos="rope")
    T = 6
    idx = rng.integers(0, V, size=(1, T))
    logits1, _ = forward(p, cfg, idx, drop=0.0)

    idx2 = idx.copy()
    idx2[0, T - 1] = (idx2[0, T - 1] + 1) % V
    logits2, _ = forward(p, cfg, idx2, drop=0.0)

    assert np.allclose(logits1[0, :T - 1], logits2[0, :T - 1], atol=1e-10)
    assert not np.allclose(logits1[0, T - 1], logits2[0, T - 1], atol=1e-6)


# ======================================================================================
# 4. KV-cache == полный forward, в т.ч. на позициях > 200 (длинный контекст)
# ======================================================================================
def test_kvcache_matches_full_forward_rope_long():
    rng = np.random.default_rng(14)
    V, d, n_layer, n_head = 20, 16, 2, 4
    p, cfg = init_params(V, d=d, n_layer=n_layer, n_head=n_head, block=10000, seed=5, pos="rope")
    T = 300
    idx = rng.integers(0, V, size=(1, T))
    logits_full, _ = forward(p, cfg, idx, drop=0.0)

    kv = KVModel(p, cfg)
    cache = kv.new_cache()
    last = None
    for t in range(T):
        last, _ = kv.step(int(idx[0, t]), t, cache)
    assert np.allclose(logits_full[0, T - 1].astype(np.float64), last, atol=1e-5)

    # Отдельно проверяем позицию ПОСЛЕ 200 (не только последнюю) — полный forward урезан до t+1 токенов
    t_check = 250
    logits_at_250, _ = forward(p, cfg, idx[:, :t_check + 1], drop=0.0)
    kv2 = KVModel(p, cfg)
    cache2 = kv2.new_cache()
    last2 = None
    for t in range(t_check + 1):
        last2, _ = kv2.step(int(idx[0, t]), t, cache2)
    assert np.allclose(logits_at_250[0, t_check].astype(np.float64), last2, atol=1e-5)


# ======================================================================================
# 5. Python ↔ JavaScript parity для rope-модели
# ======================================================================================
@pytest.mark.skipif(shutil.which("node") is None, reason="нет node")
def test_python_js_parity_rope():
    from mini_ai.bpe import BPETokenizer
    rng = np.random.default_rng(15)
    V, d, n_layer, n_head = 24, 16, 2, 4
    p, cfg = init_params(V, d=d, n_layer=n_layer, n_head=n_head, block=10000, seed=6, pos="rope")

    tok = BPETokenizer()
    tok.itos = [f"<s{i}>" for i in range(4)] + [chr(97 + i) for i in range(V - 4)]
    tok.stoi = {c: i for i, c in enumerate(tok.itos)}
    tok.merges = []

    order = list(p.keys())
    model_json = {
        "format": "mini-ai-transformer/1",
        "config": cfg,
        "vocab": tok.itos,
        "max_q": 20,
        "tokenizer": "bpe",
        "merges": [],
        "order": order,
        "shapes": {k: list(p[k].shape) for k in order},
        "weights": {k: to_f16_b64(p[k]) for k in order},
    }
    model_path = ROOT / "tests" / "_tmp_rope_model.json"
    model_path.write_text(json.dumps(model_json, ensure_ascii=False))
    try:
        ids_list = [[1] + [int(x) for x in rng.integers(4, V, size=6)] + [2] for _ in range(5)]
        out = subprocess.run(
            ["node", str(ROOT / "tests" / "js_rope_think.js"), str(model_path)],
            input=json.dumps(ids_list, ensure_ascii=False),
            capture_output=True, text=True, check=True,
        ).stdout
        js_results = json.loads(out)

        kv = KVModel(p, cfg)
        for ids, js in zip(ids_list, js_results):
            out_ids, info = kv.generate(ids, max_new=30, temperature=0.0)
            assert out_ids == js["out"], (ids, out_ids, js["out"])
            py_logits = np.asarray(info["logits"][0], dtype=np.float64)
            js_logits = np.asarray(js["firstLogits"], dtype=np.float64)
            assert np.max(np.abs(py_logits - js_logits)) < 1e-3
    finally:
        model_path.unlink(missing_ok=True)
