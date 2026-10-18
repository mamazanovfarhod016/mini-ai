"""Постоянные знания: рост сети не стирает то, что она знала."""
import numpy as np

from mini_ai.transformer import forward, grow_params, init_params


def _logits(p, cfg, idx):
    return np.asarray(forward(p, cfg, idx)[0], dtype=np.float64)


def test_same_size_is_exact_copy():
    p0, c0 = init_params(50, d=32, n_layer=2, n_head=4, block=20, seed=1)
    p1, c1 = init_params(50, d=32, n_layer=2, n_head=4, block=20, seed=2)
    grow_params(p1, c1, p0, c0)
    idx = np.random.default_rng(0).integers(0, 50, (3, 12))
    assert np.allclose(_logits(p0, c0, idx), _logits(p1, c1, idx))


def test_more_layers_answers_exactly_the_same():
    p0, c0 = init_params(50, d=32, n_layer=2, n_head=4, block=20, seed=1)
    p1, c1 = init_params(50, d=32, n_layer=5, n_head=4, block=30, seed=2)
    grow_params(p1, c1, p0, c0)
    idx = np.random.default_rng(0).integers(0, 50, (3, 12))
    assert np.allclose(_logits(p0, c0, idx), _logits(p1, c1, idx), atol=1e-5)


def test_wider_keeps_most_of_the_answer():
    p0, c0 = init_params(50, d=32, n_layer=2, n_head=4, block=20, seed=1)
    p1, c1 = init_params(50, d=48, n_layer=3, n_head=4, block=20, seed=2)
    grow_params(p1, c1, p0, c0)
    idx = np.random.default_rng(0).integers(0, 50, (4, 12))
    a, b = _logits(p0, c0, idx), _logits(p1, c1, idx)
    fresh, cf = init_params(50, d=48, n_layer=3, n_head=4, block=20, seed=3)
    corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    corr_fresh = np.corrcoef(a.ravel(), _logits(fresh, cf, idx).ravel())[0, 1]
    assert corr > 0.5 and corr > corr_fresh + 0.3


def test_cannot_shrink():
    p0, c0 = init_params(50, d=32, n_layer=3, n_head=4, block=20)
    p1, c1 = init_params(50, d=32, n_layer=2, n_head=4, block=20)
    try:
        grow_params(p1, c1, p0, c0)
    except ValueError:
        return
    raise AssertionError("уменьшение должно быть запрещено")


def test_tokenizer_extend_keeps_old_ids_and_vocab_growth():
    from mini_ai.bpe import BPETokenizer
    tok = BPETokenizer().train(["привет как дела", "привет мир", "как ты"] * 5, vocab_size=40)
    old = {w: tok.encode(w) for w in ["привет", "как", "дела"]}
    n_old = len(tok)
    added = tok.extend(["what is two plus two", "what is three"] * 30, extra=20)
    assert added > 0 and tok.itos[:n_old] == tok.itos[:n_old]
    assert all(tok.encode(w) == ids for w, ids in old.items())
    assert 4 not in tok.encode("what")                   # 4 = <unk>: английские буквы теперь известны
    p0, c0 = init_params(n_old, d=32, n_layer=2, n_head=4, block=20, seed=1)
    p1, c1 = init_params(len(tok), d=32, n_layer=2, n_head=4, block=20, seed=2)
    grow_params(p1, c1, p0, c0)
    assert np.allclose(np.asarray(p1["wte"][:n_old]), np.asarray(p0["wte"]))


# ======================================================================================
# RoPE: рост сети
# ======================================================================================
def test_learned_to_rope_drops_wpe_keeps_other_weights():
    """Старая модель (learned wpe) → новая rope-модель: wpe пропадает, остальное копируется."""
    p0, c0 = init_params(50, d=32, n_layer=2, n_head=4, block=20, seed=1, pos="learned")
    p1, c1 = init_params(50, d=32, n_layer=2, n_head=4, block=10000, seed=2, pos="rope")
    assert "wpe" in p0 and "wpe" not in p1
    grow_params(p1, c1, p0, c0)
    assert np.allclose(np.asarray(p1["wte"]), np.asarray(p0["wte"]))
    assert np.allclose(np.asarray(p1["0.attn.wqkv"]), np.asarray(p0["0.attn.wqkv"]))


def test_rope_more_layers_answers_exactly_the_same():
    """rope → rope, больше слоёв: новые слои — «пропуск», ответ на старых позициях не меняется."""
    p0, c0 = init_params(50, d=32, n_layer=2, n_head=4, block=10000, seed=1, pos="rope")
    p1, c1 = init_params(50, d=32, n_layer=5, n_head=4, block=10000, seed=2, pos="rope")
    grow_params(p1, c1, p0, c0)
    idx = np.random.default_rng(0).integers(0, 50, (3, 12))
    assert np.allclose(_logits(p0, c0, idx), _logits(p1, c1, idx), atol=1e-5)


def test_rope_wider_heads_keeps_most_of_the_answer():
    """rope → rope, шире d и больше голов (4→8): dh = d/n_head остаётся тем же (32/4=8, 64/8=8),
    поэтому старые головы продолжают использовать те же пары координат и те же углы RoPE."""
    p0, c0 = init_params(50, d=32, n_layer=2, n_head=4, block=10000, seed=1, pos="rope")
    p1, c1 = init_params(50, d=64, n_layer=3, n_head=8, block=10000, seed=2, pos="rope")
    grow_params(p1, c1, p0, c0)
    idx = np.random.default_rng(0).integers(0, 50, (4, 12))
    a, b = _logits(p0, c0, idx), _logits(p1, c1, idx)
    fresh, cf = init_params(50, d=64, n_layer=3, n_head=8, block=10000, seed=3, pos="rope")
    corr = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    corr_fresh = np.corrcoef(a.ravel(), _logits(fresh, cf, idx).ravel())[0, 1]
    assert corr > 0.5 and corr > corr_fresh + 0.3
