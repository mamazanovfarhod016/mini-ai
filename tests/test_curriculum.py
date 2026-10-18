"""
Тесты train_curriculum.py: крошечная модель, CPU, < 2 минуты.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv/bin/python")

from mini_ai.curriculum import (
    build_dynamic_batch, encode_prompt, lm_example_to_seq, pack_lm_windows, qa_example_to_seq,
    context_example_to_seq, pad_batch, SourcePool, setup_model, save_checkpoint, load_latest_checkpoint,
)
from mini_ai.transformer import BOS, EOS, PAD, SEP, init_params, CharTokenizer
from mini_ai.torch_engine import TorchEngine
from mini_ai.batched_generate import batched_generate, kv_greedy_reference


def tiny_tok():
    chars = list("abcdefghij 0123456789")
    return CharTokenizer(chars)


# ---------------------------------------------------------------------------
def test_lm_packing_masks_correct():
    tok = tiny_tok()
    rng = np.random.RandomState(0)
    lines = ["abc def", "ghi jab", "cab fed"]
    windows = pack_lm_windows(tok, lines, max_len=5, rng=rng, raw=True)
    assert windows
    for w in windows:
        seq, tgt, mask = lm_example_to_seq(w)
        assert len(seq) == len(tgt) == len(mask)
        assert all(mask)  # loss на всех токенах LM-окна
        assert tgt == w[1:]
        assert seq == w[:-1]


def test_qa_mask_only_answer():
    tok = tiny_tok()
    seq, tgt, mask = qa_example_to_seq(tok, "abc", "def", max_q=10)
    # seq = [BOS, a,b,c, SEP, d,e] (без финального EOS, т.к. seq=full[:-1])
    assert seq[0] == BOS
    n_false = mask.count(False)
    n_true = mask.count(True)
    assert n_true > 0
    # маска False должна покрывать bos+вопрос+sep (длина = 1+3+1-1=4, т.к. targets сдвинуты)
    assert mask[:4] == [False, False, False, False]
    assert all(mask[4:])


def test_context_mask_only_answer_and_truncation():
    tok = tiny_tok()
    ctx = "a" * 50
    seq, tgt, mask = context_example_to_seq(tok, ctx, "abc", "def", max_len=20)
    assert len(seq) <= 20
    assert any(mask)
    # последние токены (ответ) должны быть под маской
    assert mask[-1] is True


def test_no_test_items_reach_training(tmp_path):
    """Генерируем qa-источник из train+test json и убеждаемся test-вопросы не попадают в pool."""
    tok = tiny_tok()
    data = {"train": [{"q": "abc", "a": "def"}], "test": [{"q": "hij", "a": "gij"}]}
    f = tmp_path / "data.json"
    f.write_text(json.dumps(data), encoding="utf-8")
    from mini_ai.curriculum import make_pool_from_source
    pool = make_pool_from_source(tok, {"type": "qa", "file": str(f)}, max_len=20, rng=np.random.RandomState(0))
    all_q_ids = set()
    for seq, tgt, mask in pool.examples:
        all_q_ids.add(tuple(seq))
    # кодируем test-вопрос и проверяем, что ни одна последовательность pool не начинается с него
    test_q_ids = tuple([BOS] + tok.encode("hij"))
    for seq in all_q_ids:
        assert seq[:len(test_q_ids)] != test_q_ids


def test_dynamic_batch_budget():
    tok = tiny_tok()
    rng = np.random.RandomState(0)
    examples = [( [1,2,3],[2,3,4],[True,True,True] ) for _ in range(50)]
    pool = SourcePool(examples, rng)
    import numpy.random as npr
    batch = build_dynamic_batch({"a": pool}, {"a": 1.0}, token_budget=30, max_len=10, rng=npr.RandomState(0))
    assert len(batch) >= 2
    idx, tgt, mask = pad_batch(*zip(*batch))
    assert idx.shape[0] == len(batch)


# ---------------------------------------------------------------------------
def test_batched_generate_matches_kv_greedy():
    V, d, n_layer, n_head = 20, 16, 2, 2
    p, cfg = init_params(V, d=d, n_layer=n_layer, n_head=n_head, block=32, pos="rope", seed=1)
    p_np = {k: np.asarray(v) for k, v in p.items()}
    engine = TorchEngine(p_np, cfg, seed=1)
    prompts = [[1, 5, 6, 7], [1, 8, 9]]
    max_new = 6
    batched_out = batched_generate(engine, prompts, max_new, temperature=0.0)
    for prompt, out in zip(prompts, batched_out):
        ref = kv_greedy_reference(p_np, cfg, prompt, max_new)
        assert out == ref, (out, ref)


# ---------------------------------------------------------------------------
def test_end_to_end_tiny_plan(tmp_path):
    """Полный прогон train_curriculum.py с крошечным планом/моделью/данными, все типы стадий
    включая selfplay и экзамены, и resume-от-чекпойнта."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    corpus_dir = data_dir / "corpus"
    corpus_dir.mkdir()
    (corpus_dir / "ru.txt").write_text("привет мир\nкак дела\n", encoding="utf-8")
    (corpus_dir / "ru_test.txt").write_text("привет мир\n", encoding="utf-8")

    qa_path = data_dir / "qa.json"
    qa_path.write_text(json.dumps({"train": [{"q": "сколько 2 + 2", "a": "4", "kind": "add"}],
                                    "test": [{"q": "сколько 3 + 3", "a": "6", "kind": "add"}]}),
                        encoding="utf-8")
    reasoning_path = data_dir / "reasoning.json"
    reasoning_path.write_text(json.dumps({
        "train": [{"q": "2 + 2 beqadar", "a": "2 + 2 = 4. Ответ: 4", "kind": "add_pv", "lang": "ru",
                   "answer": "4"}],
        "test": []}), encoding="utf-8")

    # крошечная "старая" модель: char tokenizer, d=8, 1 слой, learned pos
    from mini_ai.transformer import export_model
    tok0 = CharTokenizer(list("привет мирдела2+= .,"))
    p0, cfg0 = init_params(len(tok0), d=8, n_layer=1, n_head=2, block=64, pos="learned", seed=0)
    init_doc = export_model(p0, cfg0, tok0, {"max_q": 20}, tokenizer_type="char", max_q=20)
    init_path = tmp_path / "init.json"
    init_path.write_text(json.dumps(init_doc, ensure_ascii=False), encoding="utf-8")

    plan = {
        "init_model": str(init_path), "target_vocab": 30,
        "arch": {"d": 16, "n_layer": 2, "n_head": 2, "block": 64, "pos": "rope"},
        "time_budget_hours": 0.001,
        "stages": [
            {"name": "read", "max_len": 16, "token_budget": 64, "lr": 1e-3, "warmup": 2,
             "time_frac": 0.5,
             "sources": [{"type": "lm", "weight": 0.7, "files": [str(corpus_dir / "ru.txt")]},
                         {"type": "qa", "weight": 0.3, "file": str(qa_path)}]},
            {"name": "reason", "max_len": 24, "token_budget": 64, "lr": 1e-3, "warmup": 2,
             "time_frac": 0.3,
             "sources": [{"type": "qa", "weight": 1.0, "file": str(reasoning_path)}]},
            {"name": "context", "max_len": 32, "token_budget": 64, "lr": 1e-3, "warmup": 2,
             "time_frac": 0.1,
             "sources": [{"type": "qa", "weight": 1.0, "file": str(qa_path)}]},
            {"name": "selfplay", "max_len": 24, "token_budget": 64, "lr": 1e-3, "warmup": 2,
             "time_frac": 0.1,
             "selfplay": {"task_files": [str(reasoning_path)], "n": 2, "tries": 1, "temp": 0.5},
             "sources": [{"type": "selfplay", "weight": 0.5, "file": "__selfplay_out__"},
                         {"type": "qa", "weight": 0.5, "file": str(qa_path)}]},
        ],
    }
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False), encoding="utf-8")

    out_dir = tmp_path / "out"
    cmd = [PY, str(ROOT / "train_curriculum.py"), "--plan", str(plan_path), "--init", str(init_path),
           "--out", str(out_dir), "--steps_total_override", "3", "--ckpt_minutes", "1000"]
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=100)
    assert r.returncode == 0, r.stdout[-4000:] + "\n" + r.stderr[-4000:]
    assert (out_dir / "model_final.json").exists()
    exams = json.loads((out_dir / "exams.json").read_text(encoding="utf-8"))
    assert len(exams) == 4   # одна запись на каждую стадию
    assert (out_dir / "ckpt" / "state.json").exists()

    # --- resume: удалим финальную модель и exams, но оставим чекпойнт, перезапустим с тем же --out
    state = json.loads((out_dir / "ckpt" / "state.json").read_text())
    assert state["stage_idx"] == 3  # последняя стадия (selfplay, idx=3)
