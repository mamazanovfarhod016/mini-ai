"""Паритет Python↔JS для диалога (история 8 реплик, промпт, анти-повтор): node tests/js_dialog_think.js.
Проверяет:
  - promptIdsDialog(JS) == [BOS] + encode_dialog_prompt_ids(Python) + [SEP], несколько случаев
    (история пустая/короче 8/длиннее 8 реплик — обрезка слева целыми репликами);
  - applyAntiRepeat(JS) даёт те же логиты, что apply_anti_repeat_ (Python) для repetition penalty
    и no-repeat-ngram-3 на синтетических логитах/истории токенов.
Используем крошечный char-токенизатор (BPETokenizer с vocab_size маленьким) — не нужна обученная
модель, только совпадение токенизации/математики.
"""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mini_ai.bpe import BPETokenizer  # noqa: E402
from mini_ai.transformer import BOS, SEP, export_model, init_params, n_params  # noqa: E402


def _tiny_model_path(tmp_path):
    texts = ["U: привет\nA: привет! как дела?\nU: норм\nA: рада слышать",
              "U: кто ты\nA: я мини ai", "реплика текст вопрос ответ"]
    tok = BPETokenizer()
    tok.train(texts, vocab_size=120)
    p, cfg = init_params(len(tok), d=8, n_layer=1, n_head=2, block=256, pos="rope")
    doc = export_model(p, cfg, tok, {"max_q": 40}, tokenizer_type="bpe", max_q=40, merges=tok.merges)
    path = tmp_path / "m.json"
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
    return path, tok


@pytest.mark.skipif(shutil.which("node") is None, reason="нет node")
def test_python_js_dialog_prompt_parity(tmp_path):
    from mini_ai.curriculum import encode_prompt
    from mini_ai.dialog import encode_dialog_prompt_ids

    model_path, tok = _tiny_model_path(tmp_path)

    cases = [
        {"history": [], "last_q": "привет", "max_history_turns": 8},
        {"history": [{"role": "user", "text": "привет"}, {"role": "bot", "text": "привет! как дела?"}],
         "last_q": "норм", "max_history_turns": 8},
        {"history": [{"role": "user" if i % 2 == 0 else "bot", "text": f"реплика {i}"} for i in range(12)],
         "last_q": "вопрос", "max_history_turns": 8},
    ]

    def enc(text):
        return encode_prompt(tok, text, raw=True)

    py_prompts = []
    for c in cases:
        hist = c["history"][-c["max_history_turns"]:] if len(c["history"]) > c["max_history_turns"] else c["history"]
        ids = encode_dialog_prompt_ids(tok, hist, c["last_q"], enc)
        py_prompts.append([BOS] + ids + [SEP])

    out = subprocess.run(
        ["node", str(ROOT / "tests/js_dialog_think.js"), str(model_path)],
        input=json.dumps({"cases": cases}), capture_output=True, text=True, timeout=30,
    )
    assert out.returncode == 0, out.stderr
    js_result = json.loads(out.stdout)
    assert js_result["prompt_ids"] == py_prompts


@pytest.mark.skipif(shutil.which("node") is None, reason="нет node")
def test_python_js_antirepeat_parity(tmp_path):
    import torch
    from mini_ai.batched_generate import apply_anti_repeat_

    model_path, tok = _tiny_model_path(tmp_path)
    vocab_size = len(tok)

    rng = np.random.RandomState(0)
    cases = []
    for gen_ids in ([], [1, 2, 3], [1, 2, 3, 1, 2]):
        logits = rng.uniform(-2, 2, size=vocab_size).tolist()
        cases.append({"logits": logits, "gen_ids": gen_ids, "rp": 1.3, "ngram": 3})

    py_results = []
    for c in cases:
        t = torch.tensor([c["logits"]], dtype=torch.float32)
        apply_anti_repeat_(t, [c["gen_ids"]], c["rp"], c["ngram"])
        py_results.append(t[0].tolist())

    out = subprocess.run(
        ["node", str(ROOT / "tests/js_dialog_think.js"), str(model_path)],
        input=json.dumps({"antirepeat_cases": cases}), capture_output=True, text=True, timeout=30,
    )
    assert out.returncode == 0, out.stderr
    js_result = json.loads(out.stdout)["antirepeat_logits"]
    for py, js in zip(py_results, js_result):
        assert np.allclose(py, js, atol=1e-4)
