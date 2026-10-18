"""
Тесты BPE-токенизатора Mini AI v0.3 (mini_ai/bpe.py).

Проверяем:
  1. encode -> decode восстанавливает исходный текст для ВСЕХ вопросов и
     ответов data/intents.json (на корпусе, на котором токенизатор обучен);
  2. Python encode == JS encode (app/www/bpe.js через tests/js_bpe_encode.js)
     на ВСЕХ фразах и ответах — 100% совпадение id;
  3. (если BPE-модель уже обучена) логиты и greedy-ответ Python == JS на
     20 вопросах — tests/js_bpe_think.js.
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

from mini_ai.bpe import BPETokenizer, TIME_TOK, DATE_TOK  # noqa: E402
from mini_ai.transformer import clean_question  # noqa: E402

BPE_MODEL_PATH = ROOT / "model_transformer_bpe.json"


def _corpus_texts():
    """Весь текст intents.json: очищенные вопросы + ответы (с ①②)."""
    data = json.loads((ROOT / "data/intents.json").read_text(encoding="utf-8"))
    texts = []
    for it in data["intents"]:
        for p in it["patterns"]:
            texts.append(clean_question(p))
        for lang, answers in it["responses"].items():
            for a in answers:
                texts.append(a.replace("{time}", TIME_TOK).replace("{date}", DATE_TOK))
    return texts


@pytest.fixture(scope="module")
def tok_and_texts():
    texts = _corpus_texts()
    tok = BPETokenizer()
    tok.train(texts, vocab_size=1000)
    return tok, texts


# ======================================================================================
# 1. encode -> decode == исходный текст
# ======================================================================================
def test_encode_decode_roundtrip_all_corpus(tok_and_texts):
    tok, texts = tok_and_texts
    bad = []
    for t in texts:
        got = tok.decode(tok.encode(t))
        if got != t:
            bad.append((t, got))
    assert not bad, f"{len(bad)} несовпадений, пример: {bad[:5]}"


def test_no_unk_on_training_corpus(tok_and_texts):
    """На своём обучающем корпусе <unk> не должен появляться (все символы известны)."""
    from mini_ai.bpe import UNK
    tok, texts = tok_and_texts
    for t in texts:
        assert UNK not in tok.encode(t), t


def test_specials_and_sentinels_fixed_ids():
    tok = BPETokenizer()
    assert tok.itos[:5] == ["<pad>", "<bos>", "<sep>", "<eos>", "<unk>"]


def test_time_date_tokens_not_merged(tok_and_texts):
    """① и ② — отдельные неделимые токены, не участвуют ни в одном merge."""
    tok, texts = tok_and_texts
    for a, b in tok.merges:
        assert TIME_TOK not in (a, b), "TIME_TOK не должен участвовать в merge"
        assert DATE_TOK not in (a, b), "DATE_TOK не должен участвовать в merge"


# ======================================================================================
# 2. Python encode == JS encode (100% совпадение на всём корпусе)
# ======================================================================================
@pytest.mark.skipif(shutil.which("node") is None, reason="нет node")
def test_python_js_bpe_encode_parity(tok_and_texts, tmp_path):
    tok, texts = tok_and_texts
    model_path = tmp_path / "bpe_tok.json"
    model_path.write_text(json.dumps(tok.to_dict(), ensure_ascii=False), encoding="utf-8")

    out = subprocess.run(
        ["node", str(ROOT / "tests/js_bpe_encode.js"), str(model_path)],
        input=json.dumps(texts, ensure_ascii=False),
        capture_output=True, text=True, check=True,
    ).stdout
    js_results = json.loads(out)
    assert len(js_results) == len(texts)

    mismatches = []
    for t, js in zip(texts, js_results):
        py_ids = tok.encode(t)
        if py_ids != js["ids"]:
            mismatches.append((t, py_ids, js["ids"]))
    assert not mismatches, f"{len(mismatches)} несовпадений encode, пример: {mismatches[:3]}"

    decode_mismatches = []
    for t, js in zip(texts, js_results):
        py_decoded = tok.decode(tok.encode(t))
        if py_decoded != js["decoded"]:
            decode_mismatches.append((t, py_decoded, js["decoded"]))
    assert not decode_mismatches, f"{len(decode_mismatches)} несовпадений decode"


# ======================================================================================
# 3. (если BPE-модель обучена) логиты и greedy-ответ Python == JS на 20 вопросах
# ======================================================================================
@pytest.mark.skipif(shutil.which("node") is None, reason="нет node")
def test_python_js_bpe_transformer_parity():
    if not BPE_MODEL_PATH.exists():
        pytest.skip("model_transformer_bpe.json ещё не обучена")
    from train_transformer import build_prompt
    from mini_ai.transformer import KVModel, load_model

    data = json.loads((ROOT / "data/intents.json").read_text(encoding="utf-8"))
    questions = []
    for it in data["intents"][:10]:
        questions += it["patterns"][:2]
    questions = questions[:20]

    out = subprocess.run(
        ["node", str(ROOT / "tests/js_bpe_think.js"), str(BPE_MODEL_PATH)],
        input=json.dumps(questions, ensure_ascii=False),
        capture_output=True, text=True, check=True,
    ).stdout
    js_results = json.loads(out)

    p, cfg, tok, meta = load_model(str(BPE_MODEL_PATH))
    kv = KVModel(p, cfg)
    assert len(js_results) == len(questions)
    for q, js in zip(questions, js_results):
        ids = build_prompt(tok, q, meta.get("max_q", 24))
        out_ids, info = kv.generate(ids, max_new=cfg["block"] - len(ids), temperature=0.0)
        py_text = tok.decode(out_ids)
        assert py_text == js["answer"], (q, py_text, js["answer"])
        py_logits = info["logits"][0] if info["logits"] else None
        if py_logits is not None and js.get("firstLogits"):
            py = np.asarray(py_logits, dtype=np.float64)
            jsv = np.asarray(js["firstLogits"], dtype=np.float64)
            assert np.max(np.abs(py - jsv)) < 1e-3, (q, "logits mismatch")
