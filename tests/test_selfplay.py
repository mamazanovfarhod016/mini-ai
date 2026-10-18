"""Самоучёба: верные решения проверяет скрипт; экзаменационные задания в самоучёбу не попадают."""
import json

import numpy as np

from mini_ai.bpe import BPETokenizer
from mini_ai.transformer import export_model, init_params
from server import selfplay


def test_final_answer_and_simple_check():
    t = "57 − 19: 57 − 10 = 47, 47 − 9 = 38. Проверка: 38 + 19 = 57. Ответ: 38"
    assert selfplay.final_answer(t) == "38"
    item = {"answer": "38"}
    assert selfplay.simple_check(item, t)["passed"]
    assert not selfplay.simple_check(item, t.replace("47 − 9 = 38", "47 − 9 = 39"))["passed"]   # неверный шаг
    assert not selfplay.simple_check(item, t.replace("Ответ: 38", "Ответ: 37"))["passed"]       # неверный ответ
    assert selfplay.simple_check({"answer": "3/4"}, "1/4 + 2/4 = 3/4. Answer: 3/4")["passed"]


def test_run_uses_only_train_and_writes_stats(tmp_path):
    tok = BPETokenizer().train(["сколько будет 2 + 2 ответ 4"] * 5, vocab_size=60)
    p, cfg = init_params(len(tok), d=16, n_layer=1, n_head=2, block=40, seed=0)
    m = tmp_path / "m.json"
    m.write_text(json.dumps(export_model(p, cfg, tok, {}, tokenizer_type="bpe", max_q=12, merges=tok.merges)))
    tasks = tmp_path / "t.json"
    tasks.write_text(json.dumps({"train": [{"q": "сколько будет 2 + 2", "a": "Ответ: 4", "answer": "4", "kind": "add"}],
                                 "test": [{"q": "СЕКРЕТ", "a": "x", "answer": "x", "kind": "add"}]}))
    out = tmp_path / "sp.json"
    selfplay.checker = lambda: selfplay.simple_check     # без reasoncheck — проще
    good, fail = selfplay.run(str(m), str(tasks), str(out), n=5, tries=2, log=lambda *_: None)
    assert all(g["q"] != "СЕКРЕТ" for g in good)
    stats = json.loads(out.with_suffix(".stats.json").read_text())
    assert stats["tried"]["add"] == 5 and "add" in fail
    assert isinstance(np.float64(fail["add"]), np.floating)
