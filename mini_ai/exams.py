"""
Скриптовые экзамены после каждой стадии curriculum. Всё проверяется кодом, без ИИ.

run_exams(engine, tok, cfg, tokenizer_type, max_q, exam_data, log=print) -> dict
Результат пишется целиком в exams.json (список записей по стадиям).
"""
import math
import random
from pathlib import Path

from mini_ai.curriculum import decode_answer, encode_prompt, load_lm_lines, pack_lm_windows
from mini_ai.batched_generate import batched_generate
from mini_ai.transformer import BOS, SEP, clean_question


def lm_perplexity(engine, tok, files, max_len=256, batch=8, raw=True, max_windows=200, seed=0):
    """Перплексия по *_test.txt: упаковываем в окна, считаем masked_ce (как в тренировке),
    усредняем по токенам. Один язык = один файл (или список)."""
    import torch
    from mini_ai.torch_engine import forward as torch_forward, masked_ce
    rng = random.Random(seed)
    lines = load_lm_lines(files if isinstance(files, (list, tuple)) else [files])
    if not lines:
        return None
    windows = pack_lm_windows(tok, lines, max_len, rng, raw=raw)
    windows = windows[:max_windows]
    if not windows:
        return None
    total_loss, total_tok = 0.0, 0
    for s in range(0, len(windows), batch):
        chunk = windows[s:s + batch]
        T = max(len(w) for w in chunk) - 1
        import numpy as np
        idx = np.zeros((len(chunk), T), dtype=np.int64)
        tgt = np.zeros((len(chunk), T), dtype=np.int64)
        mask = np.zeros((len(chunk), T), dtype=bool)
        for i, w in enumerate(chunk):
            L = len(w) - 1
            idx[i, :L] = w[:-1]
            tgt[i, :L] = w[1:]
            mask[i, :L] = True
        logits = torch_forward(engine.p, engine.cfg, engine._t(idx, torch.long), 0.0, None)
        loss = masked_ce(logits, engine._t(tgt, torch.long), engine._t(mask, torch.bool))
        n = int(mask.sum())
        total_loss += float(loss) * n
        total_tok += n
    avg = total_loss / max(1, total_tok)
    return {"loss": avg, "ppl": math.exp(min(avg, 30)), "n_tokens": total_tok}


def exam_qa_exact(engine, tok, items, max_q=40, batch=32, max_new=60, raw=False):
    """items: [{"q":..,"a":..,"kind":..}]. Точное совпадение сгенерированного ответа с эталоном.
    Возвращает (общая доля, по видам {kind: (hits, n)}, список ошибок)."""
    by_kind, wrong = {}, []
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        prompts = [[BOS] + encode_prompt(tok, it["q"], raw)[:max_q] + [SEP] for it in chunk]
        outs = batched_generate(engine, prompts, max_new, temperature=0.0)
        for it, out in zip(chunk, outs):
            gen = decode_answer(tok, out).strip()
            ok = gen == str(it["a"]).strip() or gen in [str(x).strip() for x in it.get("answers", [])]   # разговор: любой ответ темы
            kind = it.get("kind", "default")
            h, n = by_kind.get(kind, (0, 0))
            by_kind[kind] = (h + ok, n + 1)
            if not ok and len(wrong) < 20:
                wrong.append({"q": it["q"], "expect": it["a"], "got": gen})
    hits = sum(h for h, _ in by_kind.values())
    total = sum(n for _, n in by_kind.values())
    return hits / max(1, total), by_kind, wrong


def exam_reasoning(engine, tok, items, max_q=40, batch=16, max_new=200, raw=False):
    from server import reasoncheck
    results = {"answer_ok": 0, "steps_ok": 0, "check_ok": 0, "n": 0}
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        prompts = [[BOS] + encode_prompt(tok, it["q"], raw)[:max_q] + [SEP] for it in chunk]
        outs = batched_generate(engine, prompts, max_new, temperature=0.0)
        for it, out in zip(chunk, outs):
            gen = decode_answer(tok, out)
            r = reasoncheck.check(it, gen, it.get("lang", "ru"))
            results["n"] += 1
            results["answer_ok"] += bool(r.get("answer_ok"))
            results["steps_ok"] += bool(r.get("steps_ok"))
            results["check_ok"] += bool(r.get("check_ok", True))
    n = max(1, results["n"])
    return {k: (v / n if k != "n" else v) for k, v in results.items()}


def exam_context(engine, tok, items, max_q=400, batch=4, max_new=60, raw=True):
    from server import contextcheck
    by_kind = {}
    by_len_bucket = {}
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        prompts = []
        for it in chunk:
            ctx_ids = encode_prompt(tok, it["context"], raw=True)
            q_ids = encode_prompt(tok, "\n" + it["q"], True)
            prompts.append([BOS] + ctx_ids + q_ids + [SEP])
        outs = batched_generate(engine, prompts, max_new, temperature=0.0)
        for it, out in zip(chunk, outs):
            gen = decode_answer(tok, out)
            r = contextcheck.check(it, gen)
            kind = it.get("kind", "fact")
            h, n = by_kind.get(kind, (0, 0))
            by_kind[kind] = (h + r["ok"], n + 1)
            if kind == "needle":
                nb = it.get("n_chars", 0) // 2000
                h2, n2 = by_len_bucket.get(nb, (0, 0))
                by_len_bucket[nb] = (h2 + r["ok"], n2 + 1)
    return {"by_kind": {k: h / max(1, n) for k, (h, n) in by_kind.items()},
            "needle_by_len_bucket": {k: h / max(1, n) for k, (h, n) in by_len_bucket.items()}}


def exam_transfer(engine, tok, test_items, max_q=40, batch=32, max_new=20, raw=False):
    from tools.make_transfer import score
    results = []
    for s in range(0, len(test_items), batch):
        chunk = test_items[s:s + batch]
        prompts = [[BOS] + encode_prompt(tok, it["q"], raw)[:max_q] + [SEP] for it in chunk]
        outs = batched_generate(engine, prompts, max_new, temperature=0.0)
        for it, out in zip(chunk, outs):
            results.append((it, decode_answer(tok, out)))
    return score(results)


def tokenizer_is_lossless(tok, sample_texts):
    """Детектируем lossless по round-trip: decode(encode(text)) == text для примеров кода."""
    for t in sample_texts:
        ids = tok.encode(t)
        if tok.decode(ids) != t:
            return False
    return True


def run_exams(engine, tok, tokenizer_type, max_q, exam_data, log=print):
    """exam_data: словарь с путями/данными, собранными вызывающим кодом (train_curriculum.py).
    Ключи (все опциональны — чего нет, то пропускается с меткой 'skipped'):
      lm_test_files: {lang: [path,...]}
      school_math_test, reasoning_test, context_test, transfer_test, intents_test: списки
      python_tasks_test: список (только если tokenizer lossless)
    """
    out = {}
    if exam_data.get("lm_test_files"):
        out["lm_ppl"] = {}
        for lang, files in exam_data["lm_test_files"].items():
            try:
                out["lm_ppl"][lang] = lm_perplexity(engine, tok, files)
            except Exception as e:
                out["lm_ppl"][lang] = {"error": str(e)}
        log(f"  экзамен LM ppl: {out['lm_ppl']}")
    if exam_data.get("school_math_test"):
        acc, by_kind, wrong = exam_qa_exact(engine, tok, exam_data["school_math_test"], max_q)
        out["school_math"] = {"acc": acc, "by_kind": {k: h / n for k, (h, n) in by_kind.items()},
                               "wrong_sample": wrong[:5]}
        log(f"  экзамен school_math: acc={acc:.3f}")
    if exam_data.get("reasoning_test"):
        out["reasoning"] = exam_reasoning(engine, tok, exam_data["reasoning_test"], max_q)
        log(f"  экзамен reasoning: {out['reasoning']}")
    if exam_data.get("context_test"):
        out["context"] = exam_context(engine, tok, exam_data["context_test"])
        log(f"  экзамен context: {out['context']}")
    if exam_data.get("transfer_test"):
        out["transfer"] = exam_transfer(engine, tok, exam_data["transfer_test"])
        log(f"  экзамен transfer: {out['transfer']}")
    if exam_data.get("intents_test"):
        acc, by_kind, wrong = exam_qa_exact(engine, tok, exam_data["intents_test"], max_q)
        out["intents"] = {"acc": acc, "wrong_sample": wrong[:5]}
        log(f"  экзамен intents: acc={acc:.3f}")
    if exam_data.get("python_tasks_test"):
        out["python_tasks"] = exam_python(engine, tok, exam_data["python_tasks_test"])
        log(f"  экзамен python_tasks: {out['python_tasks']}")
    return out


def exam_python(engine, tok, items, batch=8, max_new=220, max_q=200):
    """Python: модель пишет код → server/codecheck.check: скрытые тесты + оценка качества (passed = тесты И качество ≥ 80).
    Вопрос подаём raw (без clean_question — иначе код в вопросе испортится)."""
    from mini_ai.batched_generate import batched_generate
    from server import codecheck
    by = {}
    passed = runs = 0
    qual = []
    for s in range(0, len(items), batch):
        chunk = items[s:s + batch]
        prompts = [[BOS] + encode_prompt(tok, it["q"], True)[:max_q] + [SEP] for it in chunk]
        outs = batched_generate(engine, prompts, max_new=max_new, temperature=0.0)
        for it, ids in zip(chunk, outs):
            code = decode_answer(tok, ids)
            r = codecheck.check(it, code)
            passed += bool(r.get("passed"))
            runs += bool(r.get("runs"))
            qual.append(r.get("quality", 0))
            h, n = by.get(it["kind"], (0, 0))
            by[it["kind"]] = (h + bool(r.get("passed")), n + 1)
    n = max(1, len(items))
    return {"passed": passed / n, "runs": runs / n, "avg_quality": sum(qual) / max(1, len(qual)),
            "by_kind": {k: [h, m] for k, (h, m) in by.items()}}
