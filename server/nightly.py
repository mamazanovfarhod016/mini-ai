"""
Ночное самообучение Mini AI (запускается на сервере по cron).

    .venv/bin/python -m server.nightly [--steps 300] [--data server_data]

1. Собирает новые пары для обучения:
   • ответы наставника (server_data/teach_queue.json);
   • ответы из интернета, которые ПОЛЬЗОВАТЕЛЬ подтвердил кнопкой «Верно» (questions.jsonl, web_confirmed).
2. Если новых пар нет — ничего не делает.
3. Дообучает текущую BPE-модель (fine-tune) с replay старых примеров (против забывания).
4. ЭКЗАМЕН на held-out тест-вопросах (тот же детерминированный сплит, что в train_transformer.py):
   новая модель публикуется, только если точность НЕ упала больше чем на 1 п.п.
   и модель выучила хотя бы половину новых пар. Иначе — отклоняется, остаётся старая.
5. Публикует: server_data/models/model_vN.json + latest.json (его скачивает приложение).
Всё пишется в server_data/nightly.log.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mini_ai.transformer import (  # noqa: E402
    AdamW, KVModel, clean_question, clip_grads, export_model, load_model,
    loss_and_grads, lr_schedule,
)
from server import store  # noqa: E402
from train_transformer import build_pairs, build_prompt, fill_placeholders, intent_of_answer, make_batch  # noqa: E402

BASE_MODEL = ROOT / "model_transformer_bpe.json"


def log(data, msg, quiet=False):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    if not quiet:          # cron доставляет stdout в Telegram — «нечего учить» не шлём
        print(line, flush=True)
    with open(Path(data) / "nightly.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def collect_new_pairs(data):
    pairs = [{"q": x["q"], "a": x["a"], "from": x.get("from", "mentor")} for x in store.read_json(data / "teach_queue.json", [])]
    for r in store.read_jsonl(data / "questions.jsonl"):
        if r.get("web_confirmed") and r.get("a"):
            pairs.append({"q": r["q"], "a": r["a"], "from": "internet"})
    used = set(store.read_json(data / "learned_keys.json", []))
    by_q = {}
    for p in pairs:
        if len(p["a"]) > 200:
            continue                              # длинные абзацы маленькая модель не выучит
        by_q[clean_question(p["q"])] = p          # последний ответ на тот же вопрос — главный
    return [p for k, p in by_q.items() if k + "→" + p["a"] not in used]


def exam(p, cfg, tok, max_q, test_qs, intents_by_tag):
    kv = KVModel(p, cfg)
    hits = 0
    for q, tag in test_qs:
        ids = build_prompt(tok, q, max_q)
        out, _ = kv.generate(ids, max_new=cfg["block"] - len(ids), temperature=0.0)
        hits += intent_of_answer(tok.decode(out), intents_by_tag) == tag
    return hits / len(test_qs)


def knows(p, cfg, tok, max_q, pairs):
    kv = KVModel(p, cfg)
    ok = 0
    for x in pairs:
        ids = build_prompt(tok, x["q"], max_q)
        out, _ = kv.generate(ids, max_new=cfg["block"] - len(ids), temperature=0.0)
        ok += tok.decode(out).strip() == fill_placeholders(x["a"]).strip()
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "server_data"))
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--max-drop", type=float, default=0.01, help="допустимое падение точности на экзамене")
    args = ap.parse_args()
    data = Path(args.data)
    (data / "models").mkdir(parents=True, exist_ok=True)

    latest = store.read_json(data / "models/latest.json", None)
    model_path = data / "models" / latest["file"] if latest else BASE_MODEL
    p, cfg, tok, meta = load_model(str(model_path))
    if meta.get("tokenizer") != "bpe":
        sys.exit("нужна BPE-модель")
    max_q = meta.get("max_q", 24)

    new_pairs = collect_new_pairs(data)
    # пары, которые модель НЕ может закодировать без <unk> (незнакомые буквы) — пропускаем честно
    from mini_ai.bpe import UNK
    skipped = [x for x in new_pairs if UNK in tok.encode(fill_placeholders(x["a"]))]
    new_pairs = [x for x in new_pairs if x not in skipped]
    for x in skipped:
        log(data, f"пропуск (незнакомые символы): «{x['q']}» → «{x['a']}»")
    if not new_pairs:
        log(data, "новых пар нет — обучение не нужно", quiet=True)
        return

    intents = json.loads((ROOT / "data/intents.json").read_text(encoding="utf-8"))["intents"]
    intents_by_tag = {it["tag"]: it for it in intents}
    train_pairs, test_pairs = build_pairs(intents, random.Random(0), split_seed=0)
    test_qs = sorted({(q, tag) for q, a, tag, lang in test_pairs})
    replay = [(q, a) for q, a, *_ in train_pairs]
    replay += [(x["q"], x["a"]) for x in store.read_json(data / "learned_pairs.json", [])]   # не забыть прошлые уроки

    before = exam(p, cfg, tok, max_q, test_qs, intents_by_tag)
    before_new = knows(p, cfg, tok, max_q, new_pairs)
    log(data, f"новых пар: {len(new_pairs)}; экзамен ДО: {before*100:.1f}%; новых знает ДО: {before_new}/{len(new_pairs)}")

    new_ex = [(x["q"], x["a"]) for x in new_pairs]
    opt = AdamW(p, lr=args.lr, wd=0.01)
    rng = np.random.default_rng(int(time.time()))
    t0 = time.time()
    for step in range(args.steps):
        k = 12                                                   # 12 новых + 20 replay в каждом батче
        batch = [new_ex[i] for i in rng.integers(0, len(new_ex), k)] + [replay[i] for i in rng.integers(0, len(replay), 20)]
        idx, targets, mask = make_batch(tok, batch, max_q, cfg["block"], rng)
        loss, grads = loss_and_grads(p, cfg, idx, targets, mask, drop=0.05, rng=rng)
        clip_grads(grads, 1.0)
        opt.step(p, grads, lr_schedule(step, args.steps, args.lr, max(10, args.steps // 10)))
    log(data, f"дообучение: {args.steps} шагов за {(time.time()-t0)/60:.1f} мин, loss={loss:.3f}")

    after = exam(p, cfg, tok, max_q, test_qs, intents_by_tag)
    after_new = knows(p, cfg, tok, max_q, new_pairs)
    passed = after >= before - args.max_drop and after_new >= (len(new_pairs) + 1) // 2
    log(data, f"экзамен ПОСЛЕ: {after*100:.1f}% (было {before*100:.1f}%); новых знает: {after_new}/{len(new_pairs)} → "
              + ("СДАЛА, публикую" if passed else "НЕ СДАЛА, остаётся старая модель"))
    if not passed:
        return

    version = (latest["version"] if latest else 0) + 1
    m = dict(meta)
    for k in ("format", "config", "vocab", "max_q", "order", "shapes", "weights", "tokenizer", "merges"):
        m.pop(k, None)
    m.update({"model_version": version, "test_intent_accuracy": round(after, 4),
              "self_learned_pairs": len(new_pairs), "trained_at": time.strftime("%Y-%m-%d %H:%M:%S")})
    out = export_model(p, cfg, tok, m, tokenizer_type="bpe", max_q=max_q, merges=tok.merges)
    fname = f"model_v{version}.json"
    (data / "models" / fname).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    store.write_json(data / "models/latest.json", {"version": version, "file": fname,
                                                   "test_intent_acc": round(after, 4), "at": m["trained_at"]})
    learned = store.read_json(data / "learned_pairs.json", []) + [{"q": x["q"], "a": x["a"]} for x in new_pairs]
    store.write_json(data / "learned_pairs.json", learned)
    store.write_json(data / "learned_keys.json",
                     sorted(set(store.read_json(data / "learned_keys.json", [])) |
                            {clean_question(x["q"]) + "→" + x["a"] for x in new_pairs}))
    store.write_json(data / "teach_queue.json", [])
    log(data, f"опубликована версия {version} ({fname})")


if __name__ == "__main__":
    main()
