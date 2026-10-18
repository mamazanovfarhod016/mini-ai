"""
Самообучение Mini AI на сервере: темы с формулировками, экзамен на НЕВИДАННЫХ формулировках.

    .venv/bin/python -m server.nightly            # ночью: дообучение (fine-tune) на новых темах
    .venv/bin/python -m server.nightly --full     # раз в неделю: обучение с нуля на всём знании,
                                                  #   модель побольше, если знаний стало много

Что считается «поняла», а не «запомнила»:
  • модель учится на ~70% формулировок каждой темы (server/knowledge.py);
  • экзамен — на оставшихся 30%, которых она НЕ видела, + на 322 старых тест-вопросах.
Новая версия публикуется, только если
  1) старые знания не просели больше чем на 1 п.п. (экзамен на 322 вопросах),
  2) понимание НОВЫХ тем на невиданных формулировках не хуже прежней версии,
  3) (для --full) общий балл выше, чем у текущей модели.
Иначе модель отклоняется. История экзаменов → server_data/progress.json → график в приложении.
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mini_ai.transformer import (  # noqa: E402
    AdamW, KVModel, clip_grads, export_model, load_model, loss_and_grads, lr_schedule, n_params,
)
from server import knowledge, mathcheck, store  # noqa: E402
from train_transformer import build_pairs, build_prompt, fill_placeholders, is_latin, make_batch  # noqa: E402

BASE_MODEL = ROOT / "model_transformer_math.json"
if not BASE_MODEL.exists():
    BASE_MODEL = ROOT / "model_transformer_bpe.json"
MATH = ROOT / "data/math_pairs.json"


def math_sets():
    if not MATH.exists():
        return [], []
    m = json.loads(MATH.read_text(encoding="utf-8"))
    rnd = random.Random(7)
    return [(x["q"], x["a"]) for x in m["train"]], rnd.sample([(x["q"], x["a"]) for x in m["test"]], 300)


def log(data, msg, quiet=False):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    if not quiet:            # stdout → Telegram (cron); «нечего учить» не шлём
        print(line, flush=True)
    with open(Path(data) / "nightly.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def gen(kv, tok, cfg, max_q, q):
    ids = build_prompt(tok, q, max_q)
    out, _ = kv.generate(ids, max_new=cfg["block"] - len(ids), temperature=0.0)
    return tok.decode(out).strip()


def answer_set(responses):
    return {fill_placeholders(a).strip() for lst in responses.values() for a in lst}


def exam(p, cfg, tok, max_q, base_tests, topics, base_resp):
    """→ dict: base (старые 322), new (невиданные формулировки новых тем), details."""
    kv = KVModel(p, cfg)
    hits = 0
    for q, tag in base_tests:
        hits += gen(kv, tok, cfg, max_q, q) in base_resp[tag]
    new_total, new_hits, details = 0, 0, []
    for t in topics:
        ok_set = answer_set(t["responses"])
        for q in t["heldout"]:
            a = gen(kv, tok, cfg, max_q, q)
            ok = a in ok_set
            new_total += 1; new_hits += ok
            details.append({"tag": t["tag"], "q": q, "a": a, "ok": ok})
    math_test = math_sets()[1]
    m_hits = sum(gen(kv, tok, cfg, max_q, q) == a for q, a in math_test)
    return {"base": hits / len(base_tests), "math": (m_hits / len(math_test)) if math_test else None,
            "new": (new_hits / new_total) if new_total else None,
            "new_hits": new_hits, "new_total": new_total, "details": details}


def topic_pairs(topics):
    out = []
    for t in topics:
        for q in t["train"]:
            lang = "uz" if is_latin(q) else "ru"
            answers = t["responses"].get(lang) or t["responses"].get("ru") or t["responses"].get("uz")
            out += [(q, a) for a in answers]
    return out


def publish(data, p, cfg, tok, meta, max_q, res, kind, n_topics, latest):
    version = (latest["version"] if latest else 0) + 1
    m = {k: v for k, v in meta.items() if k not in
         ("format", "config", "vocab", "max_q", "order", "shapes", "weights", "tokenizer", "merges")}
    m.update({"model_version": version, "test_intent_accuracy": round(res["base"], 4),
              "new_topics_accuracy": None if res["new"] is None else round(res["new"], 4),
              "learned_topics": n_topics, "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"), "train_kind": kind})
    # ответы новых тем — в responses модели (приложение по ним понимает, что ответ «уверенный»)
    resp = dict(m.get("responses") or {})
    for t in knowledge.load(data):
        resp[t["tag"]] = {k: v for k, v in t["responses"].items() if v}
    m["responses"] = resp
    out = export_model(p, cfg, tok, m, tokenizer_type="bpe", max_q=max_q, merges=tok.merges)
    fname = f"model_v{version}.json"
    (data / "models" / fname).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    store.write_json(data / "models/latest.json", {"version": version, "file": fname, "kind": kind,
                                                   "test_intent_acc": m["test_intent_accuracy"],
                                                   "new_topics_acc": m["new_topics_accuracy"], "at": m["trained_at"]})
    return version


def record_progress(data, res, kind, version, n_topics, n_params_):
    prog = store.read_json(data / "progress.json", [])
    prog.append({"at": time.strftime("%Y-%m-%d %H:%M"), "kind": kind, "version": version,
                 "base": round(res["base"], 4), "math": None if res.get("math") is None else round(res["math"], 4), "new": None if res["new"] is None else round(res["new"], 4),
                 "new_hits": res["new_hits"], "new_total": res["new_total"], "topics": n_topics,
                 "params": n_params_})
    store.write_json(data / "progress.json", prog)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "server_data"))
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--full", action="store_true", help="обучение с нуля на всём знании (раз в неделю)")
    ap.add_argument("--max-drop", type=float, default=0.01)
    args = ap.parse_args()
    data = Path(args.data)
    (data / "models").mkdir(parents=True, exist_ok=True)

    topics = knowledge.load(data)
    trained = set(store.read_json(data / "trained_topics.json", []))
    sig = lambda t: f"{t['tag']}:{len(t['train'])}:{len(t['heldout'])}:{json.dumps(t['responses'], ensure_ascii=False, sort_keys=True)}"
    fresh = [t for t in topics if sig(t) not in trained]
    weak = mathcheck.weak_kinds(data)
    drills = {k: mathcheck.drill(k) for k in weak}
    weak_test = [(q, a) for k in weak for q, a in drills[k][1]]
    if not args.full and not fresh and not weak:
        log(data, "новых тем нет — обучение не нужно", quiet=True)
        return
    if args.full and not topics:
        log(data, "знаний ещё нет — полное переобучение не нужно", quiet=True)
        return

    latest = store.read_json(data / "models/latest.json", None)
    cur_path = data / "models" / latest["file"] if latest else BASE_MODEL
    p, cfg, tok, meta = load_model(str(cur_path))
    max_q = meta.get("max_q", 24)

    intents = json.loads((ROOT / "data/intents.json").read_text(encoding="utf-8"))["intents"]
    base_resp = {it["tag"]: answer_set(it["responses"]) for it in intents}
    train_pairs, test_pairs = build_pairs(intents, random.Random(0), split_seed=0)
    base_tests = sorted({(q, tag) for q, a, tag, lang in test_pairs})

    def weak_acc(p_, cfg_, tok_, mq_):
        if not weak_test:
            return None
        kv_ = KVModel(p_, cfg_)
        return sum(gen(kv_, tok_, cfg_, mq_, q) == a for q, a in weak_test) / len(weak_test)

    before = exam(p, cfg, tok, max_q, base_tests, topics, base_resp)
    before["weak"] = weak_acc(p, cfg, tok, max_q)
    if weak:
        log(data, f"слабые типы примеров (по твоим ошибкам): {', '.join(weak)}; на спрятанных ДО: {before['weak']*100:.0f}%")
    log(data, f"{'ПОЛНОЕ ПЕРЕОБУЧЕНИЕ' if args.full else 'дообучение'}: тем всего {len(topics)}, новых/изменённых {len(fresh)}. "
              f"ДО: старые {before['base']*100:.1f}%, новые темы на невиданных формулировках "
              f"{before['new_hits']}/{before['new_total']}")

    if args.full:
        # с нуля: intents.json + все темы; модель растёт с объёмом знаний
        merged = data / "intents_merged.json"
        store.write_json(merged, {"intents": intents + knowledge.as_intents(topics)})
        math_file = MATH
        if weak and MATH.exists():                 # + примеры на слабые типы
            md = json.loads(MATH.read_text(encoding="utf-8"))
            for k in weak:
                md["train"] += [{"q": q, "a": a, "kind": k.split("_")[0], "lang": "ru"} for q, a in drills[k][0]]
            math_file = data / "math_merged.json"
            math_file.write_text(json.dumps(md, ensure_ascii=False), encoding="utf-8")
        d = 96 if len(topics) < 40 else 128
        layers = 3 if len(topics) < 40 else 4
        out = data / "models/_candidate_full.json"
        cmd = [sys.executable, str(ROOT / "train_transformer.py"), "--tokenizer", "bpe", "--vocab", "1000",
               *(["--math", str(math_file)] if MATH.exists() else []),
               "--d", str(d), "--layers", str(layers), "--heads", "4", "--steps", "6000", "--warmup", "300",
               "--drop", "0.05", "--intents", str(merged), "--out", str(out), "--no-app",
               "--ckpt_dir", str(data / "ckpt_full"), "--ckpt_every", "100000",
               "--log", str(data / "logs/train_full.txt")]
        t0 = time.time()
        r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        if r.returncode != 0:
            log(data, f"ОШИБКА полного обучения: {r.stderr[-500:]}")
            return
        p2, cfg2, tok2, meta2 = load_model(str(out))
        max_q2 = meta2.get("max_q", 24)
        log(data, f"обучено с нуля за {(time.time()-t0)/60:.0f} мин: d={d}, слоёв={layers}, {n_params(p2):,} параметров")
        after = exam(p2, cfg2, tok2, max_q2, base_tests, topics, base_resp)
        after["weak"] = weak_acc(p2, cfg2, tok2, max_q2)
        cur_score = before["base"] + (before["new"] or 0) + (before["math"] or 0)
        new_score = after["base"] + (after["new"] or 0) + (after["math"] or 0)
        passed = after["base"] >= before["base"] - args.max_drop and new_score > cur_score and \
            (after["math"] or 0) >= (before["math"] or 0) - 0.02
        p, cfg, tok, meta, max_q = p2, cfg2, tok2, meta2, max_q2
    else:
        new_ex = topic_pairs(fresh) + [(q, a) for k in weak for q, a in drills[k][0]]
        replay = [(q, a) for q, a, *_ in train_pairs] + topic_pairs([t for t in topics if t not in fresh])
        replay += random.Random(3).sample(math_sets()[0], min(4000, len(math_sets()[0])))   # не разучиться считать
        opt = AdamW(p, lr=args.lr, wd=0.01)
        rng = np.random.default_rng(int(time.time()))
        t0 = time.time()
        for step in range(args.steps):
            batch = [new_ex[i] for i in rng.integers(0, len(new_ex), 12)] + \
                    [replay[i] for i in rng.integers(0, len(replay), 20)]
            idx, targets, mask = make_batch(tok, batch, max_q, cfg["block"], rng)
            loss, grads = loss_and_grads(p, cfg, idx, targets, mask, drop=0.05, rng=rng)
            clip_grads(grads, 1.0)
            opt.step(p, grads, lr_schedule(step, args.steps, args.lr, max(10, args.steps // 10)))
        log(data, f"дообучение: {args.steps} шагов за {(time.time()-t0)/60:.1f} мин")
        after = exam(p, cfg, tok, max_q, base_tests, topics, base_resp)
        after["weak"] = weak_acc(p, cfg, tok, max_q)
        improved = (fresh and after["new_hits"] > 0) or (weak and after["weak"] > before["weak"])
        passed = after["base"] >= before["base"] - args.max_drop and improved and \
            (after["new"] or 0) >= (before["new"] or 0) and \
            (after["math"] or 0) >= (before["math"] or 0) - 0.02

    if weak:
        log(data, f"слабые типы на спрятанных ПОСЛЕ: {after['weak']*100:.0f}% (было {before['weak']*100:.0f}%)")
    if after.get("math") is not None:
        log(data, f"счёт на невиданных примерах: {after['math']*100:.1f}% (было {(before['math'] or 0)*100:.1f}%)")
    log(data, f"ЭКЗАМЕН ПОСЛЕ: старые {after['base']*100:.1f}% (было {before['base']*100:.1f}%), "
              f"новые темы на невиданных формулировках {after['new_hits']}/{after['new_total']} "
              f"(было {before['new_hits']}/{before['new_total']}) → "
              + ("СДАЛА, публикую" if passed else "НЕ СДАЛА, остаётся прежняя модель"))
    for d_ in after["details"]:
        if not d_["ok"]:
            log(data, f"   ✗ «{d_['q']}» → «{d_['a'][:70]}»", quiet=True)
    if not passed:
        record_progress(data, before, "rejected", latest["version"] if latest else 0, len(topics), n_params(p))
        return
    version = publish(data, p, cfg, tok, meta, max_q, after, "full" if args.full else "finetune", len(topics), latest)
    record_progress(data, after, "full" if args.full else "finetune", version, len(topics), n_params(p))
    store.write_json(data / "trained_topics.json", sorted({sig(t) for t in topics}))
    if weak:                                     # статистику слабых типов начинаем заново — проверим на новых примерах
        stats = store.read_json(data / "math_stats.json", {})
        for k in weak:
            stats[k] = {"ok": 0, "wrong": 0, "examples": [], "drilled": stats.get(k, {}).get("drilled", 0) + 1}
        store.write_json(data / "math_stats.json", stats)
    log(data, f"опубликована версия {version}: приложение скачает её при следующем запуске")


if __name__ == "__main__":
    main()
