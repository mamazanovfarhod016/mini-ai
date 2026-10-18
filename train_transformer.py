"""
Обучение Mini AI v0.2 — трансформера (mini_ai/transformer.py).

    .venv/bin/python train_transformer.py

Идея очень простая: мы учим модель ПРОДОЛЖАТЬ строку вида

    <bos> как дила ? <sep> Норм, а у тебя как? <eos>

Модель видит вопрос и спецтокен <sep>, и дальше, буква за буквой, учится писать
правильный ответ (loss считаем только по буквам ОТВЕТА — буквы вопроса модель
просто «читает», угадывать их не нужно). Пары вопрос→ответ берутся из
data/intents.json: КАЖДАЯ фраза-пример намерения комбинируется с КАЖДЫМ готовым
ответом этого намерения (на том же языке) — так из 1 574 фраз получается
несколько тысяч обучающих пар, а модель видит один и тот же вопрос с разными
(но всегда правильными) ответами.

Разбиваем по ВОПРОСАМ (а не по парам): 20% вопросов каждого намерения уходят
в тест и во время обучения модель их вообще не видит — ни с одним ответом.
Так точность на тесте — это честная проверка «обобщения», а не запоминания.
"""
import argparse
import base64
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

from mini_ai.transformer import (
    BOS, EOS, PAD, SEP, CharTokenizer, KVModel, clean_question, clip_grads,
    export_model, forward, init_params, lr_schedule, n_params, softmax,
    AdamW, loss_and_grads, TIME_TOK, DATE_TOK,
)

ROOT = Path(__file__).resolve().parent
RU_KEYS = "йцукенгшщзхфывапролджэячсмитьбю"
LAT_KEYS = "qwertyuiopasdfghjklzxcvbnm"


# ======================================================================================
# 1. Опечатки в ВОПРОСАХ (как в train.py для MLP) — ответ всегда остаётся «эталонным»
# ======================================================================================
def typo(phrase, rng):
    s = list(phrase)
    letters = [i for i, c in enumerate(s) if c.isalpha()]
    if len(letters) < 3:
        return phrase
    i = rng.choice(letters)
    kind = rng.randrange(4)
    alphabet = RU_KEYS if "а" <= s[i] <= "я" else LAT_KEYS
    if kind == 0:
        del s[i]
    elif kind == 1:
        s.insert(i, s[i])
    elif kind == 2:
        s[i] = rng.choice(alphabet)
    elif i + 1 < len(s) and s[i + 1].isalpha():
        s[i], s[i + 1] = s[i + 1], s[i]
    return "".join(s)


def is_latin(text):
    lat = sum(1 for c in text if "a" <= c <= "z")
    cyr = sum(1 for c in text if "а" <= c <= "я")
    return lat > cyr


# ======================================================================================
# 2. Строим пары (вопрос, ответ) из data/intents.json
# ======================================================================================
def build_pairs(intents, rng, split_seed, test_frac=0.2):
    """Делим ВОПРОСЫ (не пары!) на train/test по каждому намерению, затем
    каждому вопросу сопоставляем все подходящие по языку ответы его намерения."""
    train_pairs, test_pairs = [], []
    for it in intents:
        patterns = sorted(set(clean_question(p) for p in it["patterns"]))
        srng = random.Random(split_seed * 1000 + hash(it["tag"]) % 97)
        srng.shuffle(patterns)
        n_test = max(1, round(len(patterns) * test_frac))
        test_q, train_q = patterns[:n_test], patterns[n_test:]
        for q_list, bucket in ((train_q, train_pairs), (test_q, test_pairs)):
            for q in q_list:
                lang = "uz" if is_latin(q) else "ru"
                answers = it["responses"].get(lang) or it["responses"].get("ru")
                for a in answers:
                    bucket.append((q, a, it["tag"], lang))
    return train_pairs, test_pairs


def augment_pairs(pairs, rng, n_typo=2):
    """К train-парам добавляем варианты вопроса с опечатками (ответ не меняется).
    На ответах опечаток НЕТ — модель должна научиться писать правильно."""
    out = list(pairs)
    for q, a, tag, lang in pairs:
        for _ in range(n_typo):
            out.append((typo(q, rng), a, tag, lang))
    return out


def fill_placeholders(answer, rng=None):
    """{time}/{date} заменяем на ОДИН фиксированный символ-сентинел (TIME_TOK/DATE_TOK).

    ДИАГНОСТИКА (18.10): раньше сюда подставлялись случайные конкретные значения
    ("08:15", "17.10.2026", ...) — модели приходилось угадывать произвольные цифры,
    и на инференсе она их выдумывала («сегодня 0.0.2026»). Теперь модель учится
    писать один и тот же символ-плейсхолдер — задача детерминирована и простая,
    а настоящие дату/время подставляет приложение ПОСЛЕ генерации
    (fill_placeholders_runtime в Python, fillPlaceholders в app.js), по этим же
    символам TIME_TOK/DATE_TOK."""
    answer = answer.replace("{time}", TIME_TOK)
    answer = answer.replace("{date}", DATE_TOK)
    return answer


# ======================================================================================
# 3. Батчи: <bos> вопрос <sep> ответ <eos>, паддинг, маска loss только на ответе
# ======================================================================================
def encode_example(tok, q, a, max_q, block):
    q_ids = tok.encode(clean_question(q))[:max_q]
    a_ids = tok.encode(a)
    seq = [BOS] + q_ids + [SEP] + a_ids + [EOS]
    seq = seq[:block]
    ans_start = 1 + len(q_ids) + 1   # позиция первого символа ответа
    return seq, ans_start


def make_batch(tok, examples, max_q, block, rng):
    """examples: список (q, a). Возвращает idx, targets, mask (loss только на ответе)."""
    encoded = [encode_example(tok, q, fill_placeholders(a, rng), max_q, block) for q, a in examples]
    T = max(len(s) for s, _ in encoded)
    B = len(encoded)
    idx = np.full((B, T), PAD, dtype=np.int64)
    targets = np.full((B, T), PAD, dtype=np.int64)
    mask = np.zeros((B, T), dtype=bool)
    for i, (seq, ans_start) in enumerate(encoded):
        L = len(seq)
        idx[i, :L] = seq
        # targets[t] = символ, который должен идти ПОСЛЕ позиции t, т.е. seq[t+1]
        targets[i, :L - 1] = seq[1:]
        # loss считаем в позициях t, где seq[t+1] — буква ОТВЕТА (включая финальный <eos>)
        for t in range(ans_start - 1, L - 1):
            mask[i, t] = True
    return idx, targets, mask


# ======================================================================================
# 4. Цикл обучения
# ======================================================================================
def evaluate(model_p, cfg, tok, pairs, max_q, block, batch=48):
    """Loss/perplexity на наборе пар (без дропаута, без аугментации)."""
    rng = np.random.default_rng(123)
    total_loss, total_tok = 0.0, 0
    for s in range(0, len(pairs), batch):
        chunk = pairs[s:s + batch]
        idx, targets, mask = make_batch(tok, [(q, a) for q, a, *_ in chunk], max_q, block, rng)
        logits, _ = forward(model_p, cfg, idx, drop=0.0, rng=None)
        V = logits.shape[-1]
        P = softmax(logits.reshape(-1, V).astype(np.float64), -1)
        t = targets.reshape(-1)
        m = mask.reshape(-1)
        logp = np.log(P[np.arange(len(t)), t] + 1e-30)
        total_loss += float(-(logp * m).sum())
        total_tok += int(m.sum())
    avg = total_loss / max(1, total_tok)
    return avg, math.exp(min(avg, 30))


def greedy_answer(kv, tok, question, block):
    ids = tok.prompt(question)
    out, _ = kv.generate(ids, max_new=block - len(ids), temperature=0.0)
    return tok.decode(out)


def answer_matches(generated, intent_tag, intents_by_tag, lang):
    """Точная метрика: дословное совпадение с одним из эталонных ответов намерения.
    {time}/{date} в эталоне заменяем на тот же сентинел TIME_TOK/DATE_TOK, который
    видела модель при обучении (fill_placeholders) — сравнение становится точным,
    без всяких «префиксов» (раньше плейсхолдер был случайным числом, теперь он
    фиксированный символ, который модель обязана воспроизвести буквально)."""
    gold = intents_by_tag[intent_tag]["responses"].get(lang) or intents_by_tag[intent_tag]["responses"].get("ru")
    gen = generated.strip()
    for g in gold:
        g_filled = g.replace("{time}", TIME_TOK).replace("{date}", DATE_TOK).strip()
        if gen == g_filled:
            return True
    return False


def softest_distance(generated, intent_tag, intents_by_tag, lang):
    """Мягкая метрика: нормированное расстояние Левенштейна до БЛИЖАЙШЕГО эталонного
    ответа правильного намерения (0 = идеально совпал, 1 = совсем не похож)."""
    gold = intents_by_tag[intent_tag]["responses"].get(lang) or intents_by_tag[intent_tag]["responses"].get("ru")
    gold = [g.replace("{time}", TIME_TOK).replace("{date}", DATE_TOK) for g in gold]
    best = 1.0
    for g in gold:
        best = min(best, levenshtein(generated, g) / max(1, max(len(generated), len(g))))
    return best


def intent_of_answer(gen, intents_by_tag):
    """Какому намерению соответствует сгенерированный текст (точное совпадение
    с любым эталонным ответом любого намерения, любого языка). Используется для
    метрики «процент правильных НАМЕРЕНИЙ» — более честной, чем просто exact-match,
    т.к. показывает, понимает ли модель СМЫСЛ вопроса (а не просто зазубрила фразу)."""
    gen_s = gen.strip()
    for tag, it in intents_by_tag.items():
        for lang, answers in it["responses"].items():
            for g in answers:
                g_filled = g.replace("{time}", TIME_TOK).replace("{date}", DATE_TOK).strip()
                if gen_s == g_filled:
                    return tag
    return None


def intent_accuracy(kv, tok, intents_by_tag, qs, block):
    """qs: список (q, true_tag, lang). Возвращает (доля верных намерений, детали)."""
    hits, details = 0, []
    for q, tag, lang in qs:
        gen = greedy_answer(kv, tok, q, block)
        pred = intent_of_answer(gen, intents_by_tag)
        ok = pred == tag
        hits += ok
        details.append({"q": q, "intent": tag, "lang": lang, "answer": gen, "pred": pred, "ok": ok})
    return hits / max(1, len(qs)), details


def levenshtein(a, b):
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[lb]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--block", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2600)
    ap.add_argument("--lr", type=float, default=2.5e-3)
    ap.add_argument("--warmup", type=int, default=150)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--max_q", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "model_transformer.json"))
    ap.add_argument("--quick", action="store_true", help="мало шагов — для smoke-теста")
    ap.add_argument("--balanced", action="store_true", default=True,
                     help="сэмплировать батчи СБАЛАНСИРОВАННО по намерениям (по умолчанию включено — "
                          "диагностика 18.10 показала, что случайная выборка по парам даёт разным "
                          "намерениям разный вес, из-за чего редкие намерения недообучены)")
    ap.add_argument("--no-balanced", dest="balanced", action="store_false")
    ap.add_argument("--ckpt_dir", default="/tmp/mini-ai-ckpt",
                     help="папка для чекпойнтов (на случай обрыва длинного обучения)")
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--eval_every", type=int, default=1000,
                     help="как часто (в шагах) считать intent-accuracy на train/test для подбора лучшего чекпойнта")
    args = ap.parse_args()
    if args.quick:
        args.steps = 40

    t_start = time.time()
    data = json.loads((ROOT / "data/intents.json").read_text(encoding="utf-8"))
    intents = data["intents"]
    intents_by_tag = {it["tag"]: it for it in intents}
    rng_py = random.Random(args.seed)

    train_pairs, test_pairs = build_pairs(intents, rng_py, split_seed=args.seed)
    print(f"Вопросов всего: train {len({q for q,*_ in train_pairs})}, "
          f"test {len({q for q,*_ in test_pairs})} (test — вопросы, НЕ виденные моделью)")
    print(f"Пар (вопрос→ответ): train {len(train_pairs)}, test {len(test_pairs)}")

    train_aug = augment_pairs(train_pairs, rng_py, n_typo=2)
    print(f"Train-пар с опечатками: {len(train_aug)}")

    # ---------- словарь символов ----------
    all_text = []
    for q, a, *_ in train_aug + test_pairs:
        all_text.append(clean_question(q))
        all_text.append(a)
    chars = set("".join(all_text))
    tok = CharTokenizer(chars)
    V = len(tok)
    print(f"Словарь: {V} символов")

    p, cfg = init_params(V, d=args.d, n_layer=args.layers, n_head=args.heads,
                          block=args.block, seed=args.seed)
    print(f"Архитектура: d={args.d}, слоёв={args.layers}, голов={args.heads}, "
          f"контекст={args.block}, d_ff={cfg['d_ff']}")
    print(f"Параметров: {n_params(p):,}")

    opt = AdamW(p, lr=args.lr, wd=0.01)
    np_rng = np.random.default_rng(args.seed)

    examples = [(q, a) for q, a, *_ in train_aug]
    n = len(examples)

    # ---------- сбалансированное сэмплирование по намерениям ----------
    # Диагностика (18.10): случайная выборка по ПАРАМ даёт намерениям с бОльшим
    # числом готовых ответов (responses) больше представленности в батчах, чем
    # намерениям с малым числом ответов — хотя по СМЫСЛУ все намерения равноважны.
    # Сбалансированная схема: на каждом шаге сначала равновероятно выбираем
    # намерение (28 штук), затем равновероятно — один из его train-примеров.
    by_tag = {}
    for i, (q, a, tag, lang) in enumerate(train_aug):
        by_tag.setdefault(tag, []).append(i)
    tags_list = sorted(by_tag.keys())
    tag_arrays = {t: np.array(idxs) for t, idxs in by_tag.items()}
    print(f"Сбалансированное сэмплирование: {'ВКЛ' if args.balanced else 'выкл'} "
          f"({len(tags_list)} намерений)")

    def sample_batch_idxs(batch_size):
        if not args.balanced:
            return np_rng.integers(0, n, size=batch_size)
        chosen_tags = np_rng.choice(tags_list, size=batch_size, replace=True)
        out = np.empty(batch_size, dtype=np.int64)
        for i, t in enumerate(chosen_tags):
            arr = tag_arrays[t]
            out[i] = arr[np_rng.integers(0, len(arr))]
        return out

    # ---------- уникальные train/test вопросы для intent-accuracy во время обучения ----------
    train_qs_eval = sorted({(q, tag, lang) for q, a, tag, lang in train_pairs})
    test_qs_eval = sorted({(q, tag, lang) for q, a, tag, lang in test_pairs})

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_test_acc = -1.0
    best_step = -1
    progress_log = []

    log_every = max(1, args.steps // 20)
    for step in range(args.steps):
        idxs = sample_batch_idxs(args.batch)
        batch_examples = [examples[i] for i in idxs]
        idx, targets, mask = make_batch(tok, batch_examples, args.max_q, args.block, np_rng)
        lr = lr_schedule(step, args.steps, args.lr, args.warmup)
        loss, grads = loss_and_grads(p, cfg, idx, targets, mask, drop=args.drop, rng=np_rng)
        gnorm = clip_grads(grads, max_norm=1.0)
        opt.step(p, grads, lr)
        if step % log_every == 0 or step == args.steps - 1:
            elapsed = time.time() - t_start
            print(f"  шаг {step:5d}/{args.steps}  loss={loss:.3f}  lr={lr:.5f}  "
                  f"||grad||={gnorm:.2f}  {elapsed:.0f}с", flush=True)

        is_ckpt_step = (step > 0 and step % args.ckpt_every == 0) or step == args.steps - 1
        if is_ckpt_step:
            kv_ckpt = KVModel(p, cfg)
            train_acc, _ = intent_accuracy(kv_ckpt, tok, intents_by_tag, train_qs_eval, args.block)
            test_acc, _ = intent_accuracy(kv_ckpt, tok, intents_by_tag, test_qs_eval, args.block)
            elapsed = time.time() - t_start
            msg = (f"[ckpt шаг {step}] train_intent_acc={train_acc*100:.1f}% "
                   f"test_intent_acc={test_acc*100:.1f}% elapsed={elapsed/60:.1f}мин")
            print(msg, flush=True)
            progress_log.append({"step": step, "train_intent_acc": train_acc, "test_intent_acc": test_acc,
                                  "elapsed_sec": elapsed})
            meta_ckpt = {"step": step, "train_intent_acc": round(train_acc, 4),
                         "test_intent_acc": round(test_acc, 4), "elapsed_sec": round(elapsed, 1)}
            out_dict_ckpt = export_model(p, cfg, tok, meta_ckpt)
            (ckpt_dir / f"ckpt_step{step}.json").write_text(
                json.dumps(out_dict_ckpt, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
            # лучший по test — отдельный файл "best.json" (не затираем все промежуточные,
            # чтобы прогресс не терялся при обрыве; но держим явный указатель на лучший)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_step = step
                (ckpt_dir / "best.json").write_text(
                    json.dumps(out_dict_ckpt, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
                print(f"  -> новый лучший чекпойнт (test_intent_acc={test_acc*100:.1f}%)", flush=True)
            (ckpt_dir / "progress.json").write_text(
                json.dumps({"progress": progress_log, "best_step": best_step, "best_test_acc": best_test_acc},
                           ensure_ascii=False, indent=1), encoding="utf-8")

    train_time = time.time() - t_start
    print(f"\nОбучение заняло {train_time/60:.1f} мин")
    print(f"Лучший чекпойнт по test_intent_acc: шаг {best_step}, {best_test_acc*100:.1f}%")

    # ---------- метрики train/test (loss/perplexity) ----------
    train_loss, train_ppl = evaluate(p, cfg, tok, train_pairs, args.max_q, args.block)
    test_loss, test_ppl = evaluate(p, cfg, tok, test_pairs, args.max_q, args.block)
    print(f"Train: loss={train_loss:.3f}  perplexity={train_ppl:.2f}")
    print(f"Test:  loss={test_loss:.3f}  perplexity={test_ppl:.2f}")

    # ---------- точность на train/test (greedy): намерение + дословный ответ ----------
    kv = KVModel(p, cfg)
    train_intent_acc, train_intent_details = intent_accuracy(kv, tok, intents_by_tag, train_qs_eval, args.block)
    test_qs = test_qs_eval
    exact_hits, soft_dists, examples_log = 0, [], []
    for q, tag, lang in test_qs:
        gen = greedy_answer(kv, tok, q, args.block)
        ok = answer_matches(gen, tag, intents_by_tag, lang)
        exact_hits += ok
        soft_dists.append(softest_distance(gen, tag, intents_by_tag, lang))
        examples_log.append({"q": q, "intent": tag, "lang": lang, "answer": gen, "exact": ok})
    exact_acc = exact_hits / max(1, len(test_qs))
    soft_score = 1 - (sum(soft_dists) / max(1, len(soft_dists)))
    test_intent_acc, _ = intent_accuracy(kv, tok, intents_by_tag, test_qs_eval, args.block)
    print(f"\nTRAIN — доля правильных НАМЕРЕНИЙ (greedy): {train_intent_acc*100:.1f}% "
          f"({sum(d['ok'] for d in train_intent_details)}/{len(train_qs_eval)})")
    print(f"TEST  — доля правильных НАМЕРЕНИЙ (greedy): {test_intent_acc*100:.1f}% "
          f"({int(round(test_intent_acc*len(test_qs_eval)))}/{len(test_qs_eval)})")
    print(f"Точная метрика (greedy-ответ дословно совпал с эталоном намерения): "
          f"{exact_acc*100:.1f}% ({exact_hits}/{len(test_qs)})")
    print(f"Мягкая метрика (1 − норм. расстояние Левенштейна до ближайшего эталона): "
          f"{soft_score*100:.1f}%")

    # ---------- сохранение весов (float16 base64, < 5 МБ) ----------
    meta = {
        "train_loss": round(train_loss, 4), "train_ppl": round(train_ppl, 3),
        "test_loss": round(test_loss, 4), "test_ppl": round(test_ppl, 3),
        "test_exact_accuracy": round(exact_acc, 4),
        "test_soft_score": round(soft_score, 4),
        "train_intent_accuracy": round(train_intent_acc, 4),
        "test_intent_accuracy": round(test_intent_acc, 4),
        "n_params": n_params(p),
        "train_time_sec": round(train_time, 1),
        "train_pairs": len(train_pairs), "train_pairs_aug": len(train_aug),
        "test_pairs": len(test_pairs), "test_questions": len(test_qs),
        "balanced_sampling": args.balanced,
        "best_ckpt_step": best_step, "best_ckpt_test_intent_acc": round(best_test_acc, 4),
        "responses": {it["tag"]: it["responses"] for it in intents},
        "labels": {it["tag"]: it["description"] for it in intents},
    }
    out_dict = export_model(p, cfg, tok, meta)
    text = json.dumps(out_dict, ensure_ascii=False, separators=(",", ":"))
    size_kb = len(text.encode()) / 1024
    print(f"\nmodel_transformer.json: {size_kb:.0f} КБ ({size_kb/1024:.2f} МБ)")
    Path(args.out).write_text(text, encoding="utf-8")
    (ROOT / "app/www/model_transformer.js").write_text(
        "window.MINI_AI_TRANSFORMER=" + text + ";\n", encoding="utf-8")

    # ---------- примеры генерации (для README/лога) ----------
    print("\n=== Примеры генерации ===")
    demo_qs = [q for q, tag, lang in test_qs[:6]]
    extra = ["превет как дила", "qalaysan do'stim", "кто тбея создал", "расскажи анекдот про программиста",
             "столица франции", "xyzqwrty bla bla", "который час сейчас покажи", "менга hayrli kun"]
    for q in demo_qs + extra:
        gen = greedy_answer(kv, tok, q, args.block)
        print(f"  «{q}» → «{gen}»")
        examples_log.append({"q": q, "answer": gen, "demo": True})

    # ---------- лог в файл ----------
    log_path = ROOT / "docs/train_log_transformer.txt"
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"Mini AI v0.2 — лог обучения трансформера\n")
        f.write(f"Архитектура: d={args.d}, layers={args.layers}, heads={args.heads}, "
                f"block={args.block}, params={n_params(p):,}\n")
        f.write(f"Время обучения: {train_time/60:.1f} мин ({args.steps} шагов, batch={args.batch}, "
                f"сбалансированное сэмплирование={'да' if args.balanced else 'нет'})\n")
        f.write(f"Лучший чекпойнт по test_intent_acc: шаг {best_step}, {best_test_acc*100:.1f}%\n")
        f.write(f"Train: loss={train_loss:.3f} ppl={train_ppl:.2f}\n")
        f.write(f"Test:  loss={test_loss:.3f} ppl={test_ppl:.2f}\n")
        f.write(f"TRAIN — доля правильных НАМЕРЕНИЙ (greedy): {train_intent_acc*100:.1f}% "
                f"({sum(d['ok'] for d in train_intent_details)}/{len(train_qs_eval)})\n")
        f.write(f"TEST  — доля правильных НАМЕРЕНИЙ (greedy): {test_intent_acc*100:.1f}% "
                f"({int(round(test_intent_acc*len(test_qs_eval)))}/{len(test_qs_eval)})\n")
        f.write(f"Точная метрика (exact match, дословно): {exact_acc*100:.1f}% ({exact_hits}/{len(test_qs)})\n")
        f.write(f"Мягкая метрика (1-Levenshtein): {soft_score*100:.1f}%\n")
        f.write(f"Размер model_transformer.json: {size_kb:.0f} КБ\n\n")
        f.write("Примеры (вопрос → ответ, невиденные тестовые вопросы):\n")
        for e in examples_log:
            tag = f" [{e.get('intent')}]" if e.get("intent") else ""
            ok = " ✓" if e.get("exact") else (" " if "exact" in e else "")
            f.write(f"  «{e['q']}»{tag} → «{e['answer']}»{ok}\n")
    print(f"\nЛог сохранён: {log_path}")
    return exact_acc


if __name__ == "__main__":
    acc = main()
    sys.exit(0)
