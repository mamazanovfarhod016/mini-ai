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
    AdamW, loss_and_grads,
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


PLACEHOLDER_TIME = ["08:15", "12:40", "19:05", "23:50", "06:30", "14:22"]
PLACEHOLDER_DATE = ["17.10.2026", "01.01.2027", "23.05.2026", "09.09.2026"]


def fill_placeholders(answer, rng):
    """{time}/{date} заменяем на случайные конкретные значения — модель учится
    копировать ЧИСЛА из спецтокена-плейсхолдера, а не запоминать одно число.
    На инференсе JS/Python сами вставляют час/дату с устройства — модель здесь
    просто видит текст с уже подставленным (случайным на обучении) значением."""
    answer = answer.replace("{time}", rng.choice(PLACEHOLDER_TIME))
    answer = answer.replace("{date}", rng.choice(PLACEHOLDER_DATE))
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
    """Точная метрика: дословное совпадение с одним из эталонных ответов намерения."""
    gold = intents_by_tag[intent_tag]["responses"].get(lang) or intents_by_tag[intent_tag]["responses"].get("ru")
    gold_clean = [g.replace("{time}", "").replace("{date}", "").strip() for g in gold]
    # т.к. плейсхолдеры заменены случайными значениями при обучении, сравниваем
    # префикс/суффикс без динамической части
    gen = generated.strip()
    for g in gold:
        g_pref = g.split("{")[0].strip()
        if gen == g.strip():
            return True
        if g_pref and gen.startswith(g_pref) and ("{time}" in g or "{date}" in g):
            return True
    return False


def softest_distance(generated, intent_tag, intents_by_tag, lang):
    """Мягкая метрика: нормированное расстояние Левенштейна до БЛИЖАЙШЕГО эталонного
    ответа правильного намерения (0 = идеально совпал, 1 = совсем не похож)."""
    gold = intents_by_tag[intent_tag]["responses"].get(lang) or intents_by_tag[intent_tag]["responses"].get("ru")
    best = 1.0
    for g in gold:
        best = min(best, levenshtein(generated, g) / max(1, max(len(generated), len(g))))
    return best


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
    log_every = max(1, args.steps // 20)
    for step in range(args.steps):
        idxs = np_rng.integers(0, n, size=args.batch)
        batch_examples = [examples[i] for i in idxs]
        idx, targets, mask = make_batch(tok, batch_examples, args.max_q, args.block, np_rng)
        lr = lr_schedule(step, args.steps, args.lr, args.warmup)
        loss, grads = loss_and_grads(p, cfg, idx, targets, mask, drop=args.drop, rng=np_rng)
        gnorm = clip_grads(grads, max_norm=1.0)
        opt.step(p, grads, lr)
        if step % log_every == 0 or step == args.steps - 1:
            elapsed = time.time() - t_start
            print(f"  шаг {step:5d}/{args.steps}  loss={loss:.3f}  lr={lr:.5f}  "
                  f"||grad||={gnorm:.2f}  {elapsed:.0f}с")

    train_time = time.time() - t_start
    print(f"\nОбучение заняло {train_time/60:.1f} мин")

    # ---------- метрики train/test ----------
    train_loss, train_ppl = evaluate(p, cfg, tok, train_pairs, args.max_q, args.block)
    test_loss, test_ppl = evaluate(p, cfg, tok, test_pairs, args.max_q, args.block)
    print(f"Train: loss={train_loss:.3f}  perplexity={train_ppl:.2f}")
    print(f"Test:  loss={test_loss:.3f}  perplexity={test_ppl:.2f}")

    # ---------- точность на НЕВИДЕННЫХ тестовых вопросах (greedy) ----------
    kv = KVModel(p, cfg)
    test_qs = sorted({(q, tag, lang) for q, a, tag, lang in test_pairs})
    exact_hits, soft_dists, examples_log = 0, [], []
    for q, tag, lang in test_qs:
        gen = greedy_answer(kv, tok, q, args.block)
        ok = answer_matches(gen, tag, intents_by_tag, lang)
        exact_hits += ok
        soft_dists.append(softest_distance(gen, tag, intents_by_tag, lang))
        examples_log.append({"q": q, "intent": tag, "lang": lang, "answer": gen, "exact": ok})
    exact_acc = exact_hits / max(1, len(test_qs))
    soft_score = 1 - (sum(soft_dists) / max(1, len(soft_dists)))
    print(f"\nТочная метрика (greedy-ответ дословно совпал с эталоном намерения): "
          f"{exact_acc*100:.1f}% ({exact_hits}/{len(test_qs)})")
    print(f"Мягкая метрика (1 − норм. расстояние Левенштейна до ближайшего эталона): "
          f"{soft_score*100:.1f}%")

    # ---------- сохранение весов (float16 base64, < 5 МБ) ----------
    meta = {
        "train_loss": round(train_loss, 4), "train_ppl": round(train_ppl, 3),
        "test_loss": round(test_loss, 4), "test_ppl": round(test_ppl, 3),
        "test_exact_accuracy": round(exact_acc, 4),
        "test_soft_score": round(soft_score, 4),
        "n_params": n_params(p),
        "train_time_sec": round(train_time, 1),
        "train_pairs": len(train_pairs), "train_pairs_aug": len(train_aug),
        "test_pairs": len(test_pairs), "test_questions": len(test_qs),
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
        f.write(f"Время обучения: {train_time/60:.1f} мин ({args.steps} шагов, batch={args.batch})\n")
        f.write(f"Train: loss={train_loss:.3f} ppl={train_ppl:.2f}\n")
        f.write(f"Test:  loss={test_loss:.3f} ppl={test_ppl:.2f}\n")
        f.write(f"Точная метрика (exact match): {exact_acc*100:.1f}% ({exact_hits}/{len(test_qs)})\n")
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
