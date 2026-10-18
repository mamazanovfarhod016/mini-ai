"""
tools/teach.py — дообучение трансформера Mini AI НОВЫМИ парами (вопрос→ответ),
которые Фарход присылает ассистенту в Telegram. Запускается НА СЕРВЕРЕ.

Одна команда:
    .venv/bin/python tools/teach.py --pairs export.json --build

Что делает:
  1. Читает файл с новыми парами (JSON: либо [{"q":..,"a":..}, ...], либо
     экспорт из приложения {"pairs": [...]}). Поддерживает вопрос с несколькими
     формулировками через "variants": ["вопрос 1", "вопрос 2", ...] — каждая
     формулировка становится отдельной парой с тем же ответом.
  2. Объединяет их с data/user_pairs.json (без дублей по normalize(вопрос)).
  3. Считает метрики СТАРОЙ модели (model_transformer.json) на контрольных
     вопросах (CONTROL_Q) + на самих новых парах ("знает ли их ДО обучения").
  4. Дообучает (fine-tune) от текущих весов: часть шагов — новые+старые
     user_pairs, часть — replay исходных train-пар (защита от забывания).
     Это быстрее полного обучения (5-15 мин вместо 55 мин), потому что весов
     не нужно учить с нуля — только «подвинуть».
  5. Считает метрики НОВОЙ модели на тех же наборах — печатает «до/после».
  6. Сохраняет веса в model_transformer.json + app/www/model_transformer.js.
  7. Если --build — поднимает versionCode/versionName в AndroidManifest.xml
     и запускает android/build.sh (пересборка APK).

Без --build можно просто проверить метрики без сборки APK (быстрее для правок).
"""
import argparse
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mini_ai.transformer import (  # noqa: E402
    AdamW, KVModel, clean_question, clip_grads, export_model, load_model,
    loss_and_grads, lr_schedule, n_params,
)
from train_transformer import build_pairs, make_batch, levenshtein  # noqa: E402

MODEL_PATH = ROOT / "model_transformer.json"
USER_PAIRS_PATH = ROOT / "data/user_pairs.json"
MANIFEST_PATH = ROOT / "android/AndroidManifest.xml"

# Контрольные вопросы — те же, что в tests/test_transformer.py (CONTROL_Q),
# по ним проверяем «не испортилось ли старое».
CONTROL_Q = [
    "привет", "как дела", "пока", "кто тебя создал", "спасибо",
    "qalaysan", "salom", "rahmat", "расскажи шутку", "который час",
]


# ======================================================================================
# 1. Загрузка и объединение пар
# ======================================================================================
def normalize_key(q):
    """Ключ для дедупликации: как clean_question, но ещё и без лишних пробелов."""
    return clean_question(q)


def load_pairs_file(path):
    """Принимает: [{"q":"...","a":"..."}], {"pairs":[...]}, вариант с
    "variants": ["формулировка 1", "формулировка 2"] вместо одного "q"."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    items = raw.get("pairs", raw) if isinstance(raw, dict) else raw
    out = []
    for it in items:
        a = it.get("a") or it.get("answer")
        if not a:
            continue
        questions = it.get("variants") or it.get("questions")
        if not questions:
            q = it.get("q") or it.get("question")
            questions = [q] if q else []
        for q in questions:
            q = str(q).strip()
            if q:
                out.append({"q": q, "a": str(a).strip()})
    return out


def merge_pairs(existing, new_pairs):
    """Объединяет без дублей по (normalize(q), a). Если тот же вопрос с НОВЫМ
    ответом — обновляем ответ (пользователь переучил)."""
    by_q = {}
    for p in existing:
        by_q[normalize_key(p["q"])] = p
    added, updated = 0, 0
    for p in new_pairs:
        key = normalize_key(p["q"])
        if key in by_q:
            if by_q[key]["a"] != p["a"]:
                by_q[key] = p
                updated += 1
        else:
            by_q[key] = p
            added += 1
    merged = list(by_q.values())
    return merged, added, updated


# ======================================================================================
# 2. Метрики: контрольные вопросы + новые пары
# ======================================================================================
def greedy(kv, tok, q, block):
    ids = tok.prompt(q)
    out, _ = kv.generate(ids, max_new=block - len(ids), temperature=0.0)
    return tok.decode(out)


def eval_control(p, cfg, tok, block):
    kv = KVModel(p, cfg)
    results = {}
    for q in CONTROL_Q:
        results[q] = greedy(kv, tok, q, block)
    return results


def eval_pairs_exact(p, cfg, tok, block, pairs):
    """Для каждой новой пары: совпал ли greedy-ответ с ожидаемым (точно, после strip)."""
    kv = KVModel(p, cfg)
    hits, details = 0, []
    for ex in pairs:
        gen = greedy(kv, tok, ex["q"], block)
        ok = gen.strip() == ex["a"].strip()
        hits += ok
        details.append({"q": ex["q"], "expected": ex["a"], "got": gen, "ok": ok})
    return hits, details


def compare_control(before, after):
    """Сколько старых контрольных ответов НЕ изменилось (не испортилось)."""
    same = sum(1 for q in CONTROL_Q if before[q] == after[q])
    return same, len(CONTROL_Q)


# ======================================================================================
# 3. Дообучение (fine-tune) с replay
# ======================================================================================
def finetune(p, cfg, tok, user_pairs, replay_pairs, steps, batch, lr, log=print):
    opt = AdamW(p, lr=lr, wd=0.01)
    rng = np.random.default_rng(42)
    user_examples = [(x["q"], x["a"]) for x in user_pairs]
    replay_examples = [(x["q"], x["a"]) for x in replay_pairs] if replay_pairs else user_examples
    t0 = time.time()
    log_every = max(1, steps // 10)
    last_loss = 0.0
    for step in range(steps):
        half = batch // 2
        idx_user = rng.integers(0, len(user_examples), size=max(1, half))
        idx_replay = rng.integers(0, len(replay_examples), size=batch - len(idx_user))
        batch_examples = [user_examples[i] for i in idx_user] + [replay_examples[i] for i in idx_replay]
        idx, targets, mask = make_batch(tok, batch_examples, 40, cfg["block"], rng)
        lr_t = lr_schedule(step, steps, lr, warmup=max(5, steps // 10))
        loss, grads = loss_and_grads(p, cfg, idx, targets, mask, drop=0.05, rng=rng)
        clip_grads(grads, max_norm=1.0)
        opt.step(p, grads, lr_t)
        last_loss = loss
        if step % log_every == 0 or step == steps - 1:
            log(f"  шаг {step+1:4d}/{steps}  loss={loss:.3f}  lr={lr_t:.5f}  {time.time()-t0:.0f}с")
    return time.time() - t0, last_loss


# ======================================================================================
# 4. Версия приложения (AndroidManifest.xml) — повышаем при пересборке
# ======================================================================================
def bump_version(manifest_path):
    text = manifest_path.read_text(encoding="utf-8")
    code_m = re.search(r'android:versionCode="(\d+)"', text)
    name_m = re.search(r'android:versionName="([\d.]+)"', text)
    old_code = int(code_m.group(1))
    old_name = name_m.group(1)
    new_code = old_code + 1
    parts = old_name.split(".")
    parts[-1] = str(int(parts[-1]) + 1)
    new_name = ".".join(parts)
    text = text.replace(f'android:versionCode="{old_code}"', f'android:versionCode="{new_code}"')
    text = text.replace(f'android:versionName="{old_name}"', f'android:versionName="{new_name}"')
    manifest_path.write_text(text, encoding="utf-8")
    return (old_code, old_name), (new_code, new_name)


# ======================================================================================
# main
# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description="Дообучение Mini AI новыми парами вопрос→ответ")
    ap.add_argument("--pairs", required=True, help="файл с новыми парами (JSON экспорт из приложения)")
    ap.add_argument("--steps", type=int, default=250, help="шагов дообучения (по умолчанию 250, ~5-10 мин)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=8e-4, help="lr fine-tune (меньше, чем при обучении с нуля)")
    ap.add_argument("--replay", default=str(ROOT / "data/replay_pairs.json"))
    ap.add_argument("--build", action="store_true", help="поднять версию и пересобрать APK")
    ap.add_argument("--dry-run", action="store_true", help="только метрики ДО, без обучения (проверка файла)")
    args = ap.parse_args()

    if not MODEL_PATH.exists():
        print("ОШИБКА: нет model_transformer.json — сначала обучи базовую модель.")
        sys.exit(1)

    # ---------- 1. новые пары ----------
    new_pairs = load_pairs_file(args.pairs)
    if not new_pairs:
        print("ОШИБКА: в файле не найдено ни одной пары (ожидается q/question/variants + a/answer).")
        sys.exit(1)
    print(f"Новых пар в файле: {len(new_pairs)}")

    existing = json.loads(USER_PAIRS_PATH.read_text(encoding="utf-8")) if USER_PAIRS_PATH.exists() else []
    merged, added, updated = merge_pairs(existing, new_pairs)
    print(f"data/user_pairs.json: было {len(existing)}, добавлено {added}, обновлено {updated}, итого {len(merged)}")
    USER_PAIRS_PATH.parent.mkdir(exist_ok=True)
    USER_PAIRS_PATH.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---------- 2. модель и метрики ДО ----------
    p, cfg, tok, meta = load_model(str(MODEL_PATH))
    block = cfg["block"]
    print(f"\nМодель: {n_params(p):,} параметров, словарь {len(tok)} символов")

    before_control = eval_control(p, cfg, tok, block)
    before_hits, before_details = eval_pairs_exact(p, cfg, tok, block, merged)
    print(f"\nДО обучения:")
    print(f"  новые пары, точный ответ: {before_hits}/{len(merged)}")
    print(f"  контрольные вопросы (для справки): " + "; ".join(f"«{q}»→«{before_control[q]}»" for q in CONTROL_Q[:3]) + " …")

    if args.dry_run:
        print("\n--dry-run: обучение пропущено.")
        return

    # ---------- 3. replay (защита от забывания) ----------
    replay_path = Path(args.replay)
    if replay_path.exists():
        replay_pairs = json.loads(replay_path.read_text(encoding="utf-8"))
        print(f"\nReplay-пар (старые, против забывания): {len(replay_pairs)}")
    else:
        print(f"\nВНИМАНИЕ: {replay_path} не найден — обучаем без replay, "
              f"запусти tools/make_replay.py заранее. Рискуем забыть старое.")
        replay_pairs = []

    # ---------- 4. дообучение ----------
    print(f"\nДообучение: {args.steps} шагов, batch={args.batch}, lr={args.lr}")
    train_time, last_loss = finetune(p, cfg, tok, merged, replay_pairs, args.steps, args.batch, args.lr)
    print(f"Готово за {train_time:.0f} сек ({train_time/60:.1f} мин), финальный loss={last_loss:.3f}")

    # ---------- 5. метрики ПОСЛЕ ----------
    after_control = eval_control(p, cfg, tok, block)
    after_hits, after_details = eval_pairs_exact(p, cfg, tok, block, merged)
    same, total = compare_control(before_control, after_control)
    print(f"\nПОСЛЕ обучения:")
    print(f"  новые пары, точный ответ: {after_hits}/{len(merged)}  (было {before_hits}/{len(merged)})")
    print(f"  контрольные вопросы не испортились: {same}/{total}")
    for q in CONTROL_Q:
        mark = "✓" if before_control[q] == after_control[q] else "✗ ИЗМЕНИЛОСЬ"
        print(f"    {mark}  «{q}»  до: «{before_control[q]}»  после: «{after_control[q]}»")
    print("\n  Новые пары подробно:")
    for d in after_details:
        mark = "✓" if d["ok"] else "✗"
        print(f"    {mark}  «{d['q']}» → «{d['got']}»  (ожидали «{d['expected']}»)")

    if same < total - 2:
        print(f"\nВНИМАНИЕ: испортилось больше 2 старых ответов ({total - same}/{total}) — "
              f"возможно, стоит уменьшить --lr или --steps, или увеличить replay.")

    # ---------- 6. сохранение весов ----------
    meta = dict(meta)
    meta.pop("format", None); meta.pop("config", None); meta.pop("vocab", None)
    meta.pop("max_q", None); meta.pop("order", None); meta.pop("shapes", None); meta.pop("weights", None)
    meta["finetuned_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["finetune_pairs"] = len(merged)
    meta["finetune_steps"] = args.steps
    out_dict = export_model(p, cfg, tok, meta)
    text = json.dumps(out_dict, ensure_ascii=False, separators=(",", ":"))
    MODEL_PATH.write_text(text, encoding="utf-8")
    (ROOT / "app/www/model_transformer.js").write_text(
        "window.MINI_AI_TRANSFORMER=" + text + ";\n", encoding="utf-8")
    print(f"\nВеса сохранены: {MODEL_PATH} ({len(text)/1024:.0f} КБ)")

    # ---------- 7. сборка APK ----------
    if args.build:
        (old_code, old_name), (new_code, new_name) = bump_version(MANIFEST_PATH)
        print(f"\nВерсия приложения: {old_name} ({old_code}) → {new_name} ({new_code})")
        print("Пересборка APK (android/build.sh)…")
        r = subprocess.run(["bash", "android/build.sh"], cwd=str(ROOT))
        if r.returncode != 0:
            print("ОШИБКА сборки APK (см. вывод выше).")
            sys.exit(1)
        apk = ROOT / "android/build/MiniAI.apk"
        print(f"\nAPK готов: {apk} ({apk.stat().st_size/1024:.0f} КБ)")


if __name__ == "__main__":
    main()
