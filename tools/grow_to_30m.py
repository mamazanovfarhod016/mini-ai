"""
Рост Mini AI до ~31 млн параметров (план, мес. 1, неделя 2) — бесплатно, на нашем сервере, без обучения.

    .venv/bin/python tools/grow_to_30m.py --init server_data/models/_book_g2q2_try1.json \
        --out server_data/models/_grow31m.json

Что делает:
  1. словарь BPE дополняется до ~8 000 токенов на текстах «литературы» uz/ru/en (выборка по ~3 млн символов
     на язык, качаем потоком из GitHub Release data-text-20261020); старые номера токенов не меняются;
  2. сеть растёт 13 × 192 → 14 × 384 (голов 8 → 16, размер головы тот же — 24), mini_ai.transformer.grow_params:
     старые веса в «левый верхний угол», новый слой — «пропуск» (x → x);
  3. проверка «знания не потерялись»: экзамены прошлых четвертей (спрятанные задания учебников) до и после роста.
     После расширения d ответы могут чуть сместиться (LayerNorm считает по всем d) — это доучивается
     на первых шагах чтения; здесь фиксируем честную цифру до обучения.
Тяжёлое обучение (чтение ~0,6 млрд токенов) — потом на Kaggle, когда обновится недельная квота GPU.
"""
import argparse
import gzip
import io
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REL = "https://github.com/mamazanovfarhod016/mini-ai/releases/download/data-text-20261020/"


def sample_release(name, chars):
    """Первые `chars` символов файла из релиза, потоком (gzip распаковываем на лету, весь файл не качаем)."""
    out, n = [], 0
    with urllib.request.urlopen(REL + name, timeout=120) as r:
        with gzip.open(r, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(line)
                    n += len(line)
                    if n >= chars:
                        break
    return out


def exam_past(model_path, n=200):
    from server.book_school import exam, stages
    st = stages()
    items = [x for p in st if (p["grade"], p["quarter"]) <= (2, 2) for x in p["test"]]
    return exam(model_path, items[:n])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--layers", type=int, default=14)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--block", type=int, default=512)
    ap.add_argument("--chars", type=int, default=3_000_000, help="символов текста на язык для словаря")
    a = ap.parse_args()

    from mini_ai.transformer import export_model, grow_params, init_params, load_model, n_params
    t0 = time.time()
    p_old, cfg_old, tok, meta = load_model(a.init)
    max_q = json.loads(Path(a.init).read_text(encoding="utf-8")).get("max_q", 24)
    print(f"Было: {cfg_old}, параметров {n_params(p_old):,}, словарь {len(tok)}", flush=True)
    texts = []
    for f in ("literature_uz.txt.gz", "literature_ru.txt.gz", "literature_en.txt.gz"):
        part = sample_release(f, a.chars)
        print(f"  {f}: {len(part):,} строк", flush=True)
        texts += part
    before = len(tok)
    tok.extend(texts, extra=max(0, a.vocab - len(tok)), min_count=20, verbose=True)
    del texts
    print(f"Словарь: {before} → {len(tok)} (старые номера те же)", flush=True)

    p_new, cfg_new = init_params(len(tok), d=a.d, n_layer=a.layers, n_head=a.heads, block=max(a.block, cfg_old["block"]),
                                 pos=cfg_old.get("pos", "learned"))
    p_new = grow_params(p_new, cfg_new, p_old, cfg_old)
    del p_old
    N = n_params(p_new)
    print(f"Стало: {cfg_new}, параметров {N:,}", flush=True)
    # meta старой модели: responses/labels интентов и т.п. нужны приложению — сохраняем всё, кроме полей формата
    skip = {"format", "config", "vocab", "max_q", "tokenizer", "order", "shapes", "weights", "merges"}
    meta = {k: v for k, v in meta.items() if k not in skip}
    meta.update({"grown_from": Path(a.init).name, "grown_at": time.strftime("%Y-%m-%d %H:%M"), "n_params": N})
    out = export_model(p_new, cfg_new, tok, meta, tokenizer_type="bpe", max_q=max_q, merges=tok.merges)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"Сохранено {a.out} ({Path(a.out).stat().st_size / 1e6:.0f} МБ) за {time.time() - t0:.0f} с", flush=True)

    acc0, _ = exam_past(a.init)
    acc1, wrong = exam_past(a.out)
    print(f"Экзамен прошлых четвертей (200 спрятанных заданий): до роста {acc0:.0%}, после роста {acc1:.0%}", flush=True)
    for w in wrong[:5]:
        print("   ", w, flush=True)


if __name__ == "__main__":
    main()
