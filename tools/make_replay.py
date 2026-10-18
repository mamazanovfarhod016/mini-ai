"""
Берём выборку исходных обучающих пар (вопрос→ответ) из data/intents.json —
нужна для «replay» (защита от забывания) при дообучении на телефоне и на сервере.

Используем ТЕ ЖЕ train-пары, что train_transformer.py строит для обучения
(build_pairs/augment_pairs, но без опечаток — чистые пары), берём случайную,
но равномерную по намерениям выборку N штук.

Запуск:  .venv/bin/python tools/make_replay.py --n 450
Результат:
  data/replay_pairs.json         [{"q":..., "a":...}, ...]
  app/www/replay_pairs.js        window.MINI_AI_REPLAY = [...]; (для приложения)
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=450)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT))
    from train_transformer import build_pairs  # noqa: E402
    import random as pyrandom

    data = json.loads((ROOT / "data/intents.json").read_text(encoding="utf-8"))
    intents = data["intents"]
    rng_py = pyrandom.Random(args.seed)
    train_pairs, _ = build_pairs(intents, rng_py, split_seed=args.seed)

    # Равномерно по намерениям: берём round(n / кол-во намерений) с каждого тега,
    # перемешиваем внутри тега, затем общий шаффл.
    by_tag = {}
    for q, a, tag, lang in train_pairs:
        by_tag.setdefault(tag, []).append({"q": q, "a": a})
    tags = sorted(by_tag)
    per_tag = max(1, args.n // len(tags))
    rng = random.Random(args.seed)
    sample = []
    for tag in tags:
        items = by_tag[tag][:]
        rng.shuffle(items)
        sample.extend(items[:per_tag])
    rng.shuffle(sample)
    sample = sample[: args.n]

    out_path = ROOT / "data/replay_pairs.json"
    out_path.write_text(json.dumps(sample, ensure_ascii=False, indent=1), encoding="utf-8")
    js_path = ROOT / "app/www/replay_pairs.js"
    js_path.write_text(
        "window.MINI_AI_REPLAY=" + json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(f"replay-пар: {len(sample)} (из {len(tags)} намерений, ~{per_tag} на тег)")
    print(f"Сохранено: {out_path}, {js_path}")


if __name__ == "__main__":
    main()
