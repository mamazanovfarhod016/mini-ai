"""
intents.json (разговор) → data/intents_pairs.json {train, test} в формате q/a — для учебного плана.
Сплит тот же, что в train_transformer.build_pairs (20% вопросов каждой темы спрятаны для экзамена).
    .venv/bin/python tools/make_intents_pairs.py
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from train_transformer import augment_pairs, build_pairs, fill_placeholders  # noqa: E402


def main():
    intents = json.loads((ROOT / "data/intents.json").read_text(encoding="utf-8"))["intents"]
    rng = random.Random(0)
    train, test = build_pairs(intents, rng, split_seed=0)
    train = augment_pairs(train, rng, n_typo=2)            # опечатки — как при обычном обучении
    out = {"train": [{"q": q, "a": fill_placeholders(a), "kind": tag, "lang": lang} for q, a, tag, lang in train],
           "test": [{"q": q, "a": fill_placeholders(a), "kind": tag, "lang": lang,
                     "answers": [fill_placeholders(x) for x in next(i for i in intents if i["tag"] == tag)["responses"].get(lang, [])]}
                    for q, a, tag, lang in test]}
    (ROOT / "data/intents_pairs.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"intents_pairs: train {len(out['train'])}, test {len(out['test'])}")


if __name__ == "__main__":
    main()
