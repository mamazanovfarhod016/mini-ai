"""
app/www/known_words.js — слова, которые модель видела при обучении (вопросы train-сплита).

Нужен приложению, чтобы понять «вопрос — про то, чего я вообще не знаю» (Эйнштейн,
фотосинтез, столица Франции) → тогда оно само идёт искать в интернет. Маленькая модель
на таких вопросах всё равно уверенно выдаёт одну из заготовок, поэтому её собственной
уверенности (softmax) верить нельзя — проверено на 18 вопросах вне тем (tools/README).

    .venv/bin/python tools/make_known_words.py
"""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mini_ai.transformer import clean_question  # noqa: E402
from train_transformer import build_pairs  # noqa: E402


def main():
    intents = json.loads((ROOT / "data/intents.json").read_text(encoding="utf-8"))["intents"]
    train, _ = build_pairs(intents, random.Random(0), split_seed=0)
    words = set()
    for q, *_ in train:
        words.update(re.findall(r"[a-zа-яё']+", clean_question(q)))
    words = sorted(w for w in words if len(w) >= 3)
    out = ROOT / "app/www/known_words.js"
    out.write_text("window.MINI_AI_KNOWN_WORDS=" + json.dumps(words, ensure_ascii=False, separators=(",", ":")) + ";\n",
                   encoding="utf-8")
    print(f"{len(words)} слов → {out} ({out.stat().st_size/1024:.1f} КБ)")


if __name__ == "__main__":
    main()
