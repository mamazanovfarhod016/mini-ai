"""
Сравнение char-модели и BPE-модели на ОДНИХ И ТЕХ ЖЕ вопросах.

    .venv/bin/python tools/compare_tokenizers.py [char.json] [bpe.json]

1. test-вопросы (тот же детерминированный сплит, что в train_transformer.py —
   модели их не видели);
2. 40 НОВЫХ перефразировок, которых нет в data/intents.json вообще.
Метрика: доля правильных НАМЕРЕНИЙ (greedy-ответ дословно = ответ нужного намерения).
Плюс: длина в токенах, скорость генерации, размер файла.
Результат → docs/tokenizer_comparison.md
"""
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mini_ai.transformer import KVModel, clean_question, load_model  # noqa: E402
from train_transformer import build_pairs, build_prompt, intent_of_answer  # noqa: E402

PARAPHRASES = [
    ("приветствую тебя дружище", "greeting"), ("здорово братан", "greeting"),
    ("salom aka", "greeting"), ("ну всё я пошёл", "goodbye"),
    ("до встречи друг", "goodbye"), ("как поживаешь", "how_are_you"),
    ("ну как ты там", "how_are_you"), ("чем занимаешься сейчас", "what_doing"),
    ("ты вообще кто такой", "who_are_you"), ("а зовут тебя как", "name"),
    ("кто тебя придумал", "creator"), ("кто твой автор", "creator"),
    ("огромное спасибо тебе", "thanks"), ("благодарю", "thanks"),
    ("да конечно", "yes"), ("нет не надо", "no"),
    ("сколько тебе годиков", "age"), ("что ты вообще умеешь делать", "abilities"),
    ("расскажи что нибудь смешное", "joke"), ("хочу анекдот", "joke"),
    ("какая погода на улице", "weather"), ("сколько сейчас часов", "time"),
    ("какая сегодня дата", "date"), ("мне нужна помощь", "help"),
    ("помоги пожалуйста", "help"), ("ты просто супер", "compliment"),
    ("мне очень грустно сегодня", "sad"), ("скучно мне", "bored"),
    ("у меня всё отлично", "good_mood"), ("ты дурак", "insult"),
    ("я ничего не понял", "not_understood"), ("что тебе больше всего нравится", "favorite"),
    ("ты где находишься", "where"), ("как ты устроен внутри", "how_work"),
    ("реши мне задачу", "out_of_scope"), ("ты ошибаешься", "wrong_feedback"),
    ("rahmat sizga", "thanks"), ("ishlaring qalay", "how_are_you"),
    ("xayr ko'rishguncha", "goodbye"), ("seni kim yaratgan", "creator"),
]


def run(path, qs, intents_by_tag):
    p, cfg, tok, meta = load_model(path)
    kv = KVModel(p, cfg)
    max_q = meta.get("max_q", 40)
    hits, n_tok, t0, details = 0, 0, time.time(), []
    for q, tag in qs:
        ids = build_prompt(tok, q, max_q)
        out, _ = kv.generate(ids, max_new=cfg["block"] - len(ids), temperature=0.0)
        n_tok += len(out)
        ans = tok.decode(out)
        ok = intent_of_answer(ans, intents_by_tag) == tag
        hits += ok
        details.append((q, tag, ans, ok))
    dt = time.time() - t0
    return {"acc": hits / len(qs), "hits": hits, "n": len(qs), "sec": dt,
            "tok_per_answer": n_tok / len(qs), "details": details,
            "V": cfg["V"], "block": cfg["block"],
            "size_kb": Path(path).stat().st_size / 1024}


def main():
    char_path = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "model_transformer.json")
    bpe_path = sys.argv[2] if len(sys.argv) > 2 else str(ROOT / "model_transformer_bpe.json")
    data = json.loads((ROOT / "data/intents.json").read_text(encoding="utf-8"))
    intents_by_tag = {it["tag"]: it for it in data["intents"]}
    known = {clean_question(p) for it in data["intents"] for p in it["patterns"]}
    para = [(q, t) for q, t in PARAPHRASES if clean_question(q) not in known]
    _, test_pairs = build_pairs(data["intents"], random.Random(0), split_seed=0)
    test_qs = sorted({(q, tag) for q, a, tag, lang in test_pairs})

    res = {}
    for name, path in (("char", char_path), ("bpe", bpe_path)):
        res[name] = {"test": run(path, test_qs, intents_by_tag),
                     "para": run(path, para, intents_by_tag)}
        print(name, {k: round(v["acc"] * 100, 1) for k, v in res[name].items()}, flush=True)

    L = ["# Char vs BPE — сравнение на одних и тех же вопросах", "",
         "| | char | BPE |", "|---|---|---|"]
    row = lambda label, f: L.append(f"| {label} | {f(res['char'])} | {f(res['bpe'])} |")
    row("Тест-вопросы (не видела), намерение верно",
        lambda r: f"{r['test']['acc']*100:.1f}% ({r['test']['hits']}/{r['test']['n']})")
    row(f"Новые перефразировки ({len(para)} шт.), намерение верно",
        lambda r: f"{r['para']['acc']*100:.1f}% ({r['para']['hits']}/{r['para']['n']})")
    row("Словарь", lambda r: r["test"]["V"])
    row("Контекст (block)", lambda r: r["test"]["block"])
    row("Шагов генерации на ответ", lambda r: f"{r['test']['tok_per_answer']:.1f}")
    row("Скорость (сек на ответ, Python)",
        lambda r: f"{r['test']['sec']/r['test']['n']*1000:.0f} мс")
    row("Размер файла", lambda r: f"{r['test']['size_kb']:.0f} КБ")
    L += ["", "## Перефразировки: ответы обеих моделей", "",
          "| вопрос | намерение | char | BPE |", "|---|---|---|---|"]
    for (q, tag, ca, cok), (_, _, ba, bok) in zip(res["char"]["para"]["details"],
                                                  res["bpe"]["para"]["details"]):
        L.append(f"| {q} | {tag} | {'✓' if cok else '✗'} {ca} | {'✓' if bok else '✗'} {ba} |")
    out = ROOT / "docs/tokenizer_comparison.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("→", out)


if __name__ == "__main__":
    main()
