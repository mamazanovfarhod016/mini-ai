"""Условия (a) none и (b) context на базовой модели. Пишет результаты в runs/."""
import json, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.transformer import load_model, KVModel
from mini_ai.oneshot_bench import make_facts, score
from train_transformer import build_prompt

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/home/hermes/Developer/mini-ai/model_transformer_school.json"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 30
OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
os.makedirs(OUT, exist_ok=True)

p, cfg, tok, meta = load_model(MODEL)
kv = KVModel(p, cfg)
max_q = meta.get("max_q", 24)
facts = make_facts(N)

def ask(text, max_q_tok):
    ids = build_prompt(tok, text, max_q_tok)
    out, _ = kv.generate(ids, max_new=min(24, cfg["block"] - len(ids)), temperature=0.0)
    return tok.decode(out)

res = {}
for cond in ("none", "context"):
    t0 = time.time(); answers = []
    for f in facts:
        text = f["q"] if cond == "none" else f["fact"] + " " + f["q"]
        answers.append(ask(text, max_q if cond == "none" else cfg["block"] - 30))
    acc, rows = score(answers, facts)
    res[cond] = {"acc": acc, "sec": round(time.time() - t0, 1),
                 "rows": [{"q": f["q"], "fact": f["fact"], "gold": f["gold"], "ans": a, "got": g, "ok": h} for f, a, g, h in rows]}
    print(f"{cond}: {acc*100:.0f}%  ({res[cond]['sec']} s)")
    for f, a, g, h in rows[:5]:
        print("  ", "✓" if h else "✗", f["q"], "->", repr(a[:60]), "| gold", f["gold"])
json.dump(res, open(f"{OUT}/baseline_{os.path.basename(MODEL)}.json", "w"), ensure_ascii=False, indent=1)
