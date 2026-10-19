"""Бенчмарки как у современных ИИ (19.10.2026, решение Фархода: после полного прохождения класса — бенчмарк,
результаты ему, материалы удалить).

    .venv/bin/python -m server.benchmarks MODEL.json [--label "после 1 класса"]   # → docs/BENCHMARKS.md + json

Наборы (официальные тестовые части, модель их НИКОГДА не видит в учёбе):
  • ARC-Easy, ARC-Challenge (AllenAI, CC-BY-SA)       — школьная наука, 4 варианта, случайный уровень 25%
  • MMLU (Hendrycks et al., MIT): элементарная и школьная математика, школьная физика, школьная и вузовская
    информатика, компьютерная безопасность           — 4 варианта, случайно 25%
  • GSM8K test (OpenAI, MIT), en + ru                  — задачи со словами, свой ответ числом, случайно ≈0%
Метод для вариантов — как в lm-evaluation-harness (EleutherAI): выбираем вариант с наибольшей средней
вероятностью токенов ответа после вопроса (acc_norm по длине). Ответ модель не пишет — только оценивает.
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
BENCH = Path.home() / "Developer/mini-ai-bench"
LETTERS = "ABCDE"
MMLU = ["elementary_mathematics", "high_school_mathematics", "high_school_physics", "high_school_computer_science",
        "college_computer_science", "computer_security"]
N_ARC, N_GSM = 500, 200


def _logsoftmax(x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


class Scorer:
    def __init__(self, path):
        from mini_ai.transformer import KVModel, load_model
        self.p, self.cfg, self.tok, self.meta = load_model(path)
        self.kv = KVModel(self.p, self.cfg)
        self.max_q = self.meta.get("max_q", 24)
        self.block = self.cfg["block"]

    def choice_logprob(self, question, choices):
        """Средний log p(токены варианта | вопрос) для каждого варианта — одним батчем."""
        from mini_ai.transformer import EOS, PAD, forward
        from train_transformer import build_prompt
        pr = build_prompt(self.tok, question, max(self.max_q, 64))
        seqs, spans = [], []
        for c in choices:
            a = self.tok.encode(str(c))[: max(1, self.block - len(pr) - 2)]
            s = (pr + a + [EOS])[: self.block]
            seqs.append(s)
            spans.append((len(pr), len(s)))
        T = max(len(s) for s in seqs)
        idx = np.full((len(seqs), T), PAD, dtype=np.int64)
        for i, s in enumerate(seqs):
            idx[i, : len(s)] = s
        logits = forward(self.p, self.cfg, idx)
        logits = logits[0] if isinstance(logits, tuple) else logits
        lp = _logsoftmax(np.asarray(logits, dtype=np.float64))
        out = []
        for i, (a, b) in enumerate(spans):
            tgt = idx[i, a:b]
            vals = lp[i, a - 1:b - 1][np.arange(b - a), tgt]
            out.append(float(vals.mean()))
        return out

    def generate(self, q):
        from train_transformer import greedy_answer
        return greedy_answer(self.kv, self.tok, q, self.block, max(self.max_q, 64))


def _mc(scorer, items):
    ok = 0
    for q, choices, gold in items:
        s = scorer.choice_logprob(q, choices)
        ok += int(np.argmax(s)) == gold
    return ok / max(1, len(items)), len(items)


def load_sets(seed=0):
    import pyarrow.parquet as pq
    rng = random.Random(seed)
    sets = {}
    for name in ("ARC-Easy", "ARC-Challenge"):
        rows = pq.read_table(BENCH / f"{name}.parquet").to_pylist()
        rows = rng.sample(rows, min(N_ARC, len(rows)))
        items = []
        for r in rows:
            labels = list(r["choices"]["label"])
            if r["answerKey"] not in labels:
                continue
            items.append((r["question"], list(r["choices"]["text"]), labels.index(r["answerKey"])))
        sets[name] = items
    for sub in MMLU:
        f = BENCH / f"mmlu_{sub}.parquet"
        if f.exists():
            sets["MMLU " + sub] = [(r["question"], list(r["choices"]), int(r["answer"]))
                                   for r in pq.read_table(f).to_pylist()]
    return sets


def run(model, label=""):
    t0 = time.time()
    sc = Scorer(model)
    res = {}
    for name, items in load_sets().items():
        acc, n = _mc(sc, items)
        res[name] = {"acc": round(acc, 4), "n": n, "random": 0.25}
    gl = Path.home() / "Developer/mini-ai-books/gsm8k/levels.json"
    if gl.exists():
        test = json.loads(gl.read_text())["test"]
        for lang in ("en", "ru"):
            xs = random.Random(1).sample([x for x in test if x["lang"] == lang], N_GSM)
            ok = 0
            for x in xs:
                out = sc.generate(x["q"])
                nums = re.findall(r"-?\d+", out.split(":")[-1])
                ok += bool(nums) and nums[-1] == x["answer"]
            res[f"GSM8K {lang}"] = {"acc": round(ok / len(xs), 4), "n": len(xs), "random": 0.0}
    params = int(sum(v.size for v in sc.p.values()))
    rec = {"ts": time.strftime("%Y-%m-%d %H:%M"), "label": label, "model": str(model), "params": params,
           "minutes": round((time.time() - t0) / 60, 1), "results": res}
    hist = ROOT / "server_data/benchmarks.json"
    allr = json.loads(hist.read_text()) if hist.exists() else []
    allr.append(rec)
    hist.write_text(json.dumps(allr, ensure_ascii=False, indent=1))
    _write_md(allr)
    return rec


def _write_md(allr):
    names = list(allr[-1]["results"])
    lines = ["# Бенчмарки Mini AI", "",
             "Официальные тестовые части; модель их в учёбе не видит. Случайное угадывание: 25% (4 варианта), "
             "GSM8K около 0%. Метод для вариантов — как в EleutherAI lm-evaluation-harness (вероятность ответа).", "",
             "| Когда | Этап | Параметров | " + " | ".join(names) + " |",
             "|---|---|---|" + "---|" * len(names)]
    for r in allr:
        cells = [f"{r['results'][n]['acc'] * 100:.1f}%" if n in r["results"] else "—" for n in names]
        lines.append(f"| {r['ts']} | {r['label']} | {r['params'] / 1e6:.1f} млн | " + " | ".join(cells) + " |")
    (ROOT / "docs/BENCHMARKS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    rec = run(a.model, a.label)
    for k, v in rec["results"].items():
        print(f"{k:40s} {v['acc'] * 100:5.1f}%  (n={v['n']}, случайно {v['random'] * 100:.0f}%)")


if __name__ == "__main__":
    main()
