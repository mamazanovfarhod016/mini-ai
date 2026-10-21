"""judge.py — быстрый честный судья для эволюции архитектур. Один набор, одна цифра, ~2 мин.
  A. SciQ test 300 (учебник в памяти) — понимание формулировки.           вес 1
  B. многошаговые: 120 вопросов, где ответ требует 2–3 связанных факта из абзаца (строятся из SciQ train:
     вопрос, чей ответ НЕ стоит в одном предложении со словами вопроса, а в соседнем/через связку).  вес 1
  C. bAbI 20 × 50 — рассуждение о состоянии мира.                           вес 1
  D. рост: 100 вопросов с учителем → 100 без; разница с холодным прогоном на тех же 100.  вес 1 (как +0.10 → 1.0)
Запуск: python3 tools/judge.py [tag] [quick]   → печатает таблицу, пишет judge_<tag>.json. quick = без D."""
import os, sys, json, time, shutil, re, glob, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from mini_ai.experimental import ExperimentalMind
from mini_ai.rules import World
from mini_ai.memory_v2 import toks, content_words
OUT = os.path.expanduser("~/Developer/mini-ai-runs/learning-core"); DATA = os.path.expanduser("~/Developer/mini-ai/data")
tag = next((a for a in sys.argv[1:] if a not in ("quick",)), "base"); quick = "quick" in sys.argv
def hit(o, g): return str(g).lower() in ((o.get("text") or "") + " " + str(o.get("answer") or "")).lower()
res = {"tag": tag, "when": time.strftime("%Y-%m-%d %H:%M")}; t0 = time.time()

# A
te = pd.read_parquet(f"{DATA}/school/sciq/data/test-00000-of-00001.parquet").dropna(subset=["support"]).head(300)
shutil.rmtree("/tmp/judgeA", ignore_errors=True); m = ExperimentalMind(school=True, wiki=False, state_dir="/tmp/judgeA")
for _, r in te.iterrows(): m.kb.read_text(str(r["support"]), "учебник")
res["A_sciq"] = round(sum(hit(m.ask(r["question"], internet=False), r["correct_answer"]) for _, r in te.iterrows()) / len(te), 3)
print(f"A SciQ 300        {res['A_sciq']:.3f}", flush=True); del m; gc.collect()

# B: multi-hop-ish — answer sentence shares <=1 content word with the question (needs linking through the paragraph)
tr = pd.read_parquet(f"{DATA}/school/sciq/data/train-00000-of-00001.parquet").dropna(subset=["support"])
B = []
for _, r in tr.iloc[5000:].iterrows():
    q, g, sup = r["question"], str(r["correct_answer"]).lower(), str(r["support"])
    sents = re.split(r"(?<=[.!?])\s+", sup); qa = {w for w in content_words(toks(q)) if len(w) > 3}
    with_ans = [s_ for s_ in sents if g in s_.lower()]
    if not with_ans or len(sents) < 3: continue
    if all(len(qa & set(content_words(toks(s_)))) <= 1 for s_ in with_ans): B.append(r)
    if len(B) >= 120: break
shutil.rmtree("/tmp/judgeB", ignore_errors=True); m = ExperimentalMind(school=True, wiki=False, state_dir="/tmp/judgeB")
for r in B: m.kb.read_text(str(r["support"]), "учебник")
res["B_multihop"] = round(sum(hit(m.ask(r["question"], internet=False), r["correct_answer"]) for r in B) / max(1, len(B)), 3)
print(f"B multi-hop {len(B):3d}   {res['B_multihop']:.3f}", flush=True); del m; gc.collect()

# C
tasks = sorted([re.search(r"task_(.+)_test", p).group(1) for p in glob.glob(f"{DATA}/babi/task_*_test.parquet")], key=lambda t: int(t.split("-")[0]))
ok = n = 0
for t in tasks:
    for _, r in pd.read_parquet(f"{DATA}/babi/task_{t}_test.parquet").head(50).iterrows():
        W = World(); [W.read(s) for s in r["passage"].split("\n") if s.strip()]
        try: a, _ = W.answer(r["question"])
        except Exception: a = None
        g = r["answer"].lower(); ok += a is not None and (set(str(a).lower().replace(",", " ").split()) == set(g.replace(",", " ").split()) or g in str(a).lower()); n += 1
res["C_babi"] = round(ok / n, 3); print(f"C bAbI 20×50      {res['C_babi']:.3f}", flush=True)

# D
if not quick:
    rows = list(tr.iloc[2000:2200].iterrows()); h1, h2 = rows[:100], rows[100:]
    shutil.rmtree("/tmp/judgeD0", ignore_errors=True); m0 = ExperimentalMind(school=True, wiki=False, state_dir="/tmp/judgeD0")
    for _, r in rows: m0.kb.read_text(str(r["support"]), "учебник")
    cold = sum(hit(m0.ask(r["question"], internet=False), r["correct_answer"]) for _, r in h2) / 100; del m0; gc.collect()
    shutil.rmtree("/tmp/judgeD1", ignore_errors=True); m1 = ExperimentalMind(school=True, wiki=False, state_dir="/tmp/judgeD1")
    for _, r in rows: m1.kb.read_text(str(r["support"]), "учебник")
    for _, r in h1:
        o = m1.ask(r["question"], internet=False)
        if hit(o, r["correct_answer"]): m1.ask("yes", internet=False)
        elif o.get("kind") in ("answer", "clarify", "unknown"): m1.ask(f"no, it's {r['correct_answer']}", internet=False)
    grown = sum(hit(m1.ask(r["question"], internet=False), r["correct_answer"]) for _, r in h2) / 100; del m1; gc.collect()
    res["D_growth"] = round(grown - cold, 3); res["D_cold"] = cold; res["D_grown"] = grown
    print(f"D growth          {res['D_growth']:+.3f}  (cold {cold:.2f} → grown {grown:.2f})", flush=True)
score = res["A_sciq"] + res["B_multihop"] + res["C_babi"] + (min(1.0, max(0.0, res.get("D_growth", 0) / 0.10)) if not quick else 0)
res["score"] = round(score, 3); res["seconds"] = round(time.time() - t0)
print(f"SCORE {res['score']:.3f}   ({res['seconds']}s)")
json.dump(res, open(os.path.join(OUT, f"judge_{tag}.json"), "w"), indent=1)
