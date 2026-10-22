import sys, shutil, gc; sys.path.insert(0, ".")
import pandas as pd
from mini_ai.experimental import ExperimentalMind
tr = pd.read_parquet("data/school/sciq/data/train-00000-of-00001.parquet").dropna(subset=["support"])
def hit(o, g):
    a = ((o.get("text") or "") + " " + str(o.get("answer") or "")).lower(); return str(g).lower() in a
for lo in (2000, 4000, 6000):
    rows = list(tr.iloc[lo:lo + 300].iterrows()); h1, h2 = rows[:200], rows[200:]
    shutil.rmtree("/tmp/D0", ignore_errors=True); m0 = ExperimentalMind(school=True, wiki=False, state_dir="/tmp/D0")
    for _, r in rows: m0.kb.read_text(str(r["support"]), "учебник")
    cold = sum(hit(m0.ask(r["question"], internet=False), r["correct_answer"]) for _, r in h2) / 100; del m0; gc.collect()
    shutil.rmtree("/tmp/D1", ignore_errors=True); m1 = ExperimentalMind(school=True, wiki=False, state_dir="/tmp/D1")
    for _, r in rows: m1.kb.read_text(str(r["support"]), "учебник")
    nc = 0
    for _, r in h1:
        o = m1.ask(r["question"], internet=False)
        if hit(o, r["correct_answer"]): m1.ask("yes", internet=False)
        elif o.get("kind") in ("answer", "clarify", "unknown"): m1.ask(f"no, it's {r['correct_answer']}", internet=False); nc += 1
    used = 0; grown = 0
    for _, r in h2:
        o = m1.ask(r["question"], internet=False); grown += hit(o, r["correct_answer"]); used += bool(m1._advice)
    grown /= 100
    print(f"[{lo}] cold {cold:.2f} → grown {grown:.2f}  ({grown - cold:+.2f}) corrections {nc} cases {len(m1.cases.items)} advice-used {used}/100 gates-off {[k for k, v in m1.growth.gates.items() if not v]}", flush=True)
    del m1; gc.collect()
