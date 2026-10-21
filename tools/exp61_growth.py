"""exp61 — РОСТ МЫШЛЕНИЯ: становится ли ядро умнее от исправлений, а не только от фактов?
Протокол (SciQ test, 300 вопросов, учебник в памяти):
  A. холодный прогон: 300 вопросов без обратной связи → точность.
  B. рост: первые 150 вопросов — после каждого ответа учитель говорит «yes» (если верно) или «no, it's X».
     Ядро обновляет доверие к способам (how × форма вопроса), схемы, связки слов. Ничего не переобучается.
  C. вторые 150 вопросов — без обратной связи. Сравниваем с теми же 150 из A.
Рост есть, если C > A[150:]. Также: сколько форм вопросов получило опыт, как сместилось доверие.
Запуск: python3 tools/exp61_growth.py [n=300]"""
import os, sys, json, time, shutil, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from mini_ai.experimental import ExperimentalMind
from mini_ai.growth import qform
OUT = os.path.expanduser("~/Developer/mini-ai-runs/learning-core")
N = next((int(a[2:]) for a in sys.argv[1:] if a.startswith("n=")), 300)
df = pd.read_parquet(os.path.expanduser("~/Developer/mini-ai/data/school/sciq/data/test-00000-of-00001.parquet")).dropna(subset=["support"]).head(N)
rows = list(df.iterrows()); half = len(rows) // 2

def hit(o, gold):
    a = (o.get("text") or "") + " " + str(o.get("answer") or "")
    return str(gold).lower() in a.lower()

def fresh(tag):
    d = f"/tmp/exp61_{tag}"; shutil.rmtree(d, ignore_errors=True)
    m = ExperimentalMind(school=True, wiki=False, state_dir=d)
    for _, r in rows: m.kb.read_text(str(r["support"]), "учебник")
    return m

t0 = time.time()
# A. холодный
mA = fresh("A"); A = []
for _, r in rows: A.append(hit(mA.ask(r["question"], internet=False), r["correct_answer"]))
print(f"A cold: all {sum(A)/len(A):.3f}  first-half {sum(A[:half])/half:.3f}  second-half {sum(A[half:])/(len(A)-half):.3f}", flush=True)

# B. рост на первой половине
mB = fresh("B"); B = []; corr = 0
for _, r in rows[:half]:
    o = mB.ask(r["question"], internet=False); ok = hit(o, r["correct_answer"]); B.append(ok)
    if ok: mB.ask("yes", internet=False)
    else:
        if o.get("kind") in ("answer", "clarify", "unknown"):
            mB.ask(f"no, it's {r['correct_answer']}", internet=False); corr += 1
print(f"B growing (first half, with feedback): {sum(B)/len(B):.3f}, corrections given {corr}", flush=True)
# C. вторая половина без подсказок
C = []
for _, r in rows[half:]: C.append(hit(mB.ask(r["question"], internet=False), r["correct_answer"]))
print(f"C after growth (second half, no feedback): {sum(C)/len(C):.3f}   vs cold {sum(A[half:])/(len(A)-half):.3f}", flush=True)
print("--- growth report ---"); print(mB.growth.report())
res = dict(when=time.strftime("%Y-%m-%d %H:%M"), n=len(rows), cold_all=round(sum(A)/len(A), 3), cold_second=round(sum(A[half:])/(len(A)-half), 3),
           grow_first=round(sum(B)/len(B), 3), after_growth_second=round(sum(C)/len(C), 3), corrections=corr,
           forms_with_experience=len([k for k in mB.growth.reliab if not k.endswith("|*")]), rules=len(mB.fewshot.rules), seconds=round(time.time() - t0))
json.dump(res, open(os.path.join(OUT, "exp61_growth.json"), "w"), indent=1); print(res)
