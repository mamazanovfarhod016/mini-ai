"""school.py — ШКОЛА МЫШЛЕНИЯ: ядро проходит SciQ train (11 679 вопросов с ответами) как ученик с учителем.
На каждый вопрос: отвечает → учитель говорит «yes» или «no, it's X» → growth.py обновляет внимание, доверие,
правила «где ответ», связки слов. Ничего не переобучается градиентом; всё накопленное — в growth.json (читаемо).
Учебник (support) читается в память по ходу. Потом проверка на test 300 (холодно, без учителя) — тот же протокол,
что exp61, но ученик уже с опытом. Прогресс пишется каждые 500 вопросов, можно прервать и продолжить (resume).
Запуск: python3 tools/school.py [n=11679] [state=~/Developer/mini-ai-runs/learning-core/school_state]"""
import os, sys, json, time, shutil, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from mini_ai.experimental import ExperimentalMind
OUT = os.path.expanduser("~/Developer/mini-ai-runs/learning-core")
N = next((int(a[2:]) for a in sys.argv[1:] if a.startswith("n=")), 11679)
STATE = next((os.path.expanduser(a[6:]) for a in sys.argv[1:] if a.startswith("state=")), os.path.join(OUT, "school_state"))
RES = next((a[4:] for a in sys.argv[1:] if a.startswith("res=")), "school_result.json")
PROG = os.path.join(STATE, "progress.json")
tr = pd.read_parquet(os.path.expanduser("~/Developer/mini-ai/data/school/sciq/data/train-00000-of-00001.parquet")).dropna(subset=["support"]).head(N)
te = pd.read_parquet(os.path.expanduser("~/Developer/mini-ai/data/school/sciq/data/test-00000-of-00001.parquet")).dropna(subset=["support"]).head(300)

def hit(o, gold):
    a = (o.get("text") or "") + " " + str(o.get("answer") or "")
    return str(gold).lower() in a.lower()

os.makedirs(STATE, exist_ok=True)
prog = json.load(open(PROG)) if os.path.exists(PROG) else {"done": 0, "hits": 0, "corr": 0, "curve": []}
m = ExperimentalMind(school=True, wiki=False, state_dir=STATE)
t0 = time.time(); rows = list(tr.iterrows())
# память: учебник по уже пройденным вопросам (при resume перечитываем — быстро)
for i, (_, r) in enumerate(rows[:prog["done"]]): m.kb.read_text(str(r["support"]), "учебник")
win_h = 0; win_n = 0
for i, (_, r) in enumerate(rows):
    if i < prog["done"]: continue
    m.kb.read_text(str(r["support"]), "учебник")
    o = m.ask(r["question"], internet=False); ok = hit(o, r["correct_answer"])
    prog["hits"] += ok; win_h += ok; win_n += 1
    if ok: m.ask("yes", internet=False)
    elif o.get("kind") in ("answer", "clarify", "unknown"):
        m.ask(f"no, it's {r['correct_answer']}", internet=False); prog["corr"] += 1
    prog["done"] = i + 1
    if (i + 1) % 500 == 0 or i + 1 == len(rows):
        acc = round(win_h / max(1, win_n), 3); prog["curve"].append([i + 1, acc]); win_h = win_n = 0
        m.growth.save(); json.dump(prog, open(PROG, "w"))
        soft = [w for w in m.growth.attn if m.growth.is_soft_learned(w)]
        print(f"{i+1:6d}  last-500 acc {acc:.3f}  corrections {prog['corr']}  attn-soft {len(soft)}  slot-rules {sum(len(v) for v in m.growth.slot_rules.values())}  "
              f"forms {len([k for k in m.growth.reliab if not k.endswith('|*')])}  {time.time()-t0:.0f}s", flush=True)
print("--- growth report ---"); print(m.growth.report())
# проверка: test 300, без учителя, учебник test в память
for _, r in te.iterrows(): m.kb.read_text(str(r["support"]), "учебник")
T = [hit(m.ask(r["question"], internet=False), r["correct_answer"]) for _, r in te.iterrows()]
res = dict(when=time.strftime("%Y-%m-%d %H:%M"), train_seen=prog["done"], corrections=prog["corr"], curve=prog["curve"], test300=round(sum(T) / len(T), 3), seconds=round(time.time() - t0))
json.dump(res, open(os.path.join(OUT, RES), "w"), indent=1)
print(f"TEST 300 after school: {res['test300']:.3f}   (cold before school: 0.497)"); print(res)
