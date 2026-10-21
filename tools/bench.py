"""bench.py — единый прогон бенчмарков ядра (без обучения). Печатает таблицу и пишет JSON с датой.
  bAbI 11 задач × 1000 (rules.World) — рассуждение в простом мире
  exp57 вымышленный мир (ExperimentalMind.learn + ask) — скорость усвоения, 2–3 шага, после перезапуска
  SciQ 300 вопросов (ask с учебником) — понимание вопроса по фактам
  ops-set 13 вопросов на операторы (реальная память) — сравнение/да-нет/счёт
"""
import os, sys, json, time, re, shutil, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
OUT = os.path.expanduser("~/Developer/mini-ai-runs/learning-core"); DATA = os.path.expanduser("~/Developer/mini-ai/data/babi")
res = {"when": time.strftime("%Y-%m-%d %H:%M")}; t0 = time.time()

# 1. bAbI
from mini_ai.rules import World
TASKS = ["1-single_supporting_fact", "6-yes_no_questions", "8-lists_sets", "9-simple_negation", "10-indefinite_knowledge", "14-time_reasoning", "15-basic_deduction", "16-basic_induction", "17-positional_reasoning", "18-size_reasoning", "19-path_finding"]
babi = {}
for t in TASKS:
    df = pd.read_parquet(f"{DATA}/task_{t}_test.parquet"); ok = 0
    for _, r in df.iterrows():
        W = World(); [W.read(s) for s in r["passage"].split("\n") if s.strip()]
        a, _ = W.answer(r["question"]); g = r["answer"].lower()
        ok += a is not None and (set(str(a).lower().replace(",", " ").split()) == set(g.replace(",", " ").split()) or g in str(a).lower())
    babi[t] = round(ok / len(df), 3)
res["babi_mean"] = round(sum(babi.values()) / len(babi), 3); res["babi"] = babi
print(f"bAbI 11 задач: среднее {res['babi_mean']}  min {min(babi.values())} ({min(babi, key=babi.get)})")

# 2. exp57
sys.path.insert(0, os.path.join(os.path.dirname(__file__))); sys.argv = ["x"]
from exp57_learning_speed import world, hit
from mini_ai.experimental import ExperimentalMind
shutil.rmtree("/tmp/bench57", ignore_errors=True)
facts, direct, infer = world(); rng = random.Random(0); rng.shuffle(direct); rng.shuffle(infer); direct, infer = direct[:120], infer[:60]
m = ExperimentalMind(school=False, wiki=False, state_dir="/tmp/bench57"); t = time.time(); m.learn(" ".join(facts)); res["learn_350_sec"] = round(time.time() - t, 2)
res["w_direct"] = round(sum(hit(m.ask(q, internet=False).get("text"), g) for q, g in direct) / len(direct), 3)
res["w_infer2"] = round(sum(hit(m.ask(q, internet=False).get("text"), g) for q, g in infer) / len(infer), 3)
qs3 = [(q.replace("In which country was", "In which continent was"), "Vendar") for q, _ in infer if "born" in q][:30]
res["w_infer3"] = round(sum(hit(m.ask(q, internet=False).get("text"), g) for q, g in qs3) / len(qs3), 3)
del m; m2 = ExperimentalMind(school=False, wiki=False, state_dir="/tmp/bench57")
res["w_direct_restart"] = round(sum(hit(m2.ask(q, internet=False).get("text"), g) for q, g in direct) / len(direct), 3)
print(f"мир 350 фактов: усвоение {res['learn_350_sec']}s | прямые {res['w_direct']} | 2 шага {res['w_infer2']} | 3 шага {res['w_infer3']} | после перезапуска {res['w_direct_restart']}")

# 3. SciQ (учебник в памяти; вопрос по факту из support)
try:
    df = pd.read_parquet(os.path.expanduser("~/Developer/mini-ai/data/school/sciq/data/test-00000-of-00001.parquet")).dropna(subset=["support"]).head(300)
    ms = ExperimentalMind(school=True, wiki=False, state_dir="/tmp/bench_sciq"); ok = 0
    for _, r in df.iterrows():
        ms.kb.read_text(str(r["support"]), "учебник"); o = ms.ask(r["question"], internet=False); a = (o.get("text") or "") + " " + str(o.get("answer") or "")
        ok += str(r["correct_answer"]).lower() in a.lower()
    res["sciq_300"] = round(ok / len(df), 3); print(f"SciQ 300: {res['sciq_300']}")
except Exception as e: print("SciQ пропущен:", e)

# 4. операторы на реальной памяти
mr = ExperimentalMind(wiki=False, state_dir=os.path.expanduser("~/Developer/mini-ai-runs/memory-core/experimental_state"))
from mini_ai.dialog import Dialog; mr.dialog = Dialog(mr.kb)
OPS = [("Is Moscow bigger than Tashkent?", "yes"), ("Does Uzbekistan border Russia?", "not among"), ("Does Uzbekistan border Kazakhstan?", "yes"), ("How many countries border Uzbekistan?", "5"),
       ("Which countries border Kazakhstan?", "china"), ("Is Uzbekistan in Europe?", "no"), ("Is Uzbekistan in Central Asia?", "yes"), ("In which continent is Moscow?", "asia"), ("In which continent is Paris?", "europe"),
       ("What is the capital of France?", "paris"), ("What is the capital of Kazakhstan?", "astana"), ("In which continent is Astana?", "asia"), ("What do Uzbekistan and Kazakhstan have in common?", "countries")]
ok = 0; fails = []
for q, g in OPS:
    o = mr.ask(q, internet=False); t_ = (o.get("text") or "").lower(); h = g in t_; ok += h
    if not h: fails.append((q, (o.get("text") or "")[:80]))
res["ops_13"] = round(ok / len(OPS), 3); print(f"операторы на реальной памяти: {ok}/{len(OPS)}  промахи: {fails}")
res["seconds"] = round(time.time() - t0); json.dump(res, open(os.path.join(OUT, f"bench_{time.strftime('%Y%m%d_%H%M')}.json"), "w"), indent=1, ensure_ascii=False); print("sec", res["seconds"])
