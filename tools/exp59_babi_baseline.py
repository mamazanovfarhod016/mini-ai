"""exp59 — рассуждение на bAbI 15–19 (дедукция, индукция, позиция, размер, путь): база ядра ДО новых механизмов.

Что меряем: ядро как есть (Reader → факты, think.py цепочка, question.py рамка) на 1000 тестовых
историях каждой задачи. Ответ засчитан, если золотое слово есть в answer/text. Это КОНТРОЛЬ: ожидание —
низко (<30%), потому что в ядре нет правил вывода (транзитивность, «все X боятся Y» + «Z — X» ⇒ Z боится Y).
Числа фиксируются, чтобы следующий механизм (rules.py — вывод по типам/транзитивности) сравнивать честно.
"""
import os, sys, re, json, time, collections
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_ai.experimental import ExperimentalMind
DATA = os.path.expanduser("~/Developer/mini-ai/data/babi"); OUT = os.path.expanduser("~/Developer/mini-ai-runs/learning-core")
TASKS = ["15-basic_deduction", "16-basic_induction", "17-positional_reasoning", "18-size_reasoning", "19-path_finding", "6-yes_no_questions", "9-simple_negation", "10-indefinite_knowledge"]

def run_core(task, n=200):
    df = pd.read_parquet(f"{DATA}/task_{task}_test.parquet").head(n); ok = 0; ex = []
    for _, r in df.iterrows():
        m = ExperimentalMind(school=False, wiki=False, state_dir="/tmp/exp59_state"); m.use_dialog = False
        m.learn(r["passage"].replace("\n", " ")); o = m.ask(r["question"], internet=False)
        a = (o.get("text") or "") + " " + str(o.get("answer") or ""); h = r["answer"].lower() in a.lower(); ok += h
        if len(ex) < 2: ex.append((r["question"], o.get("answer"), r["answer"]))
    return ok / len(df), ex

if __name__ == "__main__":
    import shutil; shutil.rmtree("/tmp/exp59_state", ignore_errors=True); os.makedirs("/tmp/exp59_state", exist_ok=True)
    res = {}; t0 = time.time()
    for t in TASKS:
        # learned.txt копится между историями — чистим
        shutil.rmtree("/tmp/exp59_state", ignore_errors=True); os.makedirs("/tmp/exp59_state", exist_ok=True)
        acc, ex = run_core(t); res[t] = round(acc, 3); print(f"task {t}: ядро как есть {acc:.2f}   напр.: {ex[0]}")
    res["seconds"] = round(time.time() - t0, 1); json.dump(res, open(f"{OUT}/exp59_babi_baseline.json", "w"), indent=1); print(res)
