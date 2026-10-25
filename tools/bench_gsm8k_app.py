"""bench_gsm8k_app — GSM8K полный тест (1319) тем читателем, который СЕЙЧАС стоит в приложении (mini_ai/plan_reader.py, numpy, CPU).
Проверка, что в приложении то же число, что на Kaggle (25,1 %). Печатает итог и 3 верные задачи с шагами."""
import os, sys, io, json, time, urllib.request
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from mini_ai import plan_reader as PR
p = os.path.expanduser("~/Developer/mini-ai/data/gsm8k_test.parquet")
if not os.path.exists(p):
    open(p, "wb").write(urllib.request.urlopen(urllib.request.Request("https://huggingface.co/datasets/openai/gsm8k/resolve/refs%2Fconvert%2Fparquet/main/test/0000.parquet", headers={"User-Agent": "mini-ai"}), timeout=120).read())
d = pd.read_parquet(p); r = PR.PlanReader(); info = r.info(); t0 = time.time(); ok = 0; good = []
for i, (q, a) in enumerate(zip(d["question"], d["answer"])):
    g = float(a.split("####")[-1].strip().replace(",", "")); res = r.solve(q)
    if res["answer"] is not None and abs(res["answer"] - g) < 1e-6:
        ok += 1
        if len(good) < 3: good.append((q[:110], [f"{PR.fmt(x)} {o} {PR.fmt(y)} = {PR.fmt(z)}" for x, o, y, z in res["steps"]], g))
    if i % 300 == 299: print(f"  {i+1}: {ok/(i+1):.1%}", flush=True)
print(f"ИТОГ GSM8K (приложение, {info['params']/1e6:.1f} млн, {os.path.basename(info['path'])}): {ok}/{len(d)} = {ok/len(d):.1%} | Kaggle: {info['gsm8k_full']:.1%} | {time.time()-t0:.0f} с")
for q, st, g in good: print("  ✓", q, "→", "; ".join(st), "=", PR.fmt(g))
