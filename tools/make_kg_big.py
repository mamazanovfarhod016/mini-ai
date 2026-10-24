import base64, io, tarfile, json, os
os.makedirs("/tmp/kg_reader_big", exist_ok=True)
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    for f in ["mini_ai/__init__.py", "mini_ai/transformer.py", "mini_ai/torch_engine.py", "mini_ai/backend.py", "tools/reader_gsm8k.py", "data/gsm_new/clean2.jsonl"]:
        t.add(f)
b = base64.b64encode(buf.getvalue()).decode()
code = f'''"""Объём ×2: GSM8K-train (5978 планов) + наши 6377 задач. Тот же читатель (d=192, 3 слоя), но 25 мин вместо 7.
Контроль в том же ядре: только GSM8K, те же 25 мин. Сравнение: лучший чекпойнт и полный тест 1319."""
import os, io, base64, tarfile, sys, subprocess, json
os.chdir("/kaggle/working")
tarfile.open(fileobj=io.BytesIO(base64.b64decode({b!r})), mode="r:gz").extractall(".")
res = {{}}
for tag, extra in (("gsm8k+ours", "data/gsm_new/clean2.jsonl"), ("gsm8k", "")):
    env = dict(os.environ, MIN="25", D_T="192", NL="3", NH="6", BS="64", N_TEST_CK="300", AUG="4", DROP="0.2", FRAC="1.0", EXTRA=extra, OUT=f"/kaggle/working/{{tag}}")
    subprocess.run([sys.executable, "tools/reader_gsm8k.py"], env=env)
    try:
        m = json.load(open(f"/kaggle/working/{{tag}}/metrics.json")); res[tag] = dict(train=m["train_plans"], best_ck=m["best_ck"], best_step=m["best_step"], full=m["acc_full"], ctrl=m["ctrl_full"], curve=[(c["min"], round(c["acc"], 3)) for c in m["curve"]])
    except Exception as e: res[tag] = dict(error=str(e))
    json.dump(res, open("/kaggle/working/big.json", "w"), indent=1)
print("\\n=== ОБЪЁМ ×2 ===")
for k, v in res.items(): print(k, v)
'''
open("/tmp/kg_reader_big/big.py", "w").write(code)
json.dump({"id": "mamazanovfarhod016/mini-ai-reader-big", "title": "mini-ai-reader-big", "code_file": "big.py", "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, open("/tmp/kg_reader_big/kernel-metadata.json", "w"))
print(len(b) // 1024, "KB")
