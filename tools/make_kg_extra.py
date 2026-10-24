import base64, io, tarfile, json
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    for f in ["mini_ai/__init__.py", "mini_ai/transformer.py", "mini_ai/torch_engine.py", "mini_ai/backend.py", "tools/reader_gsm8k.py", "data/gsm_new/clean.jsonl"]:
        t.add(f)
b = base64.b64encode(buf.getvalue()).decode()
code = f'''"""Тот же прогон, что frac=1.0 в mini-ai-reader-curve (7 мин, AUG=4), но + наши 1394 задачи. Контроль: без EXTRA, второй запуск."""
import os, io, base64, tarfile, sys, subprocess, json
os.chdir("/kaggle/working")
tarfile.open(fileobj=io.BytesIO(base64.b64decode({b!r})), mode="r:gz").extractall(".")
res = {{}}
for tag, extra in (("gsm8k", ""), ("gsm8k+ours", "data/gsm_new/clean.jsonl")):
    env = dict(os.environ, MIN="7", D_T="192", NL="3", NH="6", BS="64", N_TEST_CK="300", AUG="4", DROP="0.2", FRAC="1.0", EXTRA=extra, OUT=f"/kaggle/working/{{tag}}")
    subprocess.run([sys.executable, "tools/reader_gsm8k.py"], env=env)
    try:
        m = json.load(open(f"/kaggle/working/{{tag}}/metrics.json")); res[tag] = dict(train=m["train_plans"], best_ck=m["best_ck"], full=m["acc_full"], ctrl=m["ctrl_full"])
    except Exception as e: res[tag] = dict(error=str(e))
print("\\n=== ВКЛАД НАШИХ ЗАДАЧ ===")
for k, v in res.items(): print(k, v)
json.dump(res, open("/kaggle/working/extra.json", "w"), indent=1)
'''
open("/tmp/kg_reader_extra/extra.py", "w").write(code)
json.dump({"id": "mamazanovfarhod016/mini-ai-reader-extra", "title": "mini-ai-reader-extra", "code_file": "extra.py", "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, open("/tmp/kg_reader_extra/kernel-metadata.json", "w"))
print(len(b) // 1024, "KB")
