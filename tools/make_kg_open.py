import base64, io, tarfile, json, os, sys
# две конфигурации на двух T4 параллельно: тот же читатель (чистое сравнение с 4,1 %) и побольше (данных теперь ×10)
CFG = {"open": dict(D_T="192", NL="3", NH="6"), "open-big": dict(D_T="256", NL="6", NH="8")}
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    for f in ["mini_ai/__init__.py", "mini_ai/transformer.py", "mini_ai/torch_engine.py", "mini_ai/backend.py", "tools/reader_gsm8k.py", "data/gsm_new/clean2.jsonl"]:
        t.add(f)
b = base64.b64encode(buf.getvalue()).decode()
for tag, cfg in CFG.items():
    d = f"/tmp/kg_reader_{tag}"; os.makedirs(d, exist_ok=True)
    env = dict(MIN="25", BS="64", N_TEST_CK="300", AUG="0", DROP="0.2", FRAC="1.0", EXTRA="data/gsm_new/clean2.jsonl", HF_SETS="metamath,orca", **cfg)
    code = f'''"""GSM8K + наши 6377 + открытые MetaMathQA(GSM) и Orca-Math 200k (чистка от теста внутри), {cfg}, 25 мин."""
import os, io, base64, tarfile, sys, runpy
os.chdir("/kaggle/working")
tarfile.open(fileobj=io.BytesIO(base64.b64decode({b!r})), mode="r:gz").extractall(".")
os.environ.update({env!r})
sys.argv = ["tools/reader_gsm8k.py"]; runpy.run_path("tools/reader_gsm8k.py", run_name="__main__")
'''
    open(f"{d}/run.py", "w").write(code)
    json.dump({"id": f"mamazanovfarhod016/mini-ai-reader-{tag}", "title": f"mini-ai-reader-{tag}", "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, open(f"{d}/kernel-metadata.json", "w"))
print(len(b) // 1024, "KB")
