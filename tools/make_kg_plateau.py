import base64, io, tarfile, json, os, sys
# 25.10 плато 4 %: три быстрых теста на одном и том же читателе (d=192, 3 слоя, 25 мин, GSM8K+наши+MetaMath+Orca — как «open» 4,0 %)
#   vote — те же данные, + голосование 20 планов (без градиента)
#   pre  — сначала 12 мин английского (TinyStories), потом те же 25 мин задач, + голосование 10
#   tgsm — + 400k задач TinyGSM (решения-программы → план), + голосование 10
BASE = dict(MIN="25", D_T="192", NL="3", NH="6", BS="64", N_TEST_CK="300", AUG="0", DROP="0.2", FRAC="1.0", EXTRA="data/gsm_new/clean2.jsonl", HF_SETS="metamath,orca")
CFG = {"vote": dict(VOTE="20"),
       "pre": dict(PRE="tinystories", PRE_MIN="12", VOTE="10"),
       "tgsm": dict(HF_SETS="metamath,orca,tinygsm", TG_N="400000", VOTE="10"),
       # 25.10 утро: победители вместе (английский + TinyGSM весь шард), долго; маленький и побольше параллельно на двух T4
       "long": dict(HF_SETS="metamath,orca,tinygsm", TG_N="700000", PRE="tinystories", PRE_MIN="30", MIN="150", VOTE="10"),
       "long-big": dict(HF_SETS="metamath,orca,tinygsm", TG_N="700000", PRE="tinystories", PRE_MIN="30", MIN="150", VOTE="10", D_T="256", NL="6", NH="8", BS="48"),
       # 25.10: петля самоулучшения (STaR) на рецепте tgsm (8,7 %): 3 круга × 4000 новых вопросов TinyGSM без ответов, 8 планов, согласие ≥50 %
       "star": dict(HF_SETS="metamath,orca,tinygsm", TG_N="400000", VOTE="10", STAR="3", STAR_POOL="12000", STAR_PER="4000", STAR_K="8", STAR_FT="8", STAR_AGREE="0.5")}
only = sys.argv[1:] or ["vote", "pre", "tgsm"]
buf = io.BytesIO()
with tarfile.open(fileobj=buf, mode="w:gz") as t:
    for f in ["mini_ai/__init__.py", "mini_ai/transformer.py", "mini_ai/torch_engine.py", "mini_ai/backend.py", "tools/reader_gsm8k.py", "tools/tinygsm_conv.py", "data/gsm_new/clean2.jsonl"]:
        t.add(f)
b = base64.b64encode(buf.getvalue()).decode()
for tag in only:
    env = dict(BASE, **CFG[tag])
    d = f"/tmp/kg_reader_{tag}"; os.makedirs(d, exist_ok=True)
    code = f'''"""плато-тест {tag}: {CFG[tag]}"""
import os, io, base64, tarfile, sys, runpy
os.chdir("/kaggle/working")
tarfile.open(fileobj=io.BytesIO(base64.b64decode({b!r})), mode="r:gz").extractall(".")
os.environ.update({env!r})
sys.argv = ["tools/reader_gsm8k.py"]; runpy.run_path("tools/reader_gsm8k.py", run_name="__main__")
'''
    open(f"{d}/run.py", "w").write(code)
    json.dump({"id": f"mamazanovfarhod016/mini-ai-reader-{tag}", "title": f"mini-ai-reader-{tag}", "code_file": "run.py", "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4", "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, open(f"{d}/kernel-metadata.json", "w"))
    print(tag, d)
print(len(b) // 1024, "KB")
