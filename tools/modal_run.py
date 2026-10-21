"""modal_run.py — любой tools/*.py Mini AI на Modal CPU (параллельно с сервером).
Код репо + data (babi, sciq) + meaning_base.npz кладутся в Volume «mini-ai» один раз (upload), дальше — только изменённые файлы.
Внутри контейнера воссоздаётся та же раскладка: ~/Developer/mini-ai (код) и ~/Developer/mini-ai-runs (результаты), поэтому
скрипты работают без правок. Результаты (*.json, *.log, growth.json) забираются обратно командой pull.

  modal run tools/modal_run.py --cmd "tools/school.py"                 # школа на Modal (8 CPU, 16 ГБ)
  modal run tools/modal_run.py --cmd "tools/exp61_growth.py" --cpu 4
  modal run tools/modal_run.py --pull learning-core/school.log         # забрать файл из runs
Запуск в фоне: modal run --detach … (контейнер живёт без нашего процесса; вывод — modal app logs)."""
import os, sys, subprocess, time, json
import modal

app = modal.App("mini-ai")
vol = modal.Volume.from_name("mini-ai", create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.12").pip_install("numpy", "pandas", "pyarrow")
HOME = "/root"; REPO = f"{HOME}/Developer/mini-ai"; RUNS = f"{HOME}/Developer/mini-ai-runs"
LOCAL_REPO = os.path.expanduser("~/Developer/mini-ai"); LOCAL_RUNS = os.path.expanduser("~/Developer/mini-ai-runs")


def _sync_up():
    """локально → Volume: mini_ai/, tools/, data/{babi,school}, meaning_base.npz (по mtime+size)"""
    import hashlib
    with vol.batch_upload(force=True) as b:
        for sub in ("mini_ai", "tools", "data/babi", "data/school"):
            src = os.path.join(LOCAL_REPO, sub)
            if os.path.exists(src): b.put_directory(src, f"repo/{sub}")
        mb = os.path.join(LOCAL_RUNS, "learning-core", "meaning_base.npz")
        if os.path.exists(mb): b.put_file(mb, "runs/learning-core/meaning_base.npz")
    print("uploaded repo + data + meaning_base")


@app.function(image=image, volumes={"/vol": vol}, cpu=8, memory=16384, timeout=4 * 3600)
def run(cmd: str, cpu_hint: int = 8):
    os.makedirs(f"{HOME}/Developer", exist_ok=True)
    for link, target in ((REPO, "/vol/repo"), (RUNS, "/vol/runs")):
        if not os.path.islink(link): os.symlink(target, link)
    os.makedirs(f"{RUNS}/learning-core", exist_ok=True)
    env = dict(os.environ, HOME=HOME, OMP_NUM_THREADS=str(cpu_hint), PYTHONUNBUFFERED="1")
    t = time.time(); print(f"== {cmd} (cpu {cpu_hint}) ==", flush=True)
    p = subprocess.Popen(["python3"] + cmd.split(), cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last_commit = time.time()
    for line in p.stdout:
        if "Warning" in line or "warn(" in line: continue
        print(line.rstrip(), flush=True)
        if time.time() - last_commit > 120: vol.commit(); last_commit = time.time()   # промежуточные результаты видны снаружи
    p.wait(); vol.commit()
    print(f"== done rc={p.returncode} in {time.time()-t:.0f}s ==", flush=True)
    return p.returncode


@app.local_entrypoint()
def main(cmd: str = "", pull: str = "", cpu: int = 8, no_upload: bool = False):
    if pull:
        dst = os.path.join(LOCAL_RUNS, pull); os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, "wb") as f:
            for chunk in vol.read_file(f"runs/{pull}"): f.write(chunk)
        print("pulled", dst); return
    if not no_upload: _sync_up()
    if cmd: sys.exit(run.remote(cmd, cpu))
