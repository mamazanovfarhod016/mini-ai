"""
Обучение Mini AI на Kaggle (бесплатно, не нагружает наш сервер с METTA). Только скрипт, без ИИ.

    python3 -m server.kaggle_train --steps 6000 --out server_data/models/_candidate_full.json \
        [--intents server_data/intents_merged.json] [--math data/math_pairs.json] [-- <доп. аргументы>]

Как работает:
  1. собирает «посылку»: код (mini_ai/, train_transformer.py) + данные (intents, примеры) → tar.gz → base64;
  2. делает из неё один Python-скрипт и запускает его на Kaggle (`kaggle kernels push`), приватно;
  3. ждёт окончания (`kaggle kernels status`), скачивает модель и лог (`kaggle kernels output`).
Токен Kaggle — в ~/.kaggle/access_token (вне git). Видеокарта включится сама, когда аккаунт
подтверждён по телефону; пока обучение идёт на процессоре Kaggle (4 ядра).
Код обучения тот же самый — numpy, с нуля; Kaggle просто чужой компьютер.
"""
import argparse
import base64
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = str(ROOT / ".venv/bin/kaggle")
USER = "mamazanovfarhod016"
SLUG = "mini-ai-train"

RUNNER = r'''
import base64, io, os, subprocess, sys, tarfile, time
os.makedirs("/kaggle/working/w", exist_ok=True)
os.chdir("/kaggle/working/w")
tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz").extractall(".")
print("CPU:", os.cpu_count(), flush=True)
t = time.time()
r = subprocess.run([sys.executable, "train_transformer.py"] + ARGS + ["--out", "/kaggle/working/model.json",
                    "--no-app", "--log", "/kaggle/working/train_log.txt", "--ckpt_dir", "/kaggle/working/ckpt",
                    "--ckpt_every", "100000"])
print("EXIT", r.returncode, "за", round((time.time() - t) / 60, 1), "мин", flush=True)
sys.exit(r.returncode)
'''


def bundle(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for src, arc in files:
            tar.add(str(src), arcname=arc, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    return base64.b64encode(buf.getvalue()).decode()


def kaggle(*args, timeout=120):
    r = subprocess.run([KAGGLE, *args], capture_output=True, text=True, timeout=timeout)
    return (r.stdout + r.stderr).strip()


def run(intents=None, math=None, steps=6000, extra=(), out=None, log=print, gpu=True, poll=60, max_wait_h=11):
    files = [(ROOT / "mini_ai", "mini_ai"), (ROOT / "train_transformer.py", "train_transformer.py"),
             (Path(intents) if intents else ROOT / "data/intents.json", "data/intents.json")]
    args = ["--tokenizer", "bpe", "--vocab", "1000", "--steps", str(steps), "--warmup", "300", "--drop", "0.05",
            "--intents", "data/intents.json", *extra]
    if math:
        files.append((Path(math), "data/math_pairs.json"))
        args += ["--math", "data/math_pairs.json"]
    b64 = bundle(files)
    script = f"BUNDLE = {b64!r}\nARGS = {json.dumps(args)}\n" + RUNNER
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "train.py").write_text(script, encoding="utf-8")
        meta = {"id": f"{USER}/{SLUG}", "title": SLUG, "code_file": "train.py", "language": "python",
                "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_internet": False,
                "dataset_sources": [], "competition_sources": [], "kernel_sources": []}
        if gpu:
            meta["machine_shape"] = "NvidiaTeslaT4"
        (Path(d) / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        log(f"Kaggle: отправляю обучение ({len(b64)//1024} КБ, {steps} шагов)")
        res = kaggle("kernels", "push", "-p", d, timeout=300)
    if "successfully" not in res:
        raise RuntimeError(f"push не удался: {res[-300:]}")
    t0 = time.time()
    while True:
        time.sleep(poll)
        st = kaggle("kernels", "status", f"{USER}/{SLUG}")
        if "COMPLETE" in st:
            break
        if "ERROR" in st or "CANCEL" in st:
            raise RuntimeError(f"Kaggle: {st}")
        if time.time() - t0 > max_wait_h * 3600:
            raise TimeoutError("Kaggle: слишком долго")
    with tempfile.TemporaryDirectory() as d:
        kaggle("kernels", "output", f"{USER}/{SLUG}", "-p", d, timeout=600)
        model, tlog = Path(d) / "model.json", Path(d) / "train_log.txt"
        if not model.exists():
            logs = list(Path(d).glob("*.log"))
            tail = logs[0].read_text(errors="replace")[-1500:] if logs else "нет лога"
            raise RuntimeError(f"Kaggle: модели нет в результате. {tail}")
        out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(model.read_bytes())
        if tlog.exists():
            (out.parent / (out.stem + "_log.txt")).write_bytes(tlog.read_bytes())
    log(f"Kaggle: готово за {(time.time()-t0)/60:.0f} мин → {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intents"); ap.add_argument("--math")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cpu", action="store_true", help="без видеокарты")
    ap.add_argument("extra", nargs="*", help="доп. аргументы train_transformer.py (после --)")
    a = ap.parse_args()
    run(a.intents, a.math, a.steps, a.extra, a.out, gpu=not a.cpu)


if __name__ == "__main__":
    main()
