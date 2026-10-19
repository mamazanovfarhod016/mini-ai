"""
Обучение Mini AI на Kaggle (бесплатно, не нагружает наш сервер с METTA). Только скрипт, без ИИ.

    python3 -m server.kaggle_train --steps 6000 --out server_data/models/_candidate_full.json \
        [--intents server_data/intents_merged.json] [--math data/math_pairs.json] [-- <доп. аргументы>]

Как работает:
  1. собирает «посылку»: код (mini_ai/, train_transformer.py) + данные (intents, примеры) → tar.gz → base64;
  2. делает из неё один Python-скрипт и запускает его на Kaggle (`kaggle kernels push`), приватно;
  3. ждёт окончания (`kaggle kernels status`), скачивает модель и лог (`kaggle kernels output`).
Токен Kaggle — в ~/.kaggle/access_token (вне git). Аккаунт подтверждён по телефону → дают
видеокарту Tesla T4; обучение идёт с --gpu (mini_ai/backend.py: numpy → CuPy, формулы те же).
Код обучения тот же самый, с нуля; Kaggle просто чужой компьютер.
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
for g in ("tools/make_school_math.py", "tools/make_reasoning.py", "tools/make_logic_math.py"):
    if os.path.exists(g):
        subprocess.run([sys.executable, g], check=True)
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


DATASET = "mini-ai-init"      # приватный набор данных Kaggle: модель, с которой продолжаем (> 1 МБ в посылку не влезает)


def upload_init(model_path, log=print, wait_s=600):
    """Кладём текущую модель в приватный набор данных Kaggle (новой версией) и ждём, пока он будет готов."""
    with tempfile.TemporaryDirectory() as d:
        Path(d, "model.json").write_bytes(Path(model_path).read_bytes())
        Path(d, "dataset-metadata.json").write_text(json.dumps(
            {"title": DATASET, "id": f"{USER}/{DATASET}", "licenses": [{"name": "CC0-1.0"}]}), encoding="utf-8")
        res = kaggle("datasets", "version", "-p", d, "-m", time.strftime("%Y-%m-%d %H:%M"), timeout=600)
        if "not found" in res.lower() or "404" in res:
            res = kaggle("datasets", "create", "-p", d, timeout=600)
    if "successful" not in res.lower() and "being created" not in res.lower():
        raise RuntimeError(f"не удалось загрузить модель на Kaggle: {res[-300:]}")
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if "ready" in kaggle("datasets", "status", f"{USER}/{DATASET}").lower():
            log(f"Kaggle: модель для продолжения загружена ({Path(model_path).stat().st_size // 1024} КБ)")
            return
        time.sleep(15)
    raise TimeoutError("Kaggle: набор данных с моделью не готов")


def run(intents=None, math=None, steps=6000, extra=(), out=None, log=print, gpu=True, poll=60, max_wait_h=11,
        init=None, logic=False):
    """init — путь к текущей модели: учим ДАЛЬШЕ с неё (постоянные знания), а не с нуля."""
    files = [(ROOT / "mini_ai", "mini_ai"), (ROOT / "train_transformer.py", "train_transformer.py"),
             (Path(intents) if intents else ROOT / "data/intents.json", "data/intents.json")]
    args = ["--tokenizer", "bpe", "--vocab", "1000", "--steps", str(steps), "--warmup", "300", "--drop", "0.05",
            "--intents", "data/intents.json", *extra]
    if gpu:
        # на видеокарте backprop считает PyTorch (он уже стоит на Kaggle); сеть и Adam — наши,
        # совпадение с numpy-эталоном проверяет tests/test_torch_engine.py
        args += ["--engine", "torch"]
    maths = []
    if math:
        files.append((Path(math), "data/math_pairs.json"))
        maths.append("data/math_pairs.json")
    # школьная математика 1–6 класс: файл 8 МБ не влезает в посылку Kaggle (лимит ~1 МБ) —
    # шлём генератор (он детерминированный, seed=0) и собираем задания прямо там
    gen = ROOT / "tools/make_school_math.py"
    if gen.exists():
        files.append((gen, "tools/make_school_math.py"))
        maths.append("data/school_math.json")
    # 19.10: логика — пошаговые рассуждения (make_reasoning) и цепочки/минусы/формулы (make_logic_math);
    # файлы большие → шлём генераторы (детерминированные), задания собираются на Kaggle
    for g, data in ((ROOT / "tools/make_reasoning.py", "data/reasoning.json"),
                    (ROOT / "tools/make_logic_math.py", "data/logic_math.json")):
        if logic and g.exists():
            files.append((g, f"tools/{g.name}"))
            maths.append(data)
    if maths:
        args += ["--math", ",".join(maths)]
    if init:
        upload_init(init, log=log)
        args += ["--init", f"/kaggle/input/{DATASET}/model.json"]
    b64 = bundle(files)
    script = f"BUNDLE = {b64!r}\nARGS = {json.dumps(args)}\n" + RUNNER
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "train.py").write_text(script, encoding="utf-8")
        meta = {"id": f"{USER}/{SLUG}", "title": SLUG, "code_file": "train.py", "language": "python",
                "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_internet": False,
                "dataset_sources": [f"{USER}/{DATASET}"] if init else [], "competition_sources": [], "kernel_sources": []}
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
    ap.add_argument("--init", help="модель, с которой продолжить обучение (постоянные знания)")
    ap.add_argument("--logic", action="store_true", help="+ пошаговые рассуждения и логика (reasoning, logic_math)")
    ap.add_argument("extra", nargs="*", help="доп. аргументы train_transformer.py (после --)")
    a = ap.parse_args()
    run(a.intents, a.math, a.steps, a.extra, a.out, gpu=not a.cpu, init=a.init, logic=a.logic)


if __name__ == "__main__":
    main()
