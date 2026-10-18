"""
Один запуск всего curriculum на Kaggle (GPU T4, интернет ВКЛЮЧЕН — для make_corpus/make_parallel).

    .venv/bin/python -m server.kaggle_curriculum --init model_transformer_school.json \
        --out server_data/curriculum_out [--plan docs/plan_tonight.json] [--smoke]

Как работает:
  1. upload_init(init) — кладём стартовую модель в приватный датасет mini-ai-init (как в kaggle_train.py).
  2. Бандл кода (mini_ai/, server/, tools/, train_curriculum.py, docs/plan*.json) → base64 tar.
  3. Kaggle-скрипт на старте: распаковывает бандл, строит корпус (tools/make_corpus.py),
     параллель (tools/make_parallel.py, или использует data/parallel.json если он есть в бандле —
     он всего 3 МБ и укладывается в лимит ~1 МБ? если не влезает, качаем на Kaggle),
     reasoning/school_math/math/context_tasks/transfer/python_tasks генераторами, затем
     train_curriculum.py --init /kaggle/input/mini-ai-init/model.json --out /kaggle/working/out.
  4. poll (kernels status), затем download (kernels output) — model_final.json, exams.json,
     log.txt, последние чекпойнты стадии 'selfplay' (ckpt/).
  5. Ничего тяжёлого не выполняется локально — здесь только сборка/отправка/скачивание.

--smoke: собирает ТОТ ЖЕ пайплайн, но с крошечным планом (CPU, малый бюджет времени) —
используется для сквозной проверки бандлинга/скачивания без реального обучения.
"""
import argparse
import base64
import io
import json
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KAGGLE = str(ROOT / ".venv/bin/kaggle")
USER = "mamazanovfarhod016"
SLUG = "mini-ai-curriculum"
DATASET = "mini-ai-init"

RUNNER = r'''
import base64, io, os, subprocess, sys, tarfile, time
os.makedirs("/kaggle/working/w", exist_ok=True)
os.chdir("/kaggle/working/w")
tarfile.open(fileobj=io.BytesIO(base64.b64decode(BUNDLE)), mode="r:gz").extractall(".")
print("CPU:", os.cpu_count(), flush=True)
print("kaggle input:", os.listdir("/kaggle/input") if os.path.exists("/kaggle/input") else "НЕТ /kaggle/input", flush=True)

def run(cmd):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print("WARN: команда завершилась с ошибкой", r.returncode, flush=True)
    return r.returncode

if BUILD_DATA:
    os.makedirs("data/corpus", exist_ok=True)
    run([sys.executable, "tools/make_corpus.py"])
    run([sys.executable, "tools/make_school_math.py"])
    run([sys.executable, "tools/make_math.py"])
    run([sys.executable, "tools/make_reasoning.py"])
    run([sys.executable, "tools/make_context_tasks.py"])
    run([sys.executable, "tools/make_transfer.py"])
    if not os.path.exists("data/parallel.json"):
        run([sys.executable, "tools/make_parallel.py"])
    run([sys.executable, "tools/make_python_tasks.py"])
    run([sys.executable, "tools/make_intents_pairs.py"])
else:
    # smoke-режим: не тратим время на полный корпус/генераторы — крошечный текст для проверки
    os.makedirs("data/corpus", exist_ok=True)
    if not os.path.exists("data/corpus/ru.txt"):
        with open("data/corpus/ru.txt", "w", encoding="utf-8") as f:
            f.write("привет мир\nкак дела у тебя\nсолнце светит ярко\n")

t = time.time()
init_path = INIT_PATH
if not os.path.exists(init_path):
    # fallback: искать model.json в любом примонтированном датасете (имя датасета могло
    # не совпасть, например из-за задержки версии)
    for root, dirs, fs in os.walk("/kaggle/input"):
        if "model.json" in fs:
            init_path = os.path.join(root, "model.json")
            print("INIT_PATH не найден, использую", init_path, flush=True)
            break
cmd = [sys.executable, "train_curriculum.py", "--init", init_path, "--out", "/kaggle/working/out",
       "--ckpt_minutes", str(CKPT_MINUTES)]
if PLAN_JSON:
    with open("plan.json", "w") as f:
        f.write(PLAN_JSON)
    cmd += ["--plan", "plan.json"]
if STEPS_OVERRIDE:
    cmd += ["--steps_total_override", str(STEPS_OVERRIDE)]
r = subprocess.run(cmd)
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


def upload_init(model_path, log=print, wait_s=600):
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
            log(f"Kaggle: модель загружена ({Path(model_path).stat().st_size // 1024} КБ)")
            return
        time.sleep(15)
    raise TimeoutError("Kaggle: набор данных с моделью не готов")


def run(init, out, plan=None, gpu=True, build_data=True, ckpt_minutes=15, steps_override=0,
        poll=60, max_wait_h=11.5, log=print):
    upload_init(init, log=log)
    files = [
        (ROOT / "mini_ai", "mini_ai"), (ROOT / "server", "server"), (ROOT / "tools", "tools"),
        (ROOT / "train_curriculum.py", "train_curriculum.py"),
        # make_intents_pairs.py импортирует train_transformer и читает data/intents.json
        (ROOT / "train_transformer.py", "train_transformer.py"),
        (ROOT / "data" / "intents.json", "data/intents.json"),
        # make_parallel.py на Kaggle падает без кэша корпуса → берём готовый parallel.json
        (ROOT / "data" / "parallel.json", "data/parallel.json"),
    ]
    plan_json = Path(plan).read_text(encoding="utf-8") if plan else ""
    b64 = bundle(files)
    script = (f"BUNDLE = {b64!r}\nINIT_PATH = {f'/kaggle/input/{DATASET}/model.json'!r}\n"
              f"BUILD_DATA = {build_data!r}\nPLAN_JSON = {plan_json!r}\nCKPT_MINUTES = {ckpt_minutes!r}\n"
              f"STEPS_OVERRIDE = {steps_override!r}\n") + RUNNER
    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "run.py").write_text(script, encoding="utf-8")
        meta = {"id": f"{USER}/{SLUG}", "title": SLUG, "code_file": "run.py", "language": "python",
                "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_internet": True,
                "dataset_sources": [f"{USER}/{DATASET}"], "competition_sources": [], "kernel_sources": []}
        if gpu:
            meta["machine_shape"] = "NvidiaTeslaT4"
        (Path(d) / "kernel-metadata.json").write_text(json.dumps(meta), encoding="utf-8")
        log(f"Kaggle: отправляю curriculum ({len(b64)//1024} КБ)")
        res = kaggle("kernels", "push", "-p", d, timeout=300)
    if "successfully" not in res:
        raise RuntimeError(f"push не удался: {res[-500:]}")
    t0 = time.time()
    while True:
        time.sleep(poll)
        st = kaggle("kernels", "status", f"{USER}/{SLUG}")
        log(f"Kaggle status: {st}")
        if "COMPLETE" in st:
            break
        if "ERROR" in st or "CANCEL" in st:
            raise RuntimeError(f"Kaggle: {st}")
        if time.time() - t0 > max_wait_h * 3600:
            raise TimeoutError("Kaggle: слишком долго")
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{USER}/{SLUG}", "-p", str(out), timeout=900)
    log(f"Kaggle: готово за {(time.time()-t0)/60:.0f} мин → {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plan", default=None)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--no-build-data", dest="build_data", action="store_false")
    ap.add_argument("--ckpt_minutes", type=float, default=15)
    ap.add_argument("--steps_override", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="крошечный smoke-прогон (CPU, малый план)")
    a = ap.parse_args()
    if a.smoke:
        # лёгкий smoke: НЕ строим полный корпус/генераторы на Kaggle (это долго для нашего
        # предварительного прогона) — используем крошечный встроенный текст, который
        # раннер сам пишет в data/corpus/ru.txt перед запуском train_curriculum.py.
        tiny_plan = {
            "init_model": a.init, "target_vocab": 1210,
            "arch": {"d": 192, "n_layer": 6, "n_head": 8, "block": 224, "pos": "rope"},
            "time_budget_hours": 0.01,
            "stages": [
                {"name": "read", "max_len": 16, "token_budget": 64, "lr": 1e-3, "warmup": 2,
                 "time_frac": 1.0,
                 "sources": [{"type": "lm", "weight": 1.0, "files": ["data/corpus/ru.txt"]}]},
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(tiny_plan, f, ensure_ascii=False)
            a.plan = f.name
        a.build_data = False
        a.cpu = True
        a.steps_override = 3
    run(a.init, a.out, a.plan, gpu=not a.cpu, build_data=a.build_data, ckpt_minutes=a.ckpt_minutes,
        steps_override=a.steps_override)


if __name__ == "__main__":
    main()
