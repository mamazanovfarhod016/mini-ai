"""Собирает colab/train.ipynb (запуск: python3 tools/make_colab.py). Блокнот для Google Colab:
одна кнопка «Запустить всё» → берёт код с GitHub → учит на видеокарте Colab → отправляет модель на сервер
(server/app.py /api/colab/upload, карантин; в приложение — только после экзамена)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

INTRO = """# Mini AI — обучение в Google Colab

1. Сверху: **Среда выполнения → Сменить среду → T4 GPU** (если ещё не стоит).
2. **Среда выполнения → Выполнить все** (или Ctrl+F9).
3. Можно закрыть вкладку не сразу: обучение ~1–2 часа. Готовая модель сама уйдёт на сервер Mini AI.

Код и модель — Фарходa (github.com/mamazanovfarhod016/mini-ai), обучение с нуля, чужих моделей нет."""

CODE = r'''# Mini AI: всё в одной ячейке
import os, subprocess, sys, time, json, urllib.request
STEPS = 6000          # сколько шагов учить
BRANCH = "feat/teach"
REPO = "https://github.com/mamazanovfarhod016/mini-ai.git"

def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)

import torch
gpu = torch.cuda.is_available()
print("Видеокарта:", torch.cuda.get_device_name(0) if gpu else "НЕТ — включи: Среда выполнения → Сменить среду → T4 GPU")
assert gpu, "Нужна видеокарта: Среда выполнения → Сменить среду выполнения → T4 GPU, потом снова «Выполнить все»"

os.chdir("/content")
if not os.path.exists("mini-ai"):
    sh(f"git clone -q --depth 1 -b {BRANCH} {REPO}")
os.chdir("/content/mini-ai")
sh("git pull -q")
SERVER = json.load(urllib.request.urlopen("https://raw.githubusercontent.com/mamazanovfarhod016/mini-ai/live/server.json?t=%d" % time.time(), timeout=30))["url"].rstrip("/")
print("Сервер:", SERVER)

t = time.time()
cmd = [sys.executable, "-u", "train_transformer.py", "--init", "model_transformer_school.json",
       "--heads", "8", "--block", "115", "--engine", "torch", "--math", "data/school_math.json",
       "--math_frac", "0.4", "--lr", "4e-4", "--warmup", "300", "--steps", str(STEPS),
       "--eval_every", "1000", "--seed", "1", "--out", "/content/model.json", "--no-app",
       "--log", "/content/train_log.txt", "--ckpt_dir", "/content/ckpt"]
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
import re
for line in p.stdout:
    m = re.search(r"шаг\s+(\d+)/", line)
    if not m or int(m.group(1)) % 200 == 0:
        print(line, end="", flush=True)
p.wait()
print(f"Обучение: {(time.time()-t)/60:.0f} мин, код выхода {p.returncode}")

def upload(path, kind):
    data = open(path, "rb").read()
    req = urllib.request.Request(f"{SERVER}/api/colab/upload?kind={kind}", data=data, method="POST",
                                 headers={"Content-Type": "application/octet-stream"})
    for i in range(5):
        try:
            print(kind, "→ сервер:", urllib.request.urlopen(req, timeout=300).read().decode()); return True
        except Exception as e:
            print("не ушло, повтор:", e); time.sleep(20)
    return False

if os.path.exists("/content/train_log.txt"):
    upload("/content/train_log.txt", "log")
if p.returncode == 0 and os.path.exists("/content/model.json"):
    try:  # адрес мог смениться за время обучения
        SERVER = json.load(urllib.request.urlopen("https://raw.githubusercontent.com/mamazanovfarhod016/mini-ai/live/server.json?t=%d" % time.time(), timeout=30))["url"].rstrip("/")
    except Exception:
        pass
    if not upload("/content/model.json", "model"):
        from google.colab import files
        files.download("/content/model.json")
        print("Сервер недоступен — модель скачана на телефон, перешли файл model.json в Telegram.")
    print("ГОТОВО ✅ Можно закрыть.")
else:
    print("Ошибка обучения — лог уже на сервере, разберусь.")
'''


def main():
    nb = {
        "nbformat": 4, "nbformat_minor": 0,
        "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                     "kernelspec": {"name": "python3", "display_name": "Python 3"}},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": INTRO},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": CODE},
        ],
    }
    out = ROOT / "colab/train.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("записан", out)


if __name__ == "__main__":
    main()
