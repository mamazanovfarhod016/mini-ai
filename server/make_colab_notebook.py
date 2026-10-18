"""Собрать Colab-блокнот colab/mini_ai_curriculum.ipynb (тот же путь, что на Kaggle).

    python -m server.make_colab_notebook

Блокнот: GPU T4 → клон репо (ветка BRANCH) → модель-init с Google Drive (MyDrive/mini-ai/model.json)
→ сборка данных → train_curriculum.py → результат и чекпойнты сразу на Drive (MyDrive/mini-ai/out),
поэтому обрыв сессии Colab не теряет работу: перезапуск берёт последний чекпойнт.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BRANCH = "feat/dialog-code"
PLAN = "plans_night.json"


def cell(src, kind="code"):
    c = {"cell_type": kind, "metadata": {}, "source": src.strip("\n").splitlines(keepends=True)}
    if kind == "code":
        c.update({"execution_count": None, "outputs": []})
    return c


CELLS = [
    cell(f"""# Mini AI — обучение в Google Colab (автор: Фарход)
Перед запуском: **Среда выполнения → Сменить среду → Аппаратный ускоритель: T4 GPU**.
Потом **Среда выполнения → Выполнить все**. Вкладку держать открытой.

Модель-init: файл `model.json` в Google Drive, папка `mini-ai` (MyDrive/mini-ai/model.json).
Результат: MyDrive/mini-ai/out (model_final.json, exams.json, log.txt, чекпойнты).""", "markdown"),
    cell("""import torch, subprocess
assert torch.cuda.is_available(), "Нет GPU: Среда выполнения → Сменить среду → T4 GPU"
print("GPU:", torch.cuda.get_device_name(0))"""),
    cell("""from google.colab import drive
drive.mount("/content/drive")
import os
D = "/content/drive/MyDrive/mini-ai"
os.makedirs(D + "/out", exist_ok=True)
assert os.path.exists(D + "/model.json"), "Положи model.json в Google Drive → папка mini-ai"
print("init:", os.path.getsize(D + "/model.json") // 1024, "КБ")"""),
    cell(f"""%cd /content
!rm -rf mini-ai && git clone -q --depth 1 -b {BRANCH} https://github.com/mamazanovfarhod016/mini-ai.git
%cd /content/mini-ai
!git log --oneline -1"""),
    cell("""import subprocess, sys
def run(cmd):
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True)
    if r.returncode:
        print("WARN: ошибка", r.returncode)
os.makedirs("data/corpus", exist_ok=True)
for t in ["make_corpus", "make_school_math", "make_math", "make_reasoning", "make_context_tasks",
          "make_transfer", "make_python_tasks", "make_intents_pairs"]:
    run(f"{sys.executable} tools/{t}.py")
if not os.path.exists("data/parallel.json"):
    run(f"{sys.executable} tools/make_parallel.py")"""),
    cell(f"""# Обучение. Чекпойнты пишутся прямо на Drive — после обрыва просто «Выполнить все» ещё раз.
!python train_curriculum.py --init {{D}}/model.json --out {{D}}/out --plan {PLAN} --ckpt_minutes 10 2>&1 | tee -a {{D}}/out/colab_log.txt"""),
    cell("""!ls -la {D}/out | head -20
import json, os
p = D + "/out/exams.json"
if os.path.exists(p):
    print(json.dumps(json.load(open(p)), ensure_ascii=False, indent=1)[:3000])"""),
]


def main():
    nb = {"cells": CELLS, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                       "kernelspec": {"name": "python3", "display_name": "Python 3"}},
          "nbformat": 4, "nbformat_minor": 0}
    out = ROOT / "colab" / "mini_ai_curriculum.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
