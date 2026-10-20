"""Smoke-проверка Modal: видна ли видеокарта и сколько стоит секунда (T4, несколько секунд)."""
import modal

app = modal.App("mini-ai-gpu-check")
image = modal.Image.debian_slim().pip_install("torch")


@app.function(gpu="T4", image=image, timeout=300)
def check():
    import time

    import torch

    name = torch.cuda.get_device_name(0)
    a = torch.randn(4096, 4096, device="cuda")
    torch.cuda.synchronize()
    t = time.time()
    for _ in range(20):
        a @ a
    torch.cuda.synchronize()
    return name, round(time.time() - t, 3)


@app.local_entrypoint()
def main():
    print("GPU:", check.remote())
