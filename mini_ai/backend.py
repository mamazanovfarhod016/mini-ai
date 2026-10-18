"""
Где считать: на процессоре (numpy) или на видеокарте (CuPy).

CuPy — это тот же numpy, только массивы живут в памяти видеокарты (NVIDIA, CUDA).
Функции называются так же: cp.exp, cp.tanh, a @ b ... Поэтому формулы в transformer.py
написаны один раз через `xp` — а `xp` это либо numpy, либо cupy.

Включить видеокарту:  MINI_AI_GPU=1 python train_transformer.py ...
                  или  train_transformer.py --gpu
Без CuPy / без видеокарты всё работает как раньше, на numpy.
Никакого PyTorch: backprop, Adam и всё остальное — по-прежнему наши формулы.
"""
import os

import numpy as np

xp = np
GPU = False


def use_gpu(on=True):
    """Переключить вычисления на видеокарту. Возвращает True, если получилось."""
    global xp, GPU
    if not on:
        xp, GPU = np, False
        return False
    try:
        import cupy as cp
        cp.zeros(1).sum()            # проверка: видеокарта реально отвечает
        xp, GPU = cp, True
    except Exception as e:          # нет CuPy / нет CUDA → остаёмся на numpy
        print(f"Видеокарта недоступна ({type(e).__name__}: {e}) — считаю на процессоре")
        xp, GPU = np, False
    return GPU


def to_np(a):
    """Массив с видеокарты → обычный numpy (для сохранения модели, экзамена, логов)."""
    if GPU and not isinstance(a, np.ndarray):
        return a.get()
    return np.asarray(a)


def to_xp(a):
    """numpy → туда, где считаем (на видеокарту, если она включена)."""
    return xp.asarray(a) if GPU else a


def xp_of(a):
    """Модуль (numpy/cupy), которому принадлежит массив a."""
    return xp if GPU and not isinstance(a, np.ndarray) else np


def scatter_add(target, idx, values):
    """target[idx[i]] += values[i] с повторами (нужно для градиента эмбеддингов).
    numpy: np.add.at;  cupy: cupyx.scatter_add — то же самое, но на видеокарте."""
    if GPU:
        import cupyx
        cupyx.scatter_add(target, idx, values)
    else:
        np.add.at(target, idx, values)


_dev_rngs = {}


def device_rng(np_rng):
    """Генератор случайных чисел на видеокарте, засеянный из нашего numpy-генератора (один раз)."""
    g = _dev_rngs.get(id(np_rng))
    if g is None:
        g = _dev_rngs[id(np_rng)] = xp.random.default_rng(int(np_rng.integers(2**31)))
    return g


def sync():
    """Видеокарта считает асинхронно; для честного замера времени ждём окончания."""
    if GPU:
        xp.cuda.Device().synchronize()


if os.environ.get("MINI_AI_GPU") == "1":
    use_gpu(True)
