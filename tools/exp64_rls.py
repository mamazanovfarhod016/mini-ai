"""exp64 — РЕКУРРЕНТНАЯ ФОРМУЛА вместо решения системы: RLS (recursive least squares) с человеческим приоритетом.

Сейчас solver.py держит суммы A (F×F) и B и решает W=(A+λI)⁻¹B раз в 20 меток — O(F³) ≈ 1000³ на решение.
Новая формула: держим P = (A+λI)⁻¹ напрямую и обновляем её на КАЖДОЙ метке за O(F²) (Шерман–Моррисон):
    k = P φ / (1/w + φᵀ P φ);   W += k (y − φᵀW);   P −= k φᵀ P
Каждая метка меняет W мгновенно, без пересчёта, без забывания (без λ-forgetting). Вес метки w: человек 1.0,
самопроверка 0.3 — вес входит в формулу точно, не приближённо.
Проверка (до запуска): (1) W_RLS == W_solve с точностью 1e-3 на тех же метках (математическая тождественность);
(2) время на 1000 меток F=1000: RLS vs решать каждые 20; (3) та же GROWTH на growth_eval_light (не хуже, won/lost).
"""
import sys, time, os
sys.path.insert(0, ".")
import numpy as np

F, N, LAM = 1000, 1500, 2.0
rng = np.random.default_rng(64)
Phi = rng.normal(0, 1, (N, F)).astype(np.float64); Phi[:, -1] = 1.0
y = (rng.random(N) < 0.3).astype(np.float64); w = np.where(rng.random(N) < 0.5, 1.0, 0.3)

# --- batch solve (как сейчас) ---
t0 = time.time(); A = np.zeros((F, F)); B = np.zeros(F); n_solve = 0
for i in range(N):
    A += w[i] * np.outer(Phi[i], Phi[i]); B += w[i] * Phi[i] * y[i]
    if (i + 1) % 20 == 0: W_b = np.linalg.solve(A + LAM * np.eye(F), B); n_solve += 1
W_b = np.linalg.solve(A + LAM * np.eye(F), B); t_batch = time.time() - t0

# --- RLS ---
t0 = time.time(); P = np.eye(F) / LAM; W_r = np.zeros(F)
for i in range(N):
    phi = Phi[i]; Pphi = P @ phi; denom = 1.0 / w[i] + phi @ Pphi; k = Pphi / denom
    W_r += k * (y[i] - phi @ W_r); P -= np.outer(k, Pphi)
t_rls = time.time() - t0
print(f"max|W_rls − W_solve| = {np.abs(W_r - W_b).max():.2e}   (должно быть ~1e-10)")
print(f"batch: {t_batch:.1f}s ({n_solve} решений)   RLS: {t_rls:.1f}s  → RLS быстрее в {t_batch / t_rls:.1f}×, и W обновляется после КАЖДОЙ метки")
# per-label latency
t0 = time.time(); phi = Phi[0]; Pphi = P @ phi; k = Pphi / (1 + phi @ Pphi); W_r += k * (y[0] - phi @ W_r); P -= np.outer(k, Pphi)
print(f"одна метка RLS: {(time.time() - t0) * 1000:.1f} ms")
