"""Эксперимент 1: умеет ли сама сеть (без дообучения) различать новые факты своими hidden-state?
Условие (c): факт записан в память одним проходом; вопрос → ключ → ближайшая запись → ответ.
Перебираем: слой (из какого слоя брать вектор), пул (mean / last), память (список / Hebb-матрица),
отбеливание (нет / да), N фактов (10 / 30 / 100 / 300) — кривая интерференции.
"""
import json, sys, time, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.transformer import load_model
from mini_ai.oneshot_bench import make_facts
from mini_ai.memory_core import Encoder, AssocMemory, HebbMemory, Whitener

MODEL = "/home/hermes/Developer/mini-ai/model_transformer_school.json"
OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
p, cfg, tok, meta = load_model(MODEL)
NS = [10, 30, 100, 300]
facts_all = make_facts(max(NS), seed=7)
# фоновые фразы для отбеливания — другие факты (seed иной), модель их не «знает», это просто статистика векторов
bg_facts = make_facts(200, seed=99)

results = []
for layer in [1, 3, 5]:
    enc = Encoder(p, cfg, tok, layer)
    t0 = time.time()
    kf = {pool: np.stack([enc.key(f["fact"], pool) for f in facts_all]) for pool in ("mean", "last")}
    kq = {pool: np.stack([enc.key(f["q"], pool) for f in facts_all]) for pool in ("mean", "last")}
    bg = {pool: np.stack([enc.key(f["fact"], pool) for f in bg_facts]) for pool in ("mean", "last")}
    enc_sec = time.time() - t0
    for pool in ("mean", "last"):
        for white in (False, True):
            W = Whitener(bg[pool]) if white else (lambda x: x)
            KF, KQ = W(kf[pool]) if white else kf[pool], W(kq[pool]) if white else kq[pool]
            for mem_kind in ("list", "hebb"):
                for N in NS:
                    mem = AssocMemory() if mem_kind == "list" else HebbMemory(KF.shape[1])
                    tw = time.time()
                    for i in range(N):
                        mem.write(KF[i], i)
                    tw = time.time() - tw
                    ok = sum(mem.read(KQ[i])[0][0] == i for i in range(N))
                    acc = ok / N
                    results.append({"layer": layer, "pool": pool, "white": white, "mem": mem_kind, "N": N,
                                    "acc": acc, "write_sec": round(tw, 4)})
                    print(f"L{layer} {pool:4} white={int(white)} {mem_kind:4} N={N:3}: {acc*100:5.1f}%", flush=True)
    print(f"  (encode layer {layer}: {enc_sec:.0f} s)", flush=True)
    json.dump(results, open(f"{OUT}/exp1_t1.json", "w"), indent=1)
