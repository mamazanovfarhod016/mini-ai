"""
Батчированная генерация на PyTorch (greedy/temperature) для экзаменов и self-play.

Эталон корректности — mini_ai.transformer.KVModel.generate(temperature=0) (greedy,
посимвольный KV-cache). tests/test_curriculum.py проверяет точное совпадение на
крошечной модели между batched_generate(temperature=0) и KVModel greedy.

Реализация без KV-cache (пересчёт всего префикса на каждом шаге) — дороже по FLOPs,
но код прост и корректен, а на экзаменах/self-play генерации короткие (<=400 токенов)
и батч маленький — это приемлемо даже на T4.
"""
import numpy as np
import torch

from mini_ai.torch_engine import forward as torch_forward
from mini_ai.transformer import EOS, PAD


@torch.no_grad()
def batched_generate(engine, prompts, max_new, temperature=0.0, seed=0, pad_id=PAD, eos_id=EOS):
    """prompts: список списков id (разной длины). Возвращает список списков id (без prompt,
    обрезано по <eos> если встретился). Паддинг слева не нужен — паддинг справа + attention
    через -inf позиции мы не делаем: вместо этого каждый prompt обрабатывается с собственной
    длиной префикса через маску (простая реализация: батч с паддингом PAD справа, маска
    'реальная длина' на каждом шаге, позиции PAD в логитах игнорируются)."""
    dev = engine.dev
    B = len(prompts)
    lens = [len(p) for p in prompts]
    cur = [list(p) for p in prompts]
    done = [False] * B
    out = [[] for _ in range(B)]
    gen = torch.Generator(device=dev).manual_seed(seed) if temperature > 0 else None
    max_len_cfg = engine.cfg.get("block", 10000)
    for step in range(max_new):
        T = max(len(c) for c in cur)
        if T > max_len_cfg:
            break
        idx = torch.full((B, T), pad_id, dtype=torch.long, device=dev)
        real_pos = torch.zeros(B, dtype=torch.long, device=dev)
        for i, c in enumerate(cur):
            idx[i, :len(c)] = torch.tensor(c, dtype=torch.long, device=dev)
            real_pos[i] = len(c) - 1
        logits = torch_forward(engine.p, engine.cfg, idx, drop=0.0, gen=None)
        last_logits = logits[torch.arange(B, device=dev), real_pos]  # (B, V)
        if temperature <= 0:
            nxt = torch.argmax(last_logits, dim=-1)
        else:
            probs = torch.softmax(last_logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=gen).squeeze(-1)
        nxt = nxt.cpu().numpy()
        any_alive = False
        for i in range(B):
            if done[i]:
                continue
            t = int(nxt[i])
            if t == eos_id:
                done[i] = True
                continue
            out[i].append(t)
            cur[i].append(t)
            any_alive = True
        if not any_alive:
            break
    return out


def kv_greedy_reference(p_np, cfg, prompt_ids, max_new):
    """Numpy KVModel greedy — эталон для теста соответствия."""
    from mini_ai.transformer import KVModel
    kv = KVModel(p_np, cfg)
    out, _ = kv.generate(prompt_ids, max_new=max_new, temperature=0.0)
    return out
