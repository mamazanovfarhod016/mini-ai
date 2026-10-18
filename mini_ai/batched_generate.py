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


def apply_anti_repeat_(last_logits, gen_ids_list, repetition_penalty=1.3, no_repeat_ngram=3):
    """Мутирует last_logits (B, V) IN-PLACE: штраф за уже сгенерированные токены (repetition
    penalty — делим логит на repetition_penalty, если >0) и ПОЛНЫЙ запрет токенов, которые бы
    повторили n-грамм (no_repeat_ngram), уже встречавшийся в gen_ids_list[i]. Та же логика (тот
    же порядок: сначала penalty, потом запрет) должна быть и в app/www/transformer.js —
    см. applyAntiRepeat там (паритет Python↔JS)."""
    B = last_logits.shape[0]
    n = no_repeat_ngram
    for i in range(B):
        ids = gen_ids_list[i]
        if repetition_penalty and repetition_penalty != 1.0 and ids:
            seen = set(ids)
            for t in seen:
                v = last_logits[i, t]
                last_logits[i, t] = v / repetition_penalty if v > 0 else v * repetition_penalty
        if n and len(ids) >= n - 1:
            prefix = tuple(ids[-(n - 1):]) if n > 1 else ()
            banned = set()
            for j in range(len(ids) - n + 1):
                if tuple(ids[j:j + n - 1]) == prefix:
                    banned.add(ids[j + n - 1])
            for t in banned:
                last_logits[i, t] = -1e9
    return last_logits


@torch.no_grad()
def batched_generate(engine, prompts, max_new, temperature=0.0, seed=0, pad_id=PAD, eos_id=EOS,
                      repetition_penalty=1.0, no_repeat_ngram=0):
    """prompts: список списков id (разной длины). Возвращает список списков id (без prompt,
    обрезано по <eos> если встретился). Паддинг слева не нужен — паддинг справа + attention
    через -inf позиции мы не делаем: вместо этого каждый prompt обрабатывается с собственной
    длиной префикса через маску (простая реализация: батч с паддингом PAD справа, маска
    'реальная длина' на каждом шаге, позиции PAD в логитах игнорируются).

    repetition_penalty>1.0 и/или no_repeat_ngram>0 включают анти-повтор (см.
    apply_anti_repeat_) — используется и в экзамене диалога, и в self-play, идентично
    JS-реализации в приложении (app/www/transformer.js, applyAntiRepeat)."""
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
        if repetition_penalty != 1.0 or no_repeat_ngram:
            apply_anti_repeat_(last_logits, out, repetition_penalty, no_repeat_ngram)
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


def generate_dialog_reply(engine, tok, history_turns, last_user_text, past_bot_answers, max_new=60,
                           repetition_penalty=1.3, no_repeat_ngram=3):
    """Один диалоговый ответ с анти-повтором (используется приложением-эквивалентом/тестами
    и экзаменом): штраф+no-repeat-ngram ВНУТРИ ответа (batched_generate), а если результат
    всё равно совпадает буква-в-букву с любым прошлым ответом бота в этом диалоге
    (past_bot_answers) — одна попытка пересэмплировать с temperature=0.7 (seed=1); если и она
    совпала, отдаём как есть (не блокируем ответ бесконечно)."""
    from mini_ai.curriculum import decode_answer, encode_prompt
    from mini_ai.dialog import encode_dialog_prompt_ids
    from mini_ai.transformer import BOS, SEP

    def enc(text):
        return encode_prompt(tok, text, raw=True)

    prompt_ids = encode_dialog_prompt_ids(tok, history_turns, last_user_text, enc)
    prompt = [BOS] + prompt_ids + [SEP]
    past_norm = {str(a).strip() for a in past_bot_answers}

    for attempt, temp in enumerate((0.0, 0.7)):
        out = batched_generate(engine, [prompt], max_new, temperature=temp, seed=attempt,
                                repetition_penalty=repetition_penalty, no_repeat_ngram=no_repeat_ngram)
        text = decode_answer(tok, out[0])
        if text.strip() not in past_norm:
            return text
    return text


def kv_greedy_reference(p_np, cfg, prompt_ids, max_new):
    """Numpy KVModel greedy — эталон для теста соответствия."""
    from mini_ai.transformer import KVModel
    kv = KVModel(p_np, cfg)
    out, _ = kv.generate(prompt_ids, max_new=max_new, temperature=0.0)
    return out
