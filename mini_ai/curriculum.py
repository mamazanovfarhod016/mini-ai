"""
Мультистадийное обучение (curriculum) Mini AI: read → reason → context → self-play.

Общая идея: план (plan) — список стадий (STAGES), каждая стадия — словарь:
    {
      "name": "read",
      "steps": 4000, "lr": 2e-3, "warmup": 200,
      "max_len": 512,           # длина окна в токенах
      "token_budget": 16384,    # сколько токенов суммарно в батче (динамический batch size)
      "sources": [
          {"type": "lm", "files": ["data/corpus/ru.txt", ...], "weight": 0.85},
          {"type": "qa", "file": "data/intents_pairs.json", "weight": 0.15, "raw": False},
      ],
      "selfplay": {...}   # только для type == "selfplay" источника
    }

Движок обучения: mini_ai.torch_engine.TorchEngine (PyTorch autograd + наш AdamW/lr/clip).
Чекпойнты: export_model (mini_ai.transformer) в JSON, сохраняются каждые N минут и в конце стадии.
Resume: если ckpt_dir/state.json существует — обучение продолжается с того места.

ВАЖНО: все источники объединяются по весам (weight) при формировании каждого батча —
«смешанные» батчи, т.к. это упрощает реализацию и даёт одинаковый бакетинг длины.
"""
import json
import math
import time
from pathlib import Path

import numpy as np

from mini_ai.transformer import (
    BOS, EOS, PAD, SEP, clean_question, export_model, grow_params, init_params,
    load_model, lr_schedule, n_params,
)

ROOT = Path(__file__).resolve().parent.parent


# ======================================================================================
# 0. Единый помощник кодирования текста — чтобы потом легко переключиться на raw=True
#    (lossless-токенизация, которую добавляет параллельная ветка feat/bpe-code).
# ======================================================================================
def encode_prompt(tok, text, raw=False):
    """Токенизация одного куска текста. raw=True — пропустить clean_question (для кода/контекста,
    где lossless важнее нормализации). Сейчас raw просто кодирует как есть; когда в bpe.py
    появится prompt_ids(tok, text, raw=True) с честным lossless-путём, здесь достаточно
    будет позвать его вместо tok.encode — остальной код curriculum.py не меняется."""
    if raw:
        if hasattr(tok, "encode_raw"):
            return tok.encode_raw(text)
        return tok.encode(text)
    return tok.encode(clean_question(text))


# ======================================================================================
# 1. Источники данных → примитивные "items", которые потом режутся на окна/примеры
# ======================================================================================
def load_lm_lines(files):
    """Список файлов (по одному предложению на строку) → список непустых строк."""
    lines = []
    for f in files:
        p = Path(f)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def pack_lm_windows(tok, lines, max_len, rng, raw=False):
    """Пакуем предложения в окна длиной max_len токенов, разделяя <eos>.
    loss считаем по ВСЕМ токенам окна (mask = True везде, кроме стартового <bos>-эффекта,
    который здесь не нужен — окно это просто поток токенов)."""
    ids_all = []
    for line in lines:
        ids_all.extend(encode_prompt(tok, line, raw))
        ids_all.append(EOS)
    if not ids_all:
        return []
    windows = []
    i = 0
    while i < len(ids_all):
        w = ids_all[i:i + max_len]
        if len(w) >= 2:
            windows.append(w)
        i += max_len
    rng.shuffle(windows)
    return windows


def lm_example_to_seq(window):
    """window — список id (длина T). Возвращает (seq, targets, loss_mask) где seq=window[:-1],
    targets=window[1:], loss_mask=True везде (каждый токен предсказывает следующий)."""
    seq = window[:-1]
    tgt = window[1:]
    mask = [True] * len(seq)
    return seq, tgt, mask


def qa_example_to_seq(tok, q, a, max_q, raw=False):
    """<bos> q <sep> a <eos>, loss только на ответе (как в train_transformer.py)."""
    q_ids = encode_prompt(tok, q, raw)[:max_q]
    a_ids = encode_prompt(tok, a, raw)
    seq = [BOS] + q_ids + [SEP] + a_ids + [EOS]
    ans_start = 1 + len(q_ids) + 1
    full_seq = seq[:-1]
    tgt = seq[1:]
    mask = [False] * len(full_seq)
    for t in range(ans_start - 1, len(full_seq)):
        mask[t] = True
    return full_seq, tgt, mask


def context_example_to_seq(tok, context, q, a, max_len, raw=False):
    """<bos> context \n q <sep> a <eos>, loss только на ответе."""
    ctx_ids = encode_prompt(tok, context, raw=True)
    q_ids = encode_prompt(tok, "\n" + q, raw)
    a_ids = encode_prompt(tok, a, raw)
    seq = [BOS] + ctx_ids + q_ids + [SEP] + a_ids + [EOS]
    ans_start = 1 + len(ctx_ids) + len(q_ids) + 1
    if len(seq) > max_len:
        # режем контекст СЛЕВА (вопрос+ответ в конце — самые важные токены)
        overflow = len(seq) - max_len
        seq = [BOS] + ctx_ids[overflow:] + q_ids + [SEP] + a_ids + [EOS]
        ans_start = max(1, ans_start - overflow)
    full_seq = seq[:-1]
    tgt = seq[1:]
    mask = [False] * len(full_seq)
    for t in range(ans_start - 1, len(full_seq)):
        mask[t] = True
    return full_seq, tgt, mask


# ======================================================================================
# 2. Батчи: собираем из разных источников по весам, динамический batch size по token_budget,
#    бакетинг по длине (сортируем пул примеров перед нарезкой на батчи той же "эпохи").
# ======================================================================================
def pad_batch(seqs, tgts, masks):
    T = max(len(s) for s in seqs)
    B = len(seqs)
    idx = np.full((B, T), PAD, dtype=np.int64)
    targets = np.full((B, T), PAD, dtype=np.int64)
    mask = np.zeros((B, T), dtype=bool)
    for i, (s, t, m) in enumerate(zip(seqs, tgts, masks)):
        L = len(s)
        idx[i, :L] = s
        targets[i, :L] = t
        mask[i, :L] = m
    return idx, targets, mask


class SourcePool:
    """Держит заранее сгенерированные примеры (seq, tgt, mask) одного источника и умеет
    отдавать случайный пример (с повторным проходом/перемешиванием по исчерпании)."""

    def __init__(self, examples, rng):
        self.examples = examples
        self.rng = rng
        self.i = 0
        self._shuffle()

    def _shuffle(self):
        self.order = list(range(len(self.examples)))
        self.rng.shuffle(self.order)
        self.i = 0

    def sample(self):
        if not self.examples:
            return None
        if self.i >= len(self.order):
            self._shuffle()
        ex = self.examples[self.order[self.i]]
        self.i += 1
        return ex

    def add(self, examples):
        self.examples.extend(examples)


def build_dynamic_batch(pools, weights, token_budget, max_len, rng, min_batch=2, max_batch=256):
    """Набираем примеры по весам (weights суммируются к 1), пока не превысили token_budget
    по сумме длин (с учётом паддинга до max длины в батче — бакетинг по длине экономит память,
    поэтому сортируем кандидатов по длине перед финальной нарезкой)."""
    names = list(pools.keys())
    w = np.array([weights.get(n, 0.0) for n in names], dtype=np.float64)
    if w.sum() <= 0:
        w = np.ones(len(names))
    w = w / w.sum()
    candidates = []
    # соберём кандидатов (с запасом) — потом отсортируем по длине для компактного паддинга
    tries = 0
    while len(candidates) < max_batch and tries < max_batch * 4:
        tries += 1
        name = rng.choice(names, p=w)
        ex = pools[name].sample()
        if ex is None:
            continue
        seq, tgt, mask = ex
        if len(seq) > max_len:
            seq, tgt, mask = seq[:max_len], tgt[:max_len], mask[:max_len]
        candidates.append((seq, tgt, mask))
        if len(candidates) >= min_batch:
            total = sum(len(c[0]) for c in candidates)
            # приблизительно: оцениваем по max длине в будущем батче после сортировки
            if total >= token_budget:
                break
    candidates.sort(key=lambda c: len(c[0]))  # бакетинг по длине
    # обрезаем по реальному token_budget (с паддингом до максимума в выбранном подмножестве)
    out = []
    for c in candidates:
        trial = out + [c]
        T = max(len(x[0]) for x in trial)
        if T * len(trial) > token_budget and len(out) >= min_batch:
            break
        out.append(c)
    if not out:
        out = candidates[:min_batch] or candidates
    return out


# ======================================================================================
# 3. Батчинг генерации источников "lm"/"qa"/"context" → SourcePool
# ======================================================================================
def make_pool_from_source(tok, src, max_len, rng, raw_default=False):
    typ = src["type"]
    raw = src.get("raw", raw_default)
    examples = []
    if typ == "lm":
        lines = load_lm_lines(src["files"])
        for w in pack_lm_windows(tok, lines, max_len, rng, raw=src.get("raw", True)):
            examples.append(lm_example_to_seq(w))
    elif typ == "qa":
        data = json.loads(Path(src["file"]).read_text(encoding="utf-8"))
        max_q = src.get("max_q", 40)
        for item in data.get("train", data if isinstance(data, list) else []):
            q, a = item.get("q"), item.get("a")
            if q is None or a is None:
                continue
            examples.append(qa_example_to_seq(tok, q, a, max_q, raw))
    elif typ == "context":
        data = json.loads(Path(src["file"]).read_text(encoding="utf-8"))
        for item in data.get("train", []):
            examples.append(context_example_to_seq(tok, item["context"], item["q"], item["a"], max_len, raw))
    elif typ == "selfplay":
        data = json.loads(Path(src["file"]).read_text(encoding="utf-8")) if Path(src["file"]).exists() else {"train": []}
        max_q = src.get("max_q", 40)
        for item in data.get("train", []):
            examples.append(qa_example_to_seq(tok, item["q"], item["a"], max_q, raw))
    else:
        raise ValueError(f"неизвестный тип источника: {typ}")
    return SourcePool(examples, rng)


# ======================================================================================
# 4. Чекпойнты / resume
# ======================================================================================
def save_checkpoint(ckpt_dir, p, cfg, tok, meta, stage_idx, step, tokenizer_type):
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_doc = export_model(p, cfg, tok, meta, tokenizer_type=tokenizer_type,
                              max_q=meta.get("max_q"), merges=getattr(tok, "merges", None))
    path = ckpt_dir / f"ckpt_stage{stage_idx}_step{step}.json"
    path.write_text(json.dumps(model_doc, ensure_ascii=False), encoding="utf-8")
    state = {"stage_idx": stage_idx, "step": step, "ckpt_path": str(path)}
    (ckpt_dir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    # держим не более 2 последних чекпойнтов на диске (экономия места)
    ckpts = sorted(ckpt_dir.glob("ckpt_stage*.json"), key=lambda p_: p_.stat().st_mtime)
    for old in ckpts[:-2]:
        old.unlink(missing_ok=True)
    return path


def load_latest_checkpoint(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    state_path = ckpt_dir / "state.json"
    if not state_path.exists():
        return None
    state = json.loads(state_path.read_text())
    ckpt_path = Path(state["ckpt_path"])
    if not ckpt_path.exists():
        return None
    p, cfg, tok, meta = load_model(str(ckpt_path))
    return {"p": p, "cfg": cfg, "tok": tok, "meta": meta,
            "stage_idx": state["stage_idx"], "step": state["step"]}


# ======================================================================================
# 5. Модель: инициализация/рост
# ======================================================================================
def setup_model(init_path, train_texts, target_vocab=5000, d=192, n_layer=20, n_head=8,
                 block=10000, pos="rope", log=print):
    """init_path: старая модель (6 слоёв, learned pos, BPE V~1209). Расширяем словарь на train_texts
    (extend keeps old ids), затем init_params(новый V, d, n_layer, n_head, pos='rope', block=10000)
    и grow_params(new, old)."""
    p_old, cfg_old, tok, meta_old = load_model(init_path)
    tokenizer_type = meta_old.get("tokenizer", "bpe")
    if tokenizer_type == "bpe" and hasattr(tok, "extend"):
        added = tok.extend(train_texts, extra=max(0, target_vocab - len(tok)), verbose=True)
        log(f"Словарь расширен: +{added} токенов, V={len(tok)}")
    V = len(tok)
    p_new, cfg_new = init_params(V, d=d, n_layer=n_layer, n_head=n_head, block=block, pos=pos)
    p_new = grow_params(p_new, cfg_new, p_old, cfg_old)
    log(f"Архитектура: d={d}, слоёв={n_layer}, голов={n_head}, контекст={block}, "
        f"pos={pos}, V={V}, параметров={n_params(p_new):,}")
    return p_new, cfg_new, tok, tokenizer_type, meta_old.get("max_q", 40)
