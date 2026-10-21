"""Эксперимент 21: СКОРОСТЬ ОБУЧЕНИЯ при РАВНОМ умении. Честное сравнение на одной задаче (bAbI task1,
критерий round2, где наше ядро даёт 100%): сколько времени нужно градиентному трансформеру, чтобы
натренироваться до ТОГО ЖЕ уровня (100% / близко), и сколько времени ядру — чтобы «научиться» (= один
проход чтения, без градиента). Задача настоящая: каждая история своя, нужно найти последнее упоминание
имени и ответить следующим за ним словом места — зубрить нечем, 900 train-историй уникальны.
Данные: data/babi/task_1-single_supporting_fact_{train,test}.parquet (900 train, 1000 test).
Формат трансформера — как exp16b: "<bos> story <sep> question <sep> " -> "answer<eos>", char-level,
loss только на ответе. Мера «научилась»: точность на TEST (не train) ≥ 95% (критерий round2).
"""
import sys, os, re, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.transformer import (
    BOS, EOS, SEP, PAD, CharTokenizer, KVModel, clean_question,
    init_params, loss_and_grads, AdamW, lr_schedule, clip_grads, n_params,
)

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"


def main():
    import pandas as pd
    train_df = pd.read_parquet("data/babi/task_1-single_supporting_fact_train.parquet")
    test_df = pd.read_parquet("data/babi/task_1-single_supporting_fact_test.parquet")

    def norm(s): return re.sub(r"\s+", " ", s.strip().lower())
    train_pairs = [(norm(r["passage"]) + " <q> " + norm(r["question"]), r["answer"].strip().lower()) for _, r in train_df.iterrows()]
    test_pairs = [(norm(r["passage"]) + " <q> " + norm(r["question"]), r["answer"].strip().lower()) for _, r in test_df.iterrows()]

    chars = set()
    for q, a in train_pairs + test_pairs: chars.update(q); chars.update(a)
    tok = CharTokenizer(sorted(chars)); V = len(tok)
    print(f"train={len(train_pairs)} test={len(test_pairs)} словарь_символов={V}", flush=True)

    def encode(q, a):
        qi = tok.encode(q)[:220]; ai = tok.encode(a)[:20]
        ids = [BOS] + qi + [SEP] + ai + [EOS]
        mask = [0] * (2 + len(qi)) + [1] * (len(ai) + 1)
        return ids, mask

    enc_train = [encode(q, a) for q, a in train_pairs]

    D_MODEL, N_LAYER, N_HEAD = int(os.environ.get("D_MODEL", 64)), int(os.environ.get("N_LAYER", 2)), 4
    p, cfg = init_params(V, d=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD, pos="rope", seed=0)
    print(f"модель: d={D_MODEL} layers={N_LAYER} heads={N_HEAD} params={n_params(p):,}", flush=True)
    opt = AdamW(p, lr=4e-3, wd=0.01)
    rng = np.random.default_rng(0)
    BATCH = 32
    steps_per_epoch = max(1, len(enc_train) // BATCH)
    MAX_EPOCHS = int(os.environ.get("MAX_EPOCHS", 150))
    total_steps = MAX_EPOCHS * steps_per_epoch

    def test_accuracy(kv, n=200):
        rng_t = np.random.default_rng(1)
        idxs = rng_t.choice(len(test_pairs), size=min(n, len(test_pairs)), replace=False)
        ok = 0
        for i in idxs:
            q, gold = test_pairs[i]
            ids = [BOS] + tok.encode(q)[:220] + [SEP]
            out, _ = kv.generate(ids, max_new=20, temperature=0.0)
            pred = tok.decode(out).strip()
            ok += (pred == gold)
        return ok / len(idxs)

    t0 = time.time(); step = 0; learned_at = None; learned_epoch = None
    for ep in range(MAX_EPOCHS):
        order = rng.permutation(len(enc_train)); ep_loss = 0.0
        for bi in range(0, len(order), BATCH):
            idxs = order[bi:bi + BATCH]; batch = [enc_train[i] for i in idxs]
            T = max(len(ids) for ids, _ in batch)
            X = np.full((len(batch), T), PAD, dtype=np.int64); M = np.zeros((len(batch), T), dtype=np.float64)
            for i, (ids, mask) in enumerate(batch): X[i, :len(ids)] = ids; M[i, :len(mask)] = mask
            targets = np.zeros_like(X); targets[:, :-1] = X[:, 1:]
            mask_t = np.zeros_like(M); mask_t[:, :-1] = M[:, 1:]
            lr = lr_schedule(step, total_steps, 4e-3, warmup=max(5, total_steps // 20))
            loss, grads = loss_and_grads(p, cfg, X, targets, mask_t)
            clip_grads(grads, 1.0); opt.step(p, grads, lr); ep_loss += loss; step += 1
        if ep % 5 == 0 or ep == MAX_EPOCHS - 1:
            kv = KVModel(p, cfg)
            acc = test_accuracy(kv, n=200)
            elapsed = time.time() - t0
            print(f"  epoch {ep:3} loss {ep_loss/steps_per_epoch:.3f}  TEST acc {acc*100:5.1f}%  ({elapsed:.0f}s)", flush=True)
            if acc >= 0.95 and learned_at is None:
                learned_at = elapsed; learned_epoch = ep
                print(f"  >>> ДОСТИГНУТ критерий round2 (>=95% на тесте) за {elapsed:.1f}s, эпоха {ep}", flush=True)
                break

    total_time = time.time() - t0
    kv = KVModel(p, cfg)
    final_acc = test_accuracy(kv, n=1000)
    print(f"\nФИНАЛ: точность на ВСЕХ 1000 тестовых = {final_acc*100:.1f}%  обучение заняло {total_time:.1f}s"
          f"{f' (критерий 95% достигнут за {learned_at:.1f}s, эпоха {learned_epoch})' if learned_at else ' (критерий 95% НЕ достигнут)'}", flush=True)

    # ядро: время "обучения" = время ОДНОГО прохода чтения всех 1000 тестовых историй (без градиента)
    import importlib
    import tools.exp17_babi_reader as core
    importlib.reload(core)
    t0c = time.time()
    core.run_task("ядро (без градиента)", test_df, "loc")
    core_time = time.time() - t0c

    res = dict(train_n=len(train_pairs), test_n=len(test_pairs), params=n_params(p),
               transformer_final_acc=final_acc, transformer_total_time=round(total_time, 1),
               transformer_time_to_95pct=round(learned_at, 1) if learned_at else None,
               transformer_epoch_to_95pct=learned_epoch,
               core_time_same_test=round(core_time, 3),
               speedup_vs_total=round(total_time / core_time, 1),
               speedup_vs_95pct=(round(learned_at / core_time, 1) if learned_at else None))
    print(f"\nСРАВНЕНИЕ (то же умение, N={len(test_pairs)} тестовых вопросов):")
    print(f"  трансформер время до 95%: {res['transformer_time_to_95pct']}s (эпоха {learned_epoch})" if learned_at else "  трансформер 95% не достиг")
    print(f"  ядро время (один проход чтения тех же данных): {core_time:.3f}s")
    if learned_at: print(f"  >>> ЯДРО УЧИТСЯ БЫСТРЕЕ В {res['speedup_vs_95pct']:.0f} РАЗ (при равном умении >=95%)")
    json.dump(res, open(f"{OUT}/exp21_learning_speed.json", "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
