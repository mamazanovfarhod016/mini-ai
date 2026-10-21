"""Эксперимент 16b: ТРАНСФОРМЕР-БЕЙЗЛАЙН (градиент) на той же задаче, что exp16a (ядро без градиента).
Задача одна и та же: по слову назвать синоним. Данные — data/words/exam_ru.jsonl kind=synonym (522 пар).
Сплит по СЛОВУ (не по паре): 80% слов — обучение, 20% слов модель вообще не видела ни с одним ответом —
честная проверка обобщения (как в train_transformer.py).
Модель — crop мини-GPT из mini_ai/transformer.py (RoPE, posthook такой же, как основной Mini AI),
обучение по символам: "<bos> синоним к слову «W» <sep> Синоним к слову «W» — «S».<eos>", loss только
на символах ответа. Это ЧЕСТНЫЙ противник: тот же корпус (522 примера), тот же CPU, число эпох ограничено
таймаутом (это маленький экспиремент, не продакшен-обучение).
Метрика — как в exp16a: доля вопросов ТЕСТА, где сгенерированный ответ содержит ПРАВИЛЬНЫЙ синоним
(ровно то слово или его начало/корень, проверяем вручную по includes), плюс сравнение с 15.9% top10
безградиентного letters-кода (там это было «ранг в словаре», здесь — «сгенерировал слово верно»).
"""
import sys, os, json, re, time, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.transformer import (
    BOS, EOS, PAD, SEP, CharTokenizer, KVModel, clean_question,
    init_params, loss_and_grads, AdamW, lr_schedule, clip_grads, n_params,
)

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"


def extract_syn(a): return re.findall(r"[«\"]([^»\"]+)[»\"]", a)[1:]


def build_pairs():
    rows = [json.loads(l) for l in open("data/words/exam_ru.jsonl", encoding="utf-8")]
    syn_rows = [r for r in rows if r["kind"] == "synonym"]
    words_ = sorted({r["word"] for r in syn_rows})
    rng = random.Random(7); rng.shuffle(words_)
    n_test = max(1, round(len(words_) * 0.2))
    test_w, train_w = set(words_[:n_test]), set(words_[n_test:])
    train_pairs, test_pairs = [], []
    for r in syn_rows:
        q = r["q"]; a = r["a"]; syns = extract_syn(a)
        if not syns: continue
        (test_pairs if r["word"] in test_w else train_pairs).append((q, a, r["word"], syns))
    return train_pairs, test_pairs


def main():
    t0 = time.time()
    train_pairs, test_pairs = build_pairs()
    print(f"обучение: {len(train_pairs)} пар, тест (слова НЕ виденные): {len(test_pairs)} пар", flush=True)
    chars = set()
    for q, a, *_ in train_pairs + test_pairs:
        chars.update(clean_question(q)); chars.update(clean_question(a))
    tok = CharTokenizer(sorted(chars))
    V = len(tok)
    print(f"словарь символов: {V}", flush=True)

    def encode_pair(q, a):
        qi = tok.encode(clean_question(q))[:40]
        ai = tok.encode(clean_question(a))[:120]
        ids = [BOS] + qi + [SEP] + ai + [EOS]
        mask = [0] * (2 + len(qi)) + [1] * (len(ai) + 1)
        return ids, mask

    enc_train = [encode_pair(q, a) for q, a, *_ in train_pairs]
    maxlen = max(len(ids) for ids, _ in enc_train)
    print(f"максимальная длина последовательности: {maxlen}", flush=True)

    D_MODEL, N_LAYER, N_HEAD = 96, 3, 4
    p, cfg = init_params(V, d=D_MODEL, n_layer=N_LAYER, n_head=N_HEAD, pos="rope", seed=0)
    print(f"параметров: {n_params(p):,}", flush=True)
    opt = AdamW(p, lr=3e-3, wd=0.01)

    rng = np.random.default_rng(0)
    BATCH = 16
    EPOCHS = int(os.environ.get("EPOCHS", 40))
    steps_per_epoch = max(1, len(enc_train) // BATCH)
    total_steps = EPOCHS * steps_per_epoch
    step = 0
    for ep in range(EPOCHS):
        order = rng.permutation(len(enc_train))
        ep_loss = 0.0
        for bi in range(0, len(order), BATCH):
            idxs = order[bi:bi + BATCH]
            batch = [enc_train[i] for i in idxs]
            T = max(len(ids) for ids, _ in batch)
            X = np.full((len(batch), T), PAD, dtype=np.int64)
            M = np.zeros((len(batch), T), dtype=np.float64)
            for i, (ids, mask) in enumerate(batch):
                X[i, :len(ids)] = ids; M[i, :len(mask)] = mask
            targets = np.zeros_like(X); targets[:, :-1] = X[:, 1:]
            mask_t = np.zeros_like(M); mask_t[:, :-1] = M[:, 1:]
            lr = lr_schedule(step, total_steps, 3e-3, warmup=max(5, total_steps // 20))
            loss, grads = loss_and_grads(p, cfg, X, targets, mask_t)
            clip_grads(grads, 1.0)
            opt.step(p, grads, lr)
            ep_loss += loss; step += 1
        if ep % 5 == 0 or ep == EPOCHS - 1:
            print(f"  epoch {ep:3} loss {ep_loss/steps_per_epoch:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    kv = KVModel(p, cfg)

    def generate(q):
        ids = [BOS] + tok.encode(clean_question(q))[:40] + [SEP]
        out, _ = kv.generate(ids, max_new=120, temperature=0.0)
        return tok.decode(out)

    def check():
        n_train_ok = 0
        for q, a, w, syns in train_pairs[:60]:
            gen = generate(q)
            n_train_ok += any(s.lower() in gen for s in syns)
        n_test_ok = 0; examples = []
        for q, a, w, syns in test_pairs:
            gen = generate(q)
            ok = any(s.lower() in gen for s in syns)
            n_test_ok += ok
            if len(examples) < 8: examples.append((w, syns, gen, ok))
        return n_train_ok / min(60, len(train_pairs)), n_test_ok / max(1, len(test_pairs)), examples

    train_acc, test_acc, examples = check()
    print(f"\nТРЕНИРОВОЧНЫЕ слова (память, 60 пример.): {train_acc*100:.1f}% содержат верный синоним", flush=True)
    print(f"ТЕСТОВЫЕ слова (НЕ видела ни разу): {test_acc*100:.1f}% содержат верный синоним  [обобщение]", flush=True)
    print("примеры на тесте:", flush=True)
    for w, syns, gen, ok in examples:
        print(f"  {'OK' if ok else 'NO':3} слово={w!r} ожидали один из {syns} -> сгенерировала: {gen!r}", flush=True)

    res = dict(train_pairs=len(train_pairs), test_pairs=len(test_pairs), vocab_chars=V,
               params=n_params(p), epochs=EPOCHS, train_acc=train_acc, test_acc=test_acc,
               seconds=round(time.time() - t0, 1))
    json.dump(res, open(f"{OUT}/exp16_transformer_baseline.json", "w"), indent=1, ensure_ascii=False)
    print(f"\nитого: {res}", flush=True)


if __name__ == "__main__":
    main()
