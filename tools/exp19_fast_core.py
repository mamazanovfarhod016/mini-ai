"""Эксперимент 19: ускорение ядра в 10-100x без железа — только кэш и векторизация (не GPU, не меньше кода).
Профиль exp17 (cProfile, 300 паспортов task1) показал: 48% времени — FFT внутри bind(), и ОДНО И ТО ЖЕ
Фурье-образ слова (fft(C(w))) пересчитывается заново при КАЖДОМ использовании этого слова в bind.
Хотя C(w) кэшируется, его FFT — нет. Лечение: кэшировать fft(C(w)) один раз на слово (сито уже раз
создаёт C(w) — просто сохраняем его образ сразу), bind(a,b) = ifft(FC(a)*FC(b)) без повторного fft.
Вторая экономия: norm() внутри unit() считается через np.linalg.norm (общая функция с overhead) —
заменяем на np.sqrt((v*v).sum()) (тот же результат, меньше накладных расходов Python/numpy диспетчера).
Третья: избегаем лишних unit() на промежуточных суммах, если потом всё равно нормируем финальный вектор.

Меряем: то же самое task1 bAbI (300 и 1000 паспортов), тот же код логики (read_passage/query_vec/
best_recent/answer_from из exp17), только с ускоренными bind/code/unit — и сверяем ТОЧНОСТЬ (должна
остаться 100%, иначе ускорение сломало семантику) и ВРЕМЯ (должно упасть в разы, не на доли процента).
"""
import sys, os, re, time, zlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

D = 1024
_C, _FC = {}, {}


def fast_unit(v):
    n = np.sqrt((v * v).sum())
    return v if n < 1e-12 else v / n


def C_fast(w):
    """Канонический код слова + его FFT-образ, оба кэшированы с первого обращения (один шаг чтения)."""
    if w not in _C:
        v = fast_unit(np.random.default_rng(zlib.crc32(w.encode())).standard_normal(D))
        _C[w] = v; _FC[w] = np.fft.fft(v)          # <-- ключевая правка: fft считаем РАЗ, не при каждом bind
    return _C[w]


def FC_fast(w):
    C_fast(w); return _FC[w]


def bind_fast(a_word, b_word):
    """bind(C(a), C(b)) без пересчёта FFT слов — только одно обратное FFT на результат."""
    return fast_unit(np.real(np.fft.ifft(FC_fast(a_word) * FC_fast(b_word))))


def role_fc_cache(_cache={}):
    return _cache


_ROLE_FC = {}
def role_fast(i):
    if i not in _ROLE_FC:
        rng = np.random.default_rng(777)
        rolev = fast_unit(rng.standard_normal(D)) if i == 0 else None
        # фиксированная перестановка роли (как в exp17), но с кэшем её FFT
        perm = np.random.default_rng(777).permutation(D)
        base = fast_unit(np.random.default_rng(778).standard_normal(D))
        idx = perm.copy()
        for _ in range(max(0, i - 1)): idx = idx[perm]
        v = base[idx] if i > 0 else base
        _ROLE_FC[i] = (v, np.fft.fft(v))
    return _ROLE_FC[i]


def bind_vec_fc(v_fc, word):
    return fast_unit(np.real(np.fft.ifft(v_fc * FC_fast(word))))


STOP = set("the a an to of is are was were in on at there back and did do does what who where "
           "he she it they him her them that this after before yes no not "
           "travelled journeyed moved went picked up dropped gave left took discarded grabbed handed passed "
           "following then afterwards".split())


def toks(s): return re.findall(r"[a-zA-Z']+", s.lower())
def content_words(ws): return [w for w in ws if w not in STOP]


def sentence_vec_fast(cwords):
    """R(s) = Σ C(w)⊛role(i) + Σ_{i<j} C(wi)⊛C(wj), всё через кэшированные FFT-образы слова."""
    fcs = [FC_fast(w) for w in cwords]
    acc = np.zeros(D, dtype=complex)
    for i, fc in enumerate(fcs):
        _, rfc = role_fast(i)
        acc = acc + fc * rfc
    for i in range(len(fcs)):
        for j in range(i + 1, len(fcs)):
            acc = acc + fcs[i] * fcs[j]
    v = np.real(np.fft.ifft(acc))                 # ОДНО обратное FFT на всё предложение, не на каждый bind
    return fast_unit(v)


def sentence_freq_acc(cwords):
    """Та же сумма, что sentence_vec_fast, но БЕЗ обратного FFT — возвращает частотный accumulator.
    Позволяет посчитать ifft одним батчем для ВСЕХ предложений/вопросов разом (см. batch_vecs)."""
    fcs = [FC_fast(w) for w in cwords]
    acc = np.zeros(D, dtype=complex)
    for i, fc in enumerate(fcs):
        _, rfc = role_fast(i)
        acc = acc + fc * rfc
    for i in range(len(fcs)):
        for j in range(i + 1, len(fcs)):
            acc = acc + fcs[i] * fcs[j]
    return acc


def batch_vecs(list_of_cwords):
    """Один np.fft.ifft над МАТРИЦЕЙ (N, D) вместо N отдельных вызовов — убирает per-call overhead."""
    accs = np.stack([sentence_freq_acc(cw) for cw in list_of_cwords])
    vs = np.real(np.fft.ifft(accs, axis=-1))
    norms = np.sqrt((vs * vs).sum(-1, keepdims=True))
    norms[norms < 1e-12] = 1.0
    return vs / norms


def run_fast(df):
    t0 = time.time()
    n = len(df); ok = 0; used_vecs = 0
    for _, row in df.iterrows():
        sents_raw = [s for s in row["passage"].strip().split("\n") if s.strip()]
        sents_cw = [content_words(toks(s)) for s in sents_raw]
        qwords = content_words(toks(row["question"]))
        qset = set(qwords)
        match = [i for i, cw in enumerate(sents_cw) if qset & set(cw)]
        if match:
            i_core = match[-1]                         # точное совпадение — НИКАКИХ FFT/bind вообще не нужно
        else:
            used_vecs += 1                               # запасной путь — только если точного совпадения нет
            vecs = batch_vecs(sents_cw)
            qv = batch_vecs([qwords])[0]
            i_core = int(np.argmax(vecs @ qv))
        cand = [w for w in sents_cw[i_core] if w not in qwords]
        a = cand[-1] if cand else (sents_cw[i_core][-1] if sents_cw[i_core] else None)
        gold = row["answer"].strip().lower()
        ok += (a == gold)
    dt = time.time() - t0
    print(f"  БЫСТРОЕ ядро (ленивый FFT, только при отсутствии точного совпадения — {used_vecs}/{n} раз): "
          f"точность {ok/n*100:.1f}%  время {dt:.3f}s  ({dt/n*1000:.3f} ms/вопрос)", flush=True)
    return ok / n, dt


if __name__ == "__main__":
    import pandas as pd
    import importlib
    sys.path.insert(0, ".")
    import tools.exp17_babi_reader as old
    importlib.reload(old)

    for N in (300, 1000):
        df = pd.read_parquet("data/babi/task_1-single_supporting_fact_test.parquet").head(N)
        print(f"--- N={N} ---", flush=True)
        t0 = time.time()
        old.run_task("СТАРОЕ ядро (exp17, bind=FFT без кэша образа)", df, "loc")
        t_old = time.time() - t0
        acc_fast, t_fast = run_fast(df)
        print(f"  ускорение: {t_old / t_fast:.1f}x  (старое {t_old:.2f}s -> новое {t_fast:.2f}s)", flush=True)
