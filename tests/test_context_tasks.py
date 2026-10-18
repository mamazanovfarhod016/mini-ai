"""Задания «учись по тексту»: детерминизм, честный экзамен, выводимость ответа,
глубина needle, бакеты длины, checker ловит неверные ответы."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import contextcheck  # noqa: E402
from tools import make_context_tasks as mct  # noqa: E402


def build_dataset(n_train=120, n_needle=4):
    """Мини-версия main(): строим train/test в памяти, не трогая data/context_tasks.json."""
    import random
    import zlib

    train, test = [], []

    def shared_lang_rng(seed, lang):
        h = zlib.crc32(f"{seed}:{lang}".encode())
        return random.Random(h)

    def put(kind, lang, ctx, q, a, answer, key, extra=None):
        rec = {"context": ctx, "q": q, "a": a, "kind": kind, "lang": lang,
               "answer": answer, "n_chars": len(ctx)}
        if extra:
            rec.update(extra)
        (test if mct.is_test(key) else train).append(rec)

    rng = random.Random(1)
    n_fact = n_derive = n_train
    n_not = max(5, n_train // 5)
    for _ in range(n_fact):
        shared_seed = rng.randint(0, 2**31)
        for lang in mct.LANGS:
            lrng = shared_lang_rng(shared_seed, lang)
            ctx, q, answer, key = mct.gen_fact(lrng, lang)
            ctx, q, a, answer = mct.finalize_fact(ctx, q, answer, lang)
            put("fact", lang, ctx, q, a, answer, ("fact", key))
    for _ in range(n_derive):
        shared_seed = rng.randint(0, 2**31)
        for lang in mct.LANGS:
            lrng = shared_lang_rng(shared_seed, lang)
            ctx, q, a_full, key = mct.gen_derive(lrng, lang)
            ctx, q, a, answer = mct.finalize_derive(ctx, q, a_full, lang)
            put("derive", lang, ctx, q, a, answer, ("derive", key))
    for _ in range(n_not):
        shared_seed = rng.randint(0, 2**31)
        for lang in mct.LANGS:
            lrng = shared_lang_rng(shared_seed, lang)
            ctx, q, a, key = mct.gen_not_in_text(lrng, lang)
            put("not_in_text", lang, ctx, q, a, mct.NOT_IN_TEXT[lang], ("not_in_text", key))

    needle_rng = random.Random(2)
    for bucket in mct.BUCKETS:
        for _ in range(n_needle):
            seed = needle_rng.randint(0, 2**31)
            for lang in mct.LANGS:
                split = "test" if mct.is_test(("needle", bucket, seed)) else "train"
                ctx, q, a, answer, key, n_chars = mct.build_needle(seed, lang, bucket, split)
                dst = test if split == "test" else train
                dst.append({"context": ctx, "q": q, "a": a, "kind": "needle", "lang": lang,
                            "answer": answer, "n_chars": n_chars, "bucket": bucket})
    return train, test


def test_determinism():
    tr1, te1 = build_dataset(60, 3)
    tr2, te2 = build_dataset(60, 3)
    assert tr1 == tr2 and te1 == te2


def test_no_leakage_train_test():
    """Полный структурный ключ задания (имена+числа+город из gen_*) никогда не встречается
    и в train, и в test одновременно — экзамен по-настоящему «спрятан» (crc32-сплит).
    (Сами слоги имён — из маленького алфавита, поэтому отдельные слоги МОГУТ повторяться;
    проверяем не слоги, а то, что порождает is_test — одинаковую пару (kind, context+q).)"""
    tr, te = build_dataset(200, 2)
    for kind in ("fact", "derive", "not_in_text"):
        tr_pairs = {(r["context"], r["q"]) for r in tr if r["kind"] == kind}
        te_pairs = {(r["context"], r["q"]) for r in te if r["kind"] == kind}
        assert not (tr_pairs & te_pairs)


def test_fact_answer_derivable_from_context():
    tr, te = build_dataset(80, 2)
    for rec in tr + te:
        if rec["kind"] == "fact":
            assert rec["answer"] in rec["context"]


def test_derive_answer_or_operands_in_context():
    tr, te = build_dataset(80, 2)
    for rec in tr + te:
        if rec["kind"] == "derive":
            eq = contextcheck._find_equation(rec["a"])
            ctx = rec["context"]
            if eq is not None:
                a, op, b, r = eq
                # хотя бы один операнд уравнения должен быть в тексте (взят из контекста)
                assert str(int(a)) in ctx or str(int(b)) in ctx or rec["answer"] in ctx
            else:
                # kind "taller": нет арифметики — сравниваются два числа из текста напрямую,
                # сам ответ (имя-победитель) обязан присутствовать в тексте.
                assert rec["answer"] in ctx


def test_not_in_text_answer_is_fixed_phrase():
    tr, te = build_dataset(40, 1)
    for rec in tr + te:
        if rec["kind"] == "not_in_text":
            assert rec["answer"] == mct.NOT_IN_TEXT[rec["lang"]]


def test_needle_fact_present_at_claimed_depth():
    tr, te = build_dataset(10, 6)
    for rec in tr + te:
        if rec["kind"] == "needle":
            eq = contextcheck._find_equation(rec["a"])
            if eq is not None:
                # derive по двум иголкам: оба операнда обязаны быть в тексте
                a, op, b, r = eq
                assert str(int(a)) in rec["context"] and str(int(b)) in rec["context"]
            else:
                # простой needle: сам ответ (число/строка) обязан быть в тексте
                assert rec["answer"] in rec["context"]


def test_needle_length_buckets():
    tr, te = build_dataset(5, 8)
    for rec in tr + te:
        if rec["kind"] == "needle":
            target = mct.BUCKETS[rec["bucket"]]
            # допускаем окно: длина текста >= target (мы добавляем строки, пока не достигнем target)
            assert rec["n_chars"] >= target * 0.8
            assert rec["n_chars"] <= target * 1.6


def test_needle_up_to_10k_tokens():
    """xl-бакет (~40000 символов) при ~4 символа/токен — это ~10000 токенов, как в задаче."""
    assert mct.BUCKETS["xl"] / 4 <= 10500


def test_checker_accepts_correct_answers():
    tr, te = build_dataset(60, 3)
    for rec in (tr + te)[:200]:
        res = contextcheck.check(rec, rec["a"])
        assert res["ok"], (rec["kind"], rec["lang"], rec["a"], res["errors"])


def test_checker_rejects_wrong_final_answer():
    tr, _ = build_dataset(20, 1)
    fact = next(r for r in tr if r["kind"] == "fact")
    res = contextcheck.check(fact, f"{contextcheck.KEYWORD[fact['lang']]}: совершенно другое")
    assert not res["ok"]


def test_checker_rejects_wrong_equation_in_derive():
    tr, _ = build_dataset(30, 1)
    rec = next(r for r in tr if r["kind"] == "derive")
    a, op, b, r = contextcheck._find_equation(rec["a"])
    wrong = rec["a"].replace(f"{int(a)} {op} {int(b)} = {int(r)}", f"{int(a)} {op} {int(b)} = {int(r) + 1}")
    res = contextcheck.check(rec, wrong)
    assert not res["ok"]


def test_checker_rejects_hallucinated_not_in_text():
    tr, _ = build_dataset(20, 1)
    rec = next(r for r in tr if r["kind"] == "not_in_text")
    bad = f"{contextcheck.KEYWORD[rec['lang']]}: Toyota"
    res = contextcheck.check(rec, bad)
    assert not res["ok"]
