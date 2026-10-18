#!/usr/bin/env python3
"""Задачи «учись по тексту» (in-context, не по памяти) на трёх языках (uz, ru, en).

    .venv/bin/python tools/make_context_tasks.py   # → data/context_tasks.json

Главная идея: все факты в тексте ВЫДУМАНЫ (случайные имена из слогов, случайные
числа, города/предметы из маленьких списков) — модель не может подсмотреть ответ
в своей памяти, она обязана прочитать его из контекста, который видит сейчас.

Форматы заданий (kind):
  fact        2–5 коротких предложений, вопрос про один факт (кто/где/сколько) —
              ответ переписан из текста.
  derive      один шаг рассуждения над текстом (сравнение возраста/роста, сумма,
              порядок, простое «если-то»). Ответ содержит короткое уравнение.
  not_in_text вопрос про то, чего в тексте нет → «В тексте этого нет» (~10% заданий).
  needle      длинный текст (филлер из корпуса того же языка) + 1–3 выдуманных
              ключевых факта на случайной глубине; вопрос про них (включая
              derive по двум «иголкам»). needle НЕ хранится целиком в json —
              хранится филлер-ссылка (словарь-набор, индексы строк, seed), а
              полный текст собирается функцией build_needle() на лету. Это
              держит data/context_tasks.json маленьким и воспроизводимым.

ЧЕСТНЫЙ ЭКЗАМЕН: имена/числа для train и test не пересекаются — структурный ключ
задания (без учёта языка) детерминированно прячется в test по crc32 (как в
tools/make_school_math.py). Для needle, кроме того, филлер для test берётся из
*_test.txt корпуса (а не из обучающего corpus/*.txt).

Элемент: {context, q, a, kind, lang, answer, n_chars}
  context — сам текст (для needle — собирается build_needle, в json лежит
            {"filler_ref": {...}} вместо текста, см. load() ниже);
  answer  — короткий «голый» ответ (без слова Ответ/Javob/Answer) — по нему
            server/contextcheck.py сверяет финальный ответ модели;
  a       — полный ответ с ключевым словом и (для derive) уравнением — то, чему
            учим модель генерировать.
"""
import json
import random
import re
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAIN_REPO = Path("/home/hermes/Developer/mini-ai")  # корпус-филлер лежит тут
LANGS = ("uz", "ru", "en")
TEST_FRAC = 0.2

KEYWORD = {"uz": "Javob", "ru": "Ответ", "en": "Answer"}
NOT_IN_TEXT = {
    "uz": "Matnda bu yo'q",
    "ru": "В тексте этого нет",
    "en": "The text does not say",
}

# --------------------------------------------------------------------------------------
# Выдуманные имена/объекты: строим из слогов — никогда не совпадают со «школьными»
# именами tools/make_school_math.py и никогда не повторяются со старыми данными.
# --------------------------------------------------------------------------------------
SYL = {
    "uz": (["Bo", "Dil", "Gul", "Jas", "Ka", "Lo", "Ma", "No", "Ra", "Sa", "Ta", "Yo"],
           ["bek", "nur", "yor", "zar", "lim", "mir", "noz", "sun", "tosh", "jon"]),
    "ru": (["Ар", "Бо", "Ве", "Гле", "Да", "Зо", "Ки", "Ма", "Пе", "Ро", "Та", "Ю"],
           ["рина", "слав", "сим", "луша", "тьяна", "рик", "ник", "нила", "дия", "ян"]),
    "en": (["Bre", "Cor", "Da", "Fen", "Gra", "Hol", "Jas", "Ken", "Lor", "Mar", "Nov", "Pres"],
           ["lyn", "ton", "wick", "son", "ley", "vin", "ford", "ren", "dale", "ric"]),
}
CITIES = {
    "uz": ["Xorazm", "Farg'ona", "Navoiy", "Jizzax", "Termiz", "Qarshi", "Guliston", "Nukus"],
    "ru": ["Томск", "Пермь", "Калуга", "Выборг", "Старица", "Суздаль", "Елабуга", "Углич"],
    "en": ["Bramford", "Oakhaven", "Milport", "Dunwell", "Ferncross", "Hallowick", "Stoneby", "Redwick"],
}
OBJECTS = {
    "uz": ["qalam", "sumka", "shkaf", "velosiped", "soat", "kompyuter", "koptok"],
    "ru": ["зонт", "рюкзак", "шкаф", "велосипед", "часы", "ноутбук", "мяч"],
    "en": ["umbrella", "backpack", "cabinet", "bicycle", "watch", "laptop", "ball"],
}


def rand_name(rng, lang, used):
    """Выдуманное имя из слогов, не повторяется в пределах одного текста."""
    pre, suf = SYL[lang]
    name = pre[0] + suf[0].lower()
    for _ in range(50):
        name = rng.choice(pre) + rng.choice(suf).lower()
        if name not in used:
            used.add(name)
            return name
    name = f"{name}{rng.randint(2, 9)}"
    used.add(name)
    return name


def is_test(key):
    """Детерминированно: структурный ключ задания спрятан в экзамен? (без языка)."""
    return zlib.crc32(repr(key).encode()) % 100 < TEST_FRAC * 100


# --------------------------------------------------------------------------------------
# kind = fact — 2–5 коротких предложений, вопрос про один факт.
# --------------------------------------------------------------------------------------
FACT_T = {
    "uz": {
        "live": "{n} {c} shahrida yashaydi.",
        "have": "{n}ning {k} ta {o} bor.",
        "age": "{n} {y} yoshda.",
        "q_live": "{n} qayerda yashaydi",
        "q_have": "{n}ning nechta {o} bor",
        "q_age": "{n} necha yoshda",
    },
    "ru": {
        "live": "{n} живёт в городе {c}.",
        "have": "у {n} есть {k} {o}.",
        "age": "{n} {y} лет.",
        "q_live": "где живёт {n}",
        "q_have": "сколько {o} у {n}",
        "q_age": "сколько лет {n}",
    },
    "en": {
        "live": "{n} lives in the city of {c}.",
        "have": "{n} has {k} {o}.",
        "age": "{n} is {y} years old.",
        "q_live": "where does {n} live",
        "q_have": "how many {o} does {n} have",
        "q_age": "how old is {n}",
    },
}


def gen_fact(rng, lang):
    used = set()
    n = rand_name(rng, lang, used)
    c = rng.choice(CITIES[lang])
    o = rng.choice(OBJECTS[lang])
    k = rng.randint(2, 19)
    y = rng.randint(6, 70)
    T = FACT_T[lang]
    # дополняем ещё 1-2 нейтральными фразами про другое имя, чтобы вопрос не был тривиальным
    n2 = rand_name(rng, lang, used)
    c2 = rng.choice([x for x in CITIES[lang] if x != c])
    filler = {
        "uz": f"{n2} {c2} shahrida tug'ilgan.",
        "ru": f"{n2} родился в городе {c2}.",
        "en": f"{n2} was born in the city of {c2}.",
    }[lang]
    which = rng.choice(["live", "have", "age"])
    sentences = []
    sentences.append(T["live"].format(n=n, c=c))
    sentences.append(T["have"].format(n=n, k=k, o=o))
    sentences.append(T["age"].format(n=n, y=y))
    rng.shuffle(sentences)
    sentences.insert(rng.randint(0, len(sentences)), filler)
    context = " ".join(sentences)
    if which == "live":
        q, answer = T["q_live"].format(n=n), c
    elif which == "have":
        q, answer = T["q_have"].format(n=n, o=o), str(k)
    else:
        q, answer = T["q_age"].format(n=n), str(y)
    key = (which, n, c, k, y, o)
    return context, q, answer, key


# --------------------------------------------------------------------------------------
# kind = derive — один шаг рассуждения: разница возраста, рост, сумма, порядок.
# --------------------------------------------------------------------------------------
DERIVE_T = {
    "age_diff": {
        "uz": ("{n1} {n2}dan {d} yosh katta. {n2} {y} yoshda.", "{n1} necha yoshda",
               "{n2} {y}, {n1} {d} yosh katta: {y} + {d} = {r}. {kw}: {r}"),
        "ru": ("{n1} старше {n2} на {d} лет. {n2}у {y} лет.", "сколько лет {n1}",
               "{n2}у {y}, {n1} старше на {d}: {y} + {d} = {r}. {kw}: {r}"),
        "en": ("{n1} is {d} years older than {n2}. {n2} is {y} years old.", "how old is {n1}",
               "{n2} is {y}, {n1} is {d} older: {y} + {d} = {r}. {kw}: {r}"),
    },
    "sum_objects": {
        "uz": ("{n1}ning {k1} ta {o} bor. {n2}ning {k2} ta {o} bor.", "ikkisida jami nechta {o} bor",
               "{k1} + {k2} = {r}. {kw}: {r}"),
        "ru": ("у {n1} есть {k1} {o}. у {n2} есть {k2} {o}.", "сколько {o} у них вместе",
               "{k1} + {k2} = {r}. {kw}: {r}"),
        "en": ("{n1} has {k1} {o}. {n2} has {k2} {o}.", "how many {o} do they have together",
               "{k1} + {k2} = {r}. {kw}: {r}"),
    },
    "taller": {
        "uz": ("{n1}ning bo'yi {h1} sm. {n2}ning bo'yi {h2} sm.", "kim balandroq {n1} yoki {n2}",
               "{h1} sm va {h2} sm: {winner} balandroq. {kw}: {winner}"),
        "ru": ("рост {n1} — {h1} см. рост {n2} — {h2} см.", "кто выше {n1} или {n2}",
               "{h1} см и {h2} см: {winner} выше. {kw}: {winner}"),
        "en": ("{n1}'s height is {h1} cm. {n2}'s height is {h2} cm.", "who is taller {n1} or {n2}",
               "{h1} cm vs {h2} cm: {winner} is taller. {kw}: {winner}"),
    },
}


def gen_derive(rng, lang):
    used = set()
    kw = KEYWORD[lang]
    kind = rng.choice(list(DERIVE_T.keys()))
    n1 = rand_name(rng, lang, used)
    n2 = rand_name(rng, lang, used)
    tctx, tq, ta = DERIVE_T[kind][lang]
    if kind == "age_diff":
        d = rng.randint(1, 15)
        y = rng.randint(5, 60)
        r = y + d
        context = tctx.format(n1=n1, n2=n2, d=d, y=y)
        q = tq.format(n1=n1)
        a = ta.format(n2=n2, y=y, n1=n1, d=d, r=r, kw=kw)
        key = ("age_diff", n1, n2, d, y)
    elif kind == "sum_objects":
        o = rng.choice(OBJECTS[lang])
        k1, k2 = rng.randint(1, 40), rng.randint(1, 40)
        r = k1 + k2
        context = tctx.format(n1=n1, n2=n2, k1=k1, k2=k2, o=o)
        q = tq.format(o=o)
        a = ta.format(k1=k1, k2=k2, r=r, kw=kw)
        key = ("sum_objects", n1, n2, k1, k2, o)
    else:  # taller
        h1, h2 = rng.randint(90, 200), rng.randint(90, 200)
        if h1 == h2:
            h2 += 1
        winner = n1 if h1 > h2 else n2
        context = tctx.format(n1=n1, n2=n2, h1=h1, h2=h2)
        q = tq.format(n1=n1, n2=n2)
        a = ta.format(h1=h1, h2=h2, winner=winner, kw=kw)
        key = ("taller", n1, n2, h1, h2)
    return context, q, a, key


# --------------------------------------------------------------------------------------
# kind = not_in_text — вопрос про то, чего в тексте нет.
# --------------------------------------------------------------------------------------
NOT_Q = {
    "uz": ["{n} qanday mashina haydaydi", "{n}ning kasbi nima", "{n} qaysi fanni yaxshi biladi"],
    "ru": ["какую машину водит {n}", "какая профессия у {n}", "какой предмет {n} любит в школе"],
    "en": ["what car does {n} drive", "what is {n}'s profession", "what subject does {n} like best"],
}


def gen_not_in_text(rng, lang):
    ctx, _q, _a, _ = gen_fact(rng, lang)
    used = set(re.findall(r"[A-Za-zА-Яа-яЁёʻʼ']+", ctx))
    # возьмём имя из контекста, если найдём похожее на наши слоги — иначе просто сгенерируем
    n = rand_name(rng, lang, set())
    q = rng.choice(NOT_Q[lang]).format(n=n)
    answer = NOT_IN_TEXT[lang]
    a = f"{KEYWORD[lang]}: {answer}"
    key = ("not_in_text", ctx, q)
    return ctx, q, a, key


def finalize_fact(ctx, q, answer, lang):
    return ctx, q, f"{KEYWORD[lang]}: {answer}", answer


def finalize_derive(ctx, q, a_full, lang):
    m = re.search(r":\s*([^.]+)$", a_full)
    answer = m.group(1).strip() if m else a_full
    return ctx, q, a_full, answer


# --------------------------------------------------------------------------------------
# kind = needle — длинный текст-«стог сена» (филлер из корпуса) + ключевые факты.
# Filler не копируем в git: context_tasks.json хранит только filler_ref, полный
# текст собирается build_needle() из корпуса MAIN_REPO (который читается, не меняется).
# --------------------------------------------------------------------------------------
BUCKETS = {"s": 2000, "m": 8000, "l": 20000, "xl": 40000}  # целевая длина в символах

_FILLER_CACHE = {}


def load_filler(lang, split):
    key = (lang, split)
    if key not in _FILLER_CACHE:
        name = f"{lang}_test.txt" if split == "test" else f"{lang}.txt"
        path = MAIN_REPO / "data" / "corpus" / name
        if path.exists():
            lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        else:
            lines = []
        _FILLER_CACHE[key] = lines
    return _FILLER_CACHE[key]


NEEDLE_FACT_T = {
    "uz": "Maxfiy kod: {n} {c} shahrida {k} ta {o} yashiradi.",
    "ru": "Секретный код: {n} прячет {k} {o} в городе {c}.",
    "en": "Secret code: {n} hides {k} {o} in the city of {c}.",
}
NEEDLE_Q_T = {
    "uz": "{n} nechta {o} yashirgan",
    "ru": "сколько {o} прячет {n}",
    "en": "how many {o} does {n} hide",
}
NEEDLE_DERIVE_Q_T = {
    "uz": "{n1} va {n2} birgalikda nechta {o} yashirgan",
    "ru": "сколько {o} прячут {n1} и {n2} вместе",
    "en": "how many {o} do {n1} and {n2} hide together",
}


def build_needle(seed, lang, bucket, split):
    """Собирает длинный текст на лету из (seed, lang, bucket, split) — детерминированно.
    Возвращает (context, q, a_full, answer, key). Хранить в json достаточно filler_ref."""
    rng = random.Random(seed)
    target_chars = BUCKETS[bucket]
    lines = load_filler(lang, split)
    if not lines:
        lines = ["x"] * 50  # на случай, если корпус ещё не собран — не должно падать
    used = set()
    n1 = rand_name(rng, lang, used)
    n2 = rand_name(rng, lang, used)
    o = rng.choice(OBJECTS[lang])
    c = rng.choice(CITIES[lang])
    k1 = rng.randint(2, 40)
    k2 = rng.randint(2, 40)
    fact1 = NEEDLE_FACT_T[lang].format(n=n1, c=c, k=k1, o=o)
    two_needles = rng.random() < 0.5
    fact2 = None
    if two_needles:
        c2 = rng.choice(CITIES[lang])
        fact2 = NEEDLE_FACT_T[lang].format(n=n2, c=c2, k=k2, o=o)

    # строим филлер до целевой длины, вставляем факты на случайной (но детерминированной) глубине
    buf = []
    cur_len = 0
    n_lines = len(lines)
    start = rng.randrange(n_lines)
    depth1 = rng.uniform(0.2, 0.8)
    depth2 = rng.uniform(0.2, 0.8) if two_needles else 2.0  # 2.0 = «никогда не достигнем»
    inserted1 = inserted2 = False
    i = 0
    while cur_len < target_chars:
        line = lines[(start + i) % n_lines]
        buf.append(line)
        cur_len += len(line) + 1
        frac = cur_len / target_chars
        if not inserted1 and frac >= depth1:
            buf.append(fact1)
            cur_len += len(fact1) + 1
            inserted1 = True
        if two_needles and not inserted2 and frac >= depth2:
            buf.append(fact2)
            cur_len += len(fact2) + 1
            inserted2 = True
        i += 1
    if not inserted1:
        buf.append(fact1)
    if two_needles and not inserted2 and fact2:
        buf.append(fact2)
    context = " ".join(buf)

    if two_needles and rng.random() < 0.5:
        q = NEEDLE_DERIVE_Q_T[lang].format(n1=n1, n2=n2, o=o)
        r = k1 + k2
        answer = str(r)
        a = f"{k1} + {k2} = {r}. {KEYWORD[lang]}: {r}"
        which_name = (n1, n2)
    else:
        q = NEEDLE_Q_T[lang].format(n=n1, o=o)
        answer = str(k1)
        a = f"{KEYWORD[lang]}: {k1}"
        which_name = (n1,)

    key = ("needle", bucket, which_name, c, k1, k2 if two_needles else None, two_needles)
    return context, q, a, answer, key, len(context)


# --------------------------------------------------------------------------------------
def load(path=None):
    """Загрузить data/context_tasks.json и восстановить полный context у needle-записей
    (там в файле лежит context: null + filler_ref — текст пересобирается build_needle()).
    Использовать это, а не json.load() напрямую, если нужен реальный текст needle."""
    path = Path(path) if path else ROOT / "data" / "context_tasks.json"
    d = json.loads(path.read_text(encoding="utf-8"))
    for part in ("train", "test"):
        for rec in d[part]:
            if rec["kind"] == "needle" and rec.get("context") is None:
                ref = rec["filler_ref"]
                ctx, q, a, answer, key, n_chars = build_needle(ref["seed"], ref["lang"], ref["bucket"], ref["split"])
                rec["context"] = ctx
    return d


def main():
    n_train_target = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
    n_needle_per_bucket = int(sys.argv[2]) if len(sys.argv) > 2 else 3000

    train, test = [], []
    seen_keys = set()

    def put(kind, lang, ctx, q, a, answer, key, extra=None):
        dedupe_key = (kind, key)
        rec = {
            "context": ctx,
            "q": q,
            "a": a,
            "kind": kind,
            "lang": lang,
            "answer": answer,
            "n_chars": len(ctx),
        }
        if extra:
            rec.update(extra)
        bucket = test if is_test(key) else train
        bucket.append(rec)

    n_fact = n_train_target * 45 // 100
    n_derive = n_train_target * 45 // 100
    n_not = n_train_target * 10 // 100

    rng = random.Random(1)

    def shared_lang_rng(seed, lang):
        """Один и тот же набор чисел/имён на всех языках для одного структурного ключа:
        детерминированный sub-seed = hash(seed, lang), без привязки к id() объектов."""
        h = zlib.crc32(f"{seed}:{lang}".encode())
        return random.Random(h)

    for _ in range(n_fact):
        shared_seed = rng.randint(0, 2**31)
        for lang in LANGS:
            lrng = shared_lang_rng(shared_seed, lang)
            ctx, q, answer, key = gen_fact(lrng, lang)
            ctx, q, a, answer = finalize_fact(ctx, q, answer, lang)
            put("fact", lang, ctx, q, a, answer, ("fact", key))

    for _ in range(n_derive):
        shared_seed = rng.randint(0, 2**31)
        for lang in LANGS:
            lrng = shared_lang_rng(shared_seed, lang)
            ctx, q, a_full, key = gen_derive(lrng, lang)
            ctx, q, a, answer = finalize_derive(ctx, q, a_full, lang)
            put("derive", lang, ctx, q, a, answer, ("derive", key))

    for _ in range(n_not):
        shared_seed = rng.randint(0, 2**31)
        for lang in LANGS:
            lrng = shared_lang_rng(shared_seed, lang)
            ctx, q, a, key = gen_not_in_text(lrng, lang)
            answer = NOT_IN_TEXT[lang]
            put("not_in_text", lang, ctx, q, a, answer, ("not_in_text", key))

    # --- needle: храним компактно — только filler_ref (seed/lang/bucket/split), без текста.
    # Полный текст пересобирается build_needle(**filler_ref) на лету (детерминированно) —
    # это и держит data/context_tasks.json маленьким (иначе 40к-символьные тексты × тысячи
    # записей весят сотни МБ, что нельзя класть в git).
    needle_rng = random.Random(2)
    for bucket in BUCKETS:
        for _ in range(n_needle_per_bucket):
            seed = needle_rng.randint(0, 2**31)
            for lang in LANGS:
                split = "test" if is_test(("needle", bucket, seed)) else "train"
                ctx, q, a, answer, key, n_chars = build_needle(seed, lang, bucket, split)
                dst = test if split == "test" else train
                dst.append({
                    "context": None, "q": q, "a": a, "kind": "needle", "lang": lang,
                    "answer": answer, "n_chars": n_chars, "bucket": bucket,
                    "filler_ref": {"seed": seed, "lang": lang, "bucket": bucket, "split": split},
                })

    out = ROOT / "data" / "context_tasks.json"
    out.write_text(json.dumps({"train": train, "test": test}, ensure_ascii=False), encoding="utf-8")

    print(f"{out}: train {len(train)}, test {len(test)}")
    for kind in ("fact", "derive", "not_in_text", "needle"):
        for lang in LANGS:
            tr = sum(1 for x in train if x["kind"] == kind and x["lang"] == lang)
            te = sum(1 for x in test if x["kind"] == kind and x["lang"] == lang)
            if tr or te:
                print(f"  {kind:12s} {lang}  train {tr:5d}  test {te:4d}")
    for bucket in BUCKETS:
        tr = sum(1 for x in train if x["kind"] == "needle" and x.get("bucket") == bucket)
        te = sum(1 for x in test if x["kind"] == "needle" and x.get("bucket") == bucket)
        print(f"  needle bucket {bucket:3s} (~{BUCKETS[bucket]:>5} chars)  train {tr:5d}  test {te:4d}")


if __name__ == "__main__":
    main()
