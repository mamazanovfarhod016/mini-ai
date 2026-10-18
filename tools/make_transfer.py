"""
Тест ПЕРЕНОСА МЕЖДУ ЯЗЫКАМИ: факт учим на ОДНОМ языке — спрашиваем на ДВУХ других.

    .venv/bin/python tools/make_transfer.py      # → data/transfer.json {train, test}

Как устроено (всё выдумано, чтобы знание нельзя было взять из интернета или корпуса):
  • 600 выдуманных людей (имя из слогов) и у каждого 1 факт: город, профессия, любимое животное, возраст;
  • факт учим только на одном языке (по кругу uz → ru → en), причём в 4 формулировках вопроса;
  • экзамен: тот же вопрос на ДВУХ ДРУГИХ языках. Ответ — слово из маленького списка (город/профессия/…),
    у каждого слова есть перевод, проверка — скриптом (ответ на языке вопроса ИЛИ на языке обучения засчитываем
    отдельно: «перенос смысла» и «перенос с ответом на другом языке»).
  • Контроль: 20% фактов спрашиваем на ТОМ ЖЕ языке, но новой формулировкой — это верхняя планка.
Цифра «перенос» = доля верных ответов на других языках. У маленькой сети она будет низкой —
наша задача поднимать её от версии к версии (параллельные тексты, общие задания на 3 языках).
"""
import json
import random
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SYL = ["ra", "mo", "zi", "ta", "lu", "ke", "no", "bi", "sa", "du", "fe", "ri", "go", "va", "xu", "le", "pa", "yo"]
VALUES = {
    "city": [("Tokio", "Токио", "Tokyo"), ("Parij", "Париж", "Paris"), ("Qohira", "Каир", "Cairo"),
             ("Rim", "Рим", "Rome"), ("London", "Лондон", "London"), ("Pekin", "Пекин", "Beijing"),
             ("Madrid", "Мадрид", "Madrid"), ("Berlin", "Берлин", "Berlin")],
    "job": [("shifokor", "врач", "doctor"), ("o'qituvchi", "учитель", "teacher"), ("haydovchi", "водитель", "driver"),
            ("oshpaz", "повар", "cook"), ("dehqon", "фермер", "farmer"), ("rassom", "художник", "artist")],
    "animal": [("mushuk", "кошка", "cat"), ("it", "собака", "dog"), ("ot", "лошадь", "horse"),
               ("quyon", "кролик", "rabbit"), ("baliq", "рыба", "fish"), ("qush", "птица", "bird")],
}
LI = {"uz": 0, "ru": 1, "en": 2}
Q = {
    "city": {"uz": ["{n} qaysi shaharda yashaydi", "{n} qayerda yashaydi", "{n}ning shahri qaysi", "{n} qaysi shaharlik"],
             "ru": ["в каком городе живёт {n}", "где живёт {n}", "какой город у {n}", "{n} из какого города"],
             "en": ["which city does {n} live in", "where does {n} live", "what is the city of {n}", "what city is {n} from"]},
    "job": {"uz": ["{n} kim bo'lib ishlaydi", "{n}ning kasbi nima", "{n} qayerda ishlaydi kim bo'lib", "{n} nima ish qiladi"],
            "ru": ["кем работает {n}", "какая профессия у {n}", "чем {n} занимается на работе", "кто {n} по профессии"],
            "en": ["what is the job of {n}", "what does {n} do for work", "what is {n} by profession", "what work does {n} do"]},
    "animal": {"uz": ["{n}ning sevimli hayvoni qaysi", "{n} qaysi hayvonni yaxshi ko'radi", "{n} qanday hayvonni sevadi",
                      "{n}ga qaysi hayvon yoqadi"],
               "ru": ["какое любимое животное у {n}", "какое животное любит {n}", "кого из животных любит {n}",
                      "какое животное нравится {n}"],
               "en": ["what is the favorite animal of {n}", "which animal does {n} love", "what animal does {n} like",
                      "which animal is liked by {n}"]},
}
ANS = {"uz": "Javob: {v}", "ru": "Ответ: {v}", "en": "Answer: {v}"}


def name(rng):
    return "".join(rng.choice(SYL) for _ in range(rng.randint(2, 3))).capitalize()


def main():
    rng = random.Random(42)
    train, test, used = [], [], set()
    langs = ["uz", "ru", "en"]
    for i in range(600):
        n = name(rng)
        while n in used:
            n = name(rng)
        used.add(n)
        kind = ["city", "job", "animal"][i % 3]
        vi = rng.randrange(len(VALUES[kind]))
        learn = langs[i % 3]
        forms = Q[kind][learn]
        same_lang_control = zlib.crc32(n.encode()) % 5 == 0
        train_forms = forms[:3] if same_lang_control else forms
        for f in train_forms:
            train.append({"q": f.format(n=n), "a": ANS[learn].format(v=VALUES[kind][vi][LI[learn]]),
                          "kind": f"fact_{kind}", "lang": learn, "person": n})
        if same_lang_control:
            test.append({"q": forms[3].format(n=n), "answer": VALUES[kind][vi][LI[learn]], "kind": "same_lang",
                         "lang": learn, "learned_in": learn, "person": n, "all_answers": list(VALUES[kind][vi])})
        for ask in langs:
            if ask != learn:
                test.append({"q": rng.choice(Q[kind][ask]).format(n=n), "answer": VALUES[kind][vi][LI[ask]],
                             "kind": "transfer", "lang": ask, "learned_in": learn, "person": n,
                             "all_answers": list(VALUES[kind][vi])})
    out = ROOT / "data/transfer.json"
    out.write_text(json.dumps({"train": train, "test": test}, ensure_ascii=False), encoding="utf-8")
    print(f"{out}: train {len(train)}, test {len(test)} "
          f"(перенос {sum(t['kind'] == 'transfer' for t in test)}, тот же язык {sum(t['kind'] == 'same_lang' for t in test)})")


def score(results):
    """results: [(item, generated)] → {перенос: доля, перенос_любой_язык: доля, тот_же_язык: доля}."""
    from server.selfplay import final_answer
    agg = {"transfer": [0, 0], "transfer_any": [0, 0], "same_lang": [0, 0]}
    for item, gen in results:
        got = (final_answer(gen) or gen).strip().lower()
        ok = got == item["answer"].lower()
        any_ok = got in {a.lower() for a in item["all_answers"]}
        k = item["kind"]
        agg[k][0] += ok
        agg[k][1] += 1
        if k == "transfer":
            agg["transfer_any"][0] += any_ok
            agg["transfer_any"][1] += 1
    return {k: (h / n if n else None) for k, (h, n) in agg.items()}


if __name__ == "__main__":
    main()
