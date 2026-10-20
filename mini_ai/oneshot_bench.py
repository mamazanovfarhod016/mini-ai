"""Бенчмарк «усвоение с первого раза» (one-shot facts).

Генерируем N новых фактов с выдуманными именами и числами, которых модель не видела, в uz/ru/en,
и вопрос к каждому. Три условия:
  (a) none      — факт не показан, вопрос задан «в лоб»            (ожидаем ~0 %)
  (b) context   — факт стоит в промпте перед вопросом               (верхняя планка чтения)
  (c) memory    — факт записан в память одним проходом, при вопросе достаётся сам (ядро)
Скоринг: ответ модели → первое число / слово; exact match с эталоном.
"""
import random, re

NAMES_UZ = ["Zarina", "Bekzod", "Dilnoza", "Sardor", "Madina", "Jasur", "Nilufar", "Otabek", "Gulnora", "Sherzod",
            "Kamola", "Rustam", "Sevara", "Ulug'bek", "Mohira", "Davron", "Lola", "Temur", "Nodira", "Aziz"]
NAMES_RU = ["Зарина", "Бекзод", "Дильноза", "Сардор", "Мадина", "Жасур", "Нилуфар", "Отабек", "Гульнора", "Шерзод",
            "Камола", "Рустам", "Севара", "Улугбек", "Мохира", "Даврон", "Лола", "Темур", "Нодира", "Азиз"]
NAMES_EN = ["Zarina", "Bekzod", "Dilnoza", "Sardor", "Madina", "Jasur", "Nilufar", "Otabek", "Gulnora", "Sherzod",
            "Kamola", "Rustam", "Sevara", "Ulugbek", "Mohira", "Davron", "Lola", "Temur", "Nodira", "Aziz"]
OBJ = {  # (uz, ru, en) — вещи, которые можно считать
    "olma": ("olma", "яблок", "apples"), "kitob": ("kitob", "книг", "books"), "qalam": ("qalam", "ручек", "pens"),
    "daftar": ("daftar", "тетрадей", "notebooks"), "koptok": ("koptok", "мячей", "balls"),
}
CITY = ["Varna", "Osh", "Tartu", "Riga", "Bursa", "Lviv", "Kazan", "Batumi", "Ganja", "Namangan"]


def make_facts(n=100, seed=7):
    rng = random.Random(seed)
    facts = []
    for i in range(n):
        lang = ["uz", "ru", "en"][i % 3]
        k = rng.randrange(20)
        kind = "count" if rng.random() < 0.7 else "city"
        if kind == "count":
            num = rng.randrange(2, 60)
            obj = rng.choice(list(OBJ))
            if lang == "uz":
                nm = NAMES_UZ[k]; fact = f"{nm}da {num} ta {OBJ[obj][0]} bor."; q = f"{nm}da nechta {OBJ[obj][0]} bor?"
            elif lang == "ru":
                nm = NAMES_RU[k]; fact = f"У {nm} {num} {OBJ[obj][1]}."; q = f"Сколько {OBJ[obj][1]} у {nm}?"
            else:
                nm = NAMES_EN[k]; fact = f"{nm} has {num} {OBJ[obj][2]}."; q = f"How many {OBJ[obj][2]} does {nm} have?"
            gold = str(num)
        else:
            city = rng.choice(CITY)
            if lang == "uz":
                nm = NAMES_UZ[k]; fact = f"{nm} {city} shahrida yashaydi."; q = f"{nm} qaysi shaharda yashaydi?"
            elif lang == "ru":
                nm = NAMES_RU[k]; fact = f"{nm} живёт в городе {city}."; q = f"В каком городе живёт {nm}?"
            else:
                nm = NAMES_EN[k]; fact = f"{nm} lives in {city}."; q = f"Where does {nm} live?"
            gold = city
        facts.append({"id": i, "lang": lang, "kind": kind, "name": nm, "fact": fact, "q": q, "gold": gold})
    return facts


def extract(answer, kind):
    a = answer.strip()
    if kind == "count":
        m = re.search(r"-?\d+", a)
        return m.group(0) if m else ""
    for c in CITY:
        if c.lower() in a.lower():
            return c
    return a.split()[0].strip(".,!") if a.split() else ""


def score(answers, facts):
    ok = 0
    rows = []
    for f, a in zip(facts, answers):
        got = extract(a, f["kind"])
        hit = got.lower() == f["gold"].lower()
        ok += hit
        rows.append((f, a, got, hit))
    return ok / max(1, len(facts)), rows
