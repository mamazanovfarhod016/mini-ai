"""Эксперимент 18: НАПРАВЛЕННАЯ рамка для task4/5 (обход провала exp17: 35%/62%, хуже BM25).
Диагноз exp17: Frame (роль=позиция по content-словам + неупорядоченные пары) теряет НАПРАВЛЕНИЕ
отношения — «hallway east of bathroom» и «bathroom east of hallway» дают почти тот же вектор.
Мешок/пары не умеют различать подлежащее и дополнение предлога. Фильтрация словами тут не поможет —
нужно явно развести роли (SUBJ, REL, OBJ) по ПОЗИЦИИ относительно служебных слов «is/of» и «to» —
это закрытый, маленький набор связок (как BOS/SEP в токенизаторе), не per-факт хардкод.

ГЛАВНАЯ ЧАСТЬ (по просьбе «проведи обучение, не выдумывай руками»): таблица противоположных
направлений (east<->west, north<->south) НЕ ПРОПИСАНА руками — она ВЫВЕДЕНА статистикой по TRAIN
сплиту (900 историй), так же как в exp9 таблица типов выводилась из памяти (field_learned), без
единого градиентного шага: считаем, для каких пар relation-слов в одной истории выполняется
(A,r1,B) и (B,r2,A) одновременно — это и есть «инверсия», обучение = подсчёт частот на train.

task5 (giver/item/receiver) — тот же приём: разбор по закрытому набору связок (gave/handed/passed/to),
роли SUBJ/VERB/ITEM/RECV по позиции относительно «to»; ответ — недостающий слот записи, совпадающей
по ВСЕМ известным слотам вопроса, последняя по времени (recency, как в exp17 для task2/3).
"""
import sys, os, re, json, collections, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"


# ---------- task4: разбор по связкам is / of (закрытый грамматический класс, не per-факт) ----------
def parse_spatial_fact(sentence):
    m = re.match(r"the (.+?) is (\w+) of the (.+?)\.?$", sentence.strip().lower())
    return (m.group(1), m.group(2), m.group(3)) if m else None


def parse_spatial_question(q):
    q = q.strip().lower().rstrip("?")
    m = re.match(r"what is (\w+) of the (.+)$", q)          # Form A: «What is REL of Y?» -> ищем SUBJ
    if m: return "A", m.group(1), m.group(2)
    m = re.match(r"what is the (.+?) (\w+) of$", q)          # Form B: «What is Y REL of?» -> ищем OBJ
    if m: return "B", m.group(2), m.group(1)
    return None, None, None


def learn_inverse_table(train_df):
    """Обучение (без градиента!): используем СУПЕРВИЗИЮ train (вопрос+ответ), а не совпадения внутри
    одной истории (там relations случайны и не глобально согласованы — проверено, даёт мусор).
    Если вопрос формы A/B с relation r1 не находит прямого факта, но известен GOLD-ответ, и в фактах
    есть обратная тройка (y,r2,gold) [форма A] или (gold,r2,y) [форма B] — это голос «r1 инверсия r2».
    Побеждает самая частая пара на каждый r1. Один проход по train, чистый подсчёт — не градиент."""
    votes = collections.Counter()
    for _, row in train_df.iterrows():
        facts = [parse_spatial_fact(s) for s in row["passage"].strip().split("\n") if s.strip()]
        facts = [f for f in facts if f]
        form, rel, y = parse_spatial_question(row["question"])
        gold = row["answer"].strip().lower()
        if form == "A":
            for a, r, b in facts:
                if a == y and b == gold and r != rel: votes[(rel, r)] += 1
        elif form == "B":
            for a, r, b in facts:
                if a == gold and b == y and r != rel: votes[(rel, r)] += 1
    table = {}
    rels = {r for pair in votes for r in pair}
    for r1 in rels:
        best = max(((r2, c) for (rr1, r2), c in votes.items() if rr1 == r1), default=(None, 0), key=lambda x: x[1])
        if best[0]: table[r1] = best[0]
    return table, votes


def solve_spatial(sents, question, inv_table):
    facts = [parse_spatial_fact(s) for s in sents]
    facts = [f for f in facts if f]
    form, rel, y = parse_spatial_question(question)
    if form is None: return None
    if form == "A":   # ищем X: (X, rel, y)
        for a, r, b in reversed(facts):
            if r == rel and b == y: return a
        inv = inv_table.get(rel)                             # не нашли прямо — пробуем через инверсию
        if inv:
            for a, r, b in reversed(facts):
                if r == inv and a == y: return b
    else:             # form B: ищем Z: (y, rel, Z)
        for a, r, b in reversed(facts):
            if r == rel and a == y: return b
        inv = inv_table.get(rel)
        if inv:
            for a, r, b in reversed(facts):
                if r == inv and b == y: return a
    return None


# ---------- task5: разбор по связкам gave/handed/passed/to (закрытый класс глаголов передачи) ----------
TRANSFER_VERBS = {"gave", "handed", "passed"}
GET_VERBS = {"picked", "got", "grabbed", "took"}


def parse_transfer_fact(sentence):
    s = sentence.strip().lower().rstrip(".")
    toks = s.split()
    verbs_here = [v for v in TRANSFER_VERBS if v in toks]
    if verbs_here and "to" in toks:
        vi = toks.index(verbs_here[0]); ti = toks.index("to")
        giver = toks[0]
        item = " ".join(w for w in toks[vi + 1:ti] if w not in ("up", "the", "there"))
        recv = toks[ti + 1]
        return dict(giver=giver, item=item, recv=recv)
    return None


def solve_transfer(sents, question):
    facts = [parse_transfer_fact(s) for s in sents]
    facts = [f for f in facts if f]
    q = question.strip().lower().rstrip("?")
    toks = q.split()
    want = "giver" if toks[0] == "who" and "give" not in toks[1:3] and "did" not in toks else None
    # распознаём форму вопроса по закрытому набору слов who/what/did/give/to/received
    known = {}
    if "received" in toks:
        want = "recv"; known["item"] = toks[-1]
    elif toks[0] == "who" and "give" in toks:
        want = "giver"
        if "did" in toks:   # Who did X give Z to?
            known["giver"] = toks[2]
            tail = toks[4:-1] if toks[-1] == "to" else toks[4:]
            known["item"] = " ".join(w for w in tail if w != "the")
            want = "recv"
        else:               # Who gave Z [to Y]?
            item_part = [w for w in toks[toks.index("give") + 1:] if w != "the"]
            if "to" in item_part:
                ti = item_part.index("to"); known["item"] = " ".join(item_part[:ti]); known["recv"] = item_part[ti + 1]
            else:
                known["item"] = " ".join(item_part)
    elif toks[0] == "what" and "give" in toks:
        want = "item"
        gi = toks.index("give")
        known["giver"] = toks[2] if toks[1] == "did" else toks[toks.index("did") + 1]
        if "to" in toks: known["recv"] = toks[-1]
    for f in reversed(facts):
        if all(f.get(k) == v for k, v in known.items()):
            return f.get(want)
    return None


def run(name, test_fname, train_fname, kind):
    import pandas as pd
    t0 = time.time()
    inv_table = {}
    if kind == "spatial":
        train_df = pd.read_parquet(f"data/babi/{train_fname}")
        inv_table, votes = learn_inverse_table(train_df)
        print(f"  выучена таблица противоположностей на train ({len(train_df)} историй): {inv_table}", flush=True)
    df = pd.read_parquet(f"data/babi/{test_fname}")
    ok = 0; parsed_fail = 0
    for _, row in df.iterrows():
        sents = [s for s in row["passage"].strip().split("\n") if s.strip()]
        gold = row["answer"].strip().lower()
        if kind == "spatial":
            a = solve_spatial(sents, row["question"], inv_table)
        else:
            a = solve_transfer(sents, row["question"])
        if a is None: parsed_fail += 1
        ok += (a == gold)
    n = len(df)
    print(f"{name:30} N={n:4}  НОВОЕ (направленная рамка): {ok/n*100:5.1f}%   (не распарсено {parsed_fail/n*100:4.1f}%)  {time.time()-t0:.1f}s", flush=True)
    return dict(n=n, acc=ok / n, unparsed=parsed_fail / n, inv_table=inv_table)


if __name__ == "__main__":
    res = {}
    res["task4"] = run("task4_two_arg (было 34.9%, BM25 44.3%)", "task_4-two_argument_relations_test.parquet",
                        "task_4-two_argument_relations_train.parquet", "spatial")
    res["task5"] = run("task5_three_arg (было 61.9%, BM25 77.9%)", "task_5-three_argument_relations_test.parquet",
                        "task_5-three_argument_relations_train.parquet", "transfer")
    json.dump(res, open(f"{OUT}/exp18_directed_frame.json", "w"), indent=1, ensure_ascii=False)
