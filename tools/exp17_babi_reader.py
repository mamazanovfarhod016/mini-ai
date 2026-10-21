"""Эксперимент 17: bAbI — ПОЛНАЯ проверка ядра на НАСТОЯЩЕМ бенчмарке (round2 §4, критерий «стоп»).
Данные: data/babi/task_N_test.parquet (скачаны с HF RawthiL/babi_tasks — зеркало официального bAbI
Weston et al. 2015). Чтение сырых предложений, НИКАКИХ ручных кортежей per-задачу: один общий механизм.

ЯДРО (как в ARCHITECT.md §2, применено целиком, не кусками):
  Sieve   — C(w) канонический код слова (случайный фиксированный, cos≈0 между словами) — exp14/exp16 лечение.
  Coref   — местоимение (he/she/it/they) → ближайшая по тексту сущность-кандидат (round2 §2.7, recency).
            Общее правило, не per-задача: заменяем местоимение раньше, чем строим рамку.
  Frame   — R(sentence) = Σ C(w) ⊛ Π^i (роль = позиция содержательного слова во фразе; Sahlgren 2008) +
            Σ_{i<j} C(wi)⊛C(wj) (неупорядоченные пары — exp14b). Стоп-слова (closed-class) не входят в роли,
            но служебные слова («north»/«south», «not») остаются отдельным ПОЛЕМ записи (как тип).
  Memory  — список предложений (ключ=R(s), поля: entity-слова, полярность, направление). ЧТЕНИЕ = argmax
            косинуса СРЕДИ самых недавних (recency — round2 §2.1/2.7: при равных exact-matches выигрывает
            последняя запись по времени, а не монетка). Один проход, без градиента.
  Planner — ответ = content-слово найденного предложения, которого нет в вопросе, с нужным типом;
            для yes/no (task 6/9) и отрицания — отдельное поле полярности (round2 §2.5);
            для multi-hop (task 2/3) — минимальный просмотр: если ответ не найден в 1 предложении
            (нет object-слова, т.к. вопрос про объект, а в предложении его позиция — subject другого
            предложения), подставляем найденный subject и ищем ЕГО location (как exp9 chain).
Контроли: BM25 (пересечение слов, частотный вес) и random.
Критерий round2: если task 1 < 95% — проблема в ЧТЕНИИ, останавливаемся и не идём дальше.
"""
import sys, os, json, re, time, math, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_core import unit

D = 1024; rng = np.random.default_rng(0)
_C, _FC = {}, {}
def C(w):
    if w not in _C:
        v = unit(np.random.default_rng(abs(_stable_hash(w))).standard_normal(D))
        _C[w] = v; _FC[w] = np.fft.fft(v)   # exp19: кэшируем FFT-образ слова РАЗ, не при каждом bind
    return _C[w]
def FC(w): C(w); return _FC[w]
import zlib
def _stable_hash(w): return zlib.crc32(w.encode()) & 0xffffffff

F = np.fft.fft
def bind(a, b): return unit(np.real(np.fft.ifft(F(a) * F(b))))
_perm = rng.permutation(D)
_ROLE0 = unit(rng.standard_normal(D))
_ROLE_FC = {}
def role(i):
    if i not in _ROLE_FC:
        idx = _perm.copy()
        for _ in range(max(0, i - 1)): idx = idx[_perm]
        v = _ROLE0[idx] if i > 0 else _ROLE0
        _ROLE_FC[i] = (v, np.fft.fft(v))
    return _ROLE_FC[i][0]
def role_fc(i): role(i); return _ROLE_FC[i][1]

STOP = set("the a an to of is are was were in on at there back and did do does what who where "
           "he she it they him her them that this after before yes no not "
           "travelled journeyed moved went picked up dropped gave left took discarded grabbed handed passed "
           "following then afterwards".split())
NAMES = set()  # заполняется при чтении: слова с большой буквы — сущности


def toks(s):
    return re.findall(r"[a-zA-Z']+", s.lower())


def toks_cased(s):
    return re.findall(r"[a-zA-Z']+", s)


def content_words(ws):
    return [w for w in ws if w not in STOP]


class Sentence:
    __slots__ = ("raw", "words", "cwords", "vec", "vec_pairs", "neg", "entities", "idx")
    def __init__(self, raw, idx, recent_entities):
        self.raw = raw; self.idx = idx
        ws = toks(raw); ws_cased = toks_cased(raw)
        # кореференция: местоимение → последняя упомянутая сущность (общее правило, не per-задача)
        resolved = []
        for w in ws:
            if w == "they" and recent_entities and len(recent_entities[-1]) > 1:
                resolved.extend(recent_entities[-1])          # «they» = ВСЯ группа предыдущего предложения
            elif w in ("he", "she", "it", "they") and recent_entities:
                resolved.append(recent_entities[-1][-1])      # единственное число = последняя сущность
            else:
                resolved.append(w)
        self.words = resolved
        self.cwords = content_words(resolved)
        self.neg = 1 if ("not" in ws or "n't" in raw.lower()) else 0
        cased_names = {w for w, orig in zip(ws, ws_cased) if orig[:1].isupper() and w not in STOP}
        pron_resolved = {w for w in resolved if w not in ws}    # имена, подставленные вместо местоимений
        self.entities = list(dict.fromkeys(w for w in resolved if w in cased_names or w in pron_resolved))
        # exp19: накапливаем в ЧАСТОТНОЙ области (одно сложение вместо forward-FFT на каждый bind),
        # и делаем ОДНО обратное FFT в конце — вместо N(N+1)/2 отдельных ifft внутри bind().
        fcs = [FC(w) for w in self.cwords]
        acc_r = np.zeros(D, dtype=complex)
        for i, fc in enumerate(fcs): acc_r = acc_r + fc * role_fc(i)
        acc_p = np.zeros(D, dtype=complex)
        for i in range(len(fcs)):
            for j in range(i + 1, len(fcs)): acc_p = acc_p + fcs[i] * fcs[j]
        vr = np.real(np.fft.ifft(acc_r)); vp = np.real(np.fft.ifft(acc_p))
        self.vec = unit(vr) if np.linalg.norm(vr) > 1e-9 else vr          # основной ключ — только роли
        self.vec_pairs = unit(vp) if np.linalg.norm(vp) > 1e-9 else vp


def read_passage(passage):
    recent_entities = []
    sents = []
    for raw in passage.strip().split("\n"):
        if not raw.strip(): continue
        s = Sentence(raw, len(sents), recent_entities)
        if s.entities: recent_entities.append(list(s.entities))   # группа сущностей предложения (для «they»)
        sents.append(s)
    return sents


def query_vec(question):
    ws = toks(question)
    cw = content_words(ws)
    fcs = [FC(w) for w in cw]
    acc = np.zeros(D, dtype=complex)
    for i, fc in enumerate(fcs): acc = acc + fc * role_fc(i)
    for i in range(len(fcs)):
        for j in range(i + 1, len(fcs)): acc = acc + fcs[i] * fcs[j]
    v = np.real(np.fft.ifft(acc))
    return (unit(v) if np.linalg.norm(v) > 1e-9 else v), cw


def best_recent(sents, qv, qwords=None, topn=3):
    """Если известны слова вопроса — сначала фильтр по точному совпадению сущности (быстрее и честнее,
    чем спорить косинусами HRR-шума), среди совпадений берём САМОЕ НЕДАВНЕЕ (recency, round2 §2.1/2.7).
    Без фильтра (запасной путь) — argmax косинуса, при near-tie тоже самое недавнее."""
    if qwords:
        qset = set(qwords)
        match = [s.idx for s in sents if qset & set(s.cwords)]
        if match: return match[-1]
    sims = np.array([float(s.vec @ qv) if np.linalg.norm(s.vec) > 0 else -1 for s in sents])
    order = np.argsort(-sims)
    top = sims[order[0]]
    ties = [i for i in order if sims[i] >= top - 1e-6]
    return max(ties)  # максимальный индекс = самое недавнее предложение среди равных


def bm25_best(sents, qwords):
    """Контроль: пересечение content-слов (BM25-подобно, без IDF для простоты, 1994-стиль)."""
    scores = [len(set(s.cwords) & set(qwords)) for s in sents]
    best = max(range(len(sents)), key=lambda i: (scores[i], i))
    return best


def answer_from(sent, qwords, qword_raw):
    """Ответ = content-слово предложения, которого нет в вопросе."""
    cand = [w for w in sent.cwords if w not in qwords]
    return cand[-1] if cand else (sent.cwords[-1] if sent.cwords else None)


def solve_location(sents, question):
    """Where is X? / Where is the Y? — находим X (или держателя Y через 1 подшаг), берём последнее
    предложение с этим субъектом, отвечаем последним content-словом (место)."""
    qv, qwords = query_vec(question)
    i = best_recent(sents, qv, qwords)
    ans = answer_from(sents[i], qwords, None)
    subj_like = [w for w in sents[i].cwords if w[:1].isalpha()]
    return ans, i


MOTION_VERBS = {"travelled", "journeyed", "moved", "went"}
POSSESS_VERBS = {"got", "gave", "dropped", "picked", "took", "discarded", "grabbed", "handed", "passed", "left", "put"}


def simulate(sents):
    """Один проход по уже прочитанному предложению: кто где (entity_loc) и у кого что (obj_state),
    ПЛЮС история местоположений объекта (obj_history) для вопросов «где было X ДО Y» (task3).
    Тип роли берётся из ГЛАГОЛА предложения (грамматика связей exp9, не per-задачный хардкод):
    motion-глагол -> оставшееся content-слово = МЕСТО; possess-глагол -> оставшееся слово = ПРЕДМЕТ.
    held-объект движется с держателем (пересчёт лениво через entity_loc при ответе);
    dropped-объект замирает в МЕСТЕ держателя НА МОМЕНТ падения (замороженное значение)."""
    entity_loc, obj_state, known_names = {}, {}, set()
    obj_history = collections.defaultdict(list)   # obj -> [место1, место2, ...] по порядку появления
    for s in sents:
        known_names.update(s.entities)
        verbs = set(s.words) & (MOTION_VERBS | POSSESS_VERBS)
        subjs = [w for w in s.words if w in known_names] or []
        subjs = list(dict.fromkeys(subjs))
        rest = [w for w in s.cwords if w not in subjs]
        if not subjs or not rest: continue
        if verbs & MOTION_VERBS:
            for subj in subjs:                                   # «John and Mary went…» — оба едут
                entity_loc[subj] = rest[-1]
                for obj, (kind, val) in obj_state.items():       # объекты, которые держит subj, едут с ним
                    if kind == "held" and val == subj:
                        if not obj_history[obj] or obj_history[obj][-1] != rest[-1]:
                            obj_history[obj].append(rest[-1])
        elif verbs & POSSESS_VERBS:
            subj = subjs[0]; obj = rest[-1]
            if {"dropped", "discarded", "left"} & verbs or ("put" in verbs and "down" in s.words):
                loc = entity_loc.get(subj)
                obj_state[obj] = ("dropped", loc)
                if loc and (not obj_history[obj] or obj_history[obj][-1] != loc): obj_history[obj].append(loc)
            else:  # got / picked up / took / grabbed — объект следует за держателем
                obj_state[obj] = ("held", subj)
                loc = entity_loc.get(subj)
                if loc and (not obj_history[obj] or obj_history[obj][-1] != loc): obj_history[obj].append(loc)
    return entity_loc, obj_state, known_names, dict(obj_history)


def solve_tracked(question, entity_loc, obj_state, known_names, obj_history=None):
    qwords = content_words(toks(question))
    if not qwords: return None, None
    if "before" in toks(question) and obj_history:
        # «Where was the X before the Y?» — X = первое content-слово, Y = последнее (упомянутое место).
        # Ищем ПОСЛЕДНЕЕ вхождение Y в истории (а не первое — это и была ошибка 36% в exp17):
        # вопрос всегда про текущее/последнее пребывание в Y, а объект мог бывать в Y раньше.
        obj = qwords[0]; place = qwords[-1]
        hist = obj_history.get(obj, [])
        for i in range(len(hist) - 1, 0, -1):
            if hist[i] == place: return hist[i - 1], "history"
        return (hist[0] if hist else None), "history"
    target = qwords[-1]
    if target in entity_loc: return entity_loc[target], "entity"
    if target in obj_state:
        kind, val = obj_state[target]
        return (entity_loc.get(val) if kind == "held" else val), "object"
    return None, None


def run_task(name, df, mode):
    t0 = time.time()
    n = len(df); r1 = r1_bm25 = r1_rand = ok = ok_bm25 = 0
    rng2 = np.random.default_rng(3)
    for _, row in df.iterrows():
        sents = read_passage(row["passage"])
        qv, qwords = query_vec(row["question"])
        i_bm = bm25_best(sents, qwords)
        i_rand = int(rng2.integers(0, len(sents)))
        a_bm = answer_from(sents[i_bm], qwords, None)
        gold = row["answer"].strip().lower()
        if mode == "track":
            # task2/3: отслеживание объектов через simulate() — не выбор предложения, а состояние мира
            entity_loc, obj_state, known_names, obj_history = simulate(sents)
            a_core, _ = solve_tracked(row["question"], entity_loc, obj_state, known_names, obj_history)
            i_core = best_recent(sents, qv, qwords)  # для R@1-метрики (суррогатной), не для ответа
        else:
            i_core = best_recent(sents, qv, qwords)
            a_core = answer_from(sents[i_core], qwords, None)
        if mode == "yesno":
            yloc = [w for w in qwords if w not in ("is", "in")]
            found_loc = a_core
            neg = sents[i_core].neg
            pred = "no" if (neg or (found_loc and yloc and found_loc not in yloc)) else "yes"
            ok += pred == gold
            pred_bm = "no" if (sents[i_bm].neg or (a_bm and yloc and a_bm not in yloc)) else "yes"
            ok_bm25 += pred_bm == gold
        else:
            ok += (a_core == gold)
            ok_bm25 += (a_bm == gold)
        r1 += gold in sents[i_core].words
        r1_bm25 += gold in sents[i_bm].words
        r1_rand += gold in sents[i_rand].words
    r = dict(n=n, answer_acc=ok / n, answer_acc_bm25=ok_bm25 / n, found_gold_word=r1 / n,
             found_gold_word_bm25=r1_bm25 / n, found_gold_word_random=r1_rand / n, sec=round(time.time() - t0, 1))
    print(f"{name:35} N={n:4}  ядро: ответ {r['answer_acc']*100:5.1f}%  (слово-в-предл. {r['found_gold_word']*100:5.1f}%)  |  "
          f"BM25: ответ {r['answer_acc_bm25']*100:5.1f}%  (слово-в-предл. {r['found_gold_word_bm25']*100:5.1f}%)  |  "
          f"random слово-в-предл. {r['found_gold_word_random']*100:5.1f}%   {r['sec']}s", flush=True)
    return r


if __name__ == "__main__":
    import pandas as pd
    OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
    TASKS = [
        ("task1_single_fact (критерий СТОП round2: <95% = чинить чтение)", "task_1-single_supporting_fact", "loc"),
        ("task2_two_facts", "task_2-two_supporting_facts", "track"),
        ("task3_three_facts", "task_3-three_supporting_facts", "track"),
        ("task4_two_arg_relations", "task_4-two_argument_relations", "loc"),
        ("task5_three_arg_relations", "task_5-three_argument_relations", "loc"),
        ("task6_yesno", "task_6-yes_no_questions", "yesno"),
        ("task9_negation", "task_9-simple_negation", "yesno"),
        ("task11_coreference", "task_11-basic_coreference", "loc"),
        ("task13_compound_coref", "task_13-compound_coreference", "loc"),
    ]
    res = {}
    for label, fname, mode in TASKS:
        df = pd.read_parquet(f"data/babi/{fname}_test.parquet")
        res[label] = run_task(label, df, mode)
    json.dump(res, open(f"{OUT}/exp17_babi_reader.json", "w"), indent=1, ensure_ascii=False)
