"""exp65 — СВОБОДНАЯ РЕЧЬ БЕЗ ПАРАМЕТРОВ: собеседник, который учится говорить из разговоров (без градиента, без чужих моделей).

Чем это НЕ является: не говорилка (шаблоны), не нейромодель. Чем является — память реплик + смысл + конструкции:
  1. ПАМЯТЬ РЕПЛИК. Каждый обмен (что сказал собеседник → что ответили) — случай. Код реплики = смысл слов (meaning.py,
     без градиента) + скелет (вопрос/утверждение/просьба, лицо, время, тема). Новый вход → ближайшие случаи по смыслу.
  2. КОНСТРУКЦИИ. Из ответов-случаев извлекаются конструкции с дырками: «тебе [ADJ] перед [N] — [совет]». Слова
     собеседника и его имя/факты заполняют дырки → ответ «по-разному», а не копия.
  3. РОСТ ОТ РАЗГОВОРА. Собеседник говорит «нет, лучше так: …» или просто отвечает своей фразой → новый случай,
     вес старого падает. Собеседник сам учит её говорить — тем же механизмом, что учит фактам.
  4. ПАМЯТЬ О СОБЕСЕДНИКЕ. Имя, возраст, что любит, что говорил — факты в ядре (reader), ответы ссылаются на них.
Гипотеза и критерий (до запуска): на отложенных диалогах (data/dialogs/ru.jsonl, test split) ответ модели оценивается
по (а) тематической связности с репликой собеседника (смысл-близость ответ↔эталон), (б) разнообразию (доля уникальных
ответов на 50 входов), (в) использованию памяти о собеседнике (имя/факт вспомнен там, где эталон вспоминает).
Контроли: случайный ответ из корпуса; ближайший ответ без конструкций (копия). Старая говорилка — число из exp56: on-topic 0.35.
Запуск: tools/light.sh tools/exp65_free_speech.py
"""
import sys, os, re, json, random, collections, time
sys.path.insert(0, ".")
import numpy as np

OUT = os.path.expanduser("~/Developer/mini-ai-runs/learning-core/exp65_free_speech.json")
RU_BASE = os.path.expanduser("~/Developer/mini-ai-runs/learning-core/meaning_ru.npz")

def toks(s): return re.findall(r"[а-яёa-z0-9]+", s.lower().replace("ё", "е"))
STOP = set("и а но или же ли бы не ни это этот эта эти тот та то те он она оно они его ее их ему ей им в во на с со у к ко о об от до за из по при для без через между как что кто где когда почему зачем да нет ну вот уже еще ещё очень так там тут здесь тоже также мне тебе нам вам мой моя мое мои твой наш ваш свой я ты мы вы меня тебя себя есть был была было были будет".split())

# ---------- данные ----------
dialogs = [json.loads(l) for l in open("data/dialogs/ru.jsonl", encoding="utf-8")]
random.Random(65).shuffle(dialogs); n_test = max(20, len(dialogs) // 5)
train, test = dialogs[n_test:], dialogs[:n_test]
print(f"[data] dialogs {len(dialogs)}  train {len(train)}  test {len(test)}", flush=True)

# ---------- смысл слов без градиента (meaning.py, русская база из ruwiki/TinyStories если есть; иначе — учим из самих диалогов) ----------
from mini_ai.meaning import Meaning
M = Meaning(None)
if os.path.exists(RU_BASE): M.load(RU_BASE); print(f"[meaning] ru base loaded: {len(M.sem)} words", flush=True)
for d in train:
    for t in d["turns"]: M.read(t["text"])
print(f"[meaning] after dialogs: {len(M.sem)} words", flush=True)

def sem(words):
    v = M.sent_vec([w for w in words if w not in STOP]) if words else None
    return v

# ---------- скелет реплики (грамматика — закрытая таблица, не знание) ----------
def skeleton(s):
    t = toks(s); k = []
    if "?" in s or (t and t[0] in ("что", "кто", "где", "когда", "почему", "зачем", "как", "какой", "какая", "какие", "сколько", "помнишь", "знаешь")): k.append("Q")
    if any(w in ("я", "мне", "меня", "мой", "моя", "мои", "у меня") for w in t): k.append("I")
    if any(w in ("ты", "тебе", "тебя", "твой", "твоя") for w in t): k.append("YOU")
    if any(w.endswith(("ал", "ала", "ил", "ила", "ел", "ела", "ул", "ула")) for w in t): k.append("PAST")
    if any(w in ("завтра", "буду", "будет", "скоро", "потом", "хочу", "собираюсь", "планирую") for w in t): k.append("FUT")
    if any(w in ("боюсь", "страшно", "грустно", "плохо", "устал", "устала", "обидно", "злюсь", "переживаю", "волнуюсь") for w in t): k.append("NEG")
    if any(w in ("рад", "рада", "ура", "круто", "класс", "отлично", "здорово", "наконец", "получилось", "сдал", "сдала") for w in t): k.append("POS")
    return tuple(k)

# ---------- 1. память реплик ----------
class Case:
    __slots__ = ("inp", "out", "vec", "skel", "ctx_vec", "name", "w", "cons")
    def __init__(self, inp, out, ctx, name):
        self.inp, self.out, self.name = inp, out, name; self.vec = sem(toks(inp)); self.skel = skeleton(inp)
        self.ctx_vec = sem([w for s in ctx for w in toks(s)]) if ctx else None; self.w = 1.0

NAME_RX = re.compile(r"(?:меня зовут|я|моё имя|мое имя)\s+([А-ЯЁ][а-яё]+)")
def extract_name(s):
    m = NAME_RX.search(s); return m.group(1) if m else None

cases = []
for d in train:
    turns = d["turns"]; name = None; ctx = []
    for i in range(len(turns) - 1):
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "bot":
            name = extract_name(turns[i]["text"]) or name
            cases.append(Case(turns[i]["text"], turns[i + 1]["text"], ctx[-2:], name))
        ctx.append(turns[i]["text"])
print(f"[cases] {len(cases)} input→reply cases", flush=True)
CV = np.stack([c.vec if c.vec is not None else np.zeros(4096, np.float32) for c in cases])

# ---------- 2. конструкции: ответ с дырками ----------
# дырка [U:w] — слово ответа, которое есть во входе (собеседник его сказал) → при переносе подставляем слово нового входа
# того же смысла/позиции; [NAME] — имя; остальное — каркас (фразовая структура ответа).
def make_construction(c):
    inw = set(toks(c.inp)); out = []
    for w in re.findall(r"[а-яёА-ЯЁa-z0-9]+|[^\w\s]", c.out):
        lw = w.lower().replace("ё", "е")
        if c.name and w == c.name: out.append("[NAME]")
        elif lw in inw and lw not in STOP and len(lw) > 3: out.append(f"[U:{lw}]")
        else: out.append(w)
    return out

for c in cases: c.cons = make_construction(c)   # type: ignore[attr-defined]
n_slots = sum(1 for c in cases for t in c.cons if t.startswith("["))
print(f"[constructions] {len(cases)} with {n_slots} slots ({n_slots / max(1, len(cases)):.2f}/reply)", flush=True)

# ---------- 3. генерация: ближайшие случаи по смыслу+скелету → конструкция → заполнение из входа ----------
def fill(cons, inp, name, src_case):
    inw = [w for w in toks(inp) if w not in STOP and len(w) > 3]; out = []
    for t in cons:
        if t == "[NAME]": out.append(name or "")
        elif t.startswith("[U:"):
            w0 = t[3:-1]
            # слово нового входа, ближайшее по смыслу к слову-дырке (а не первое попавшееся)
            best, bs = None, -1.0
            for w in inw:
                s = M.similar(w0, w)
                if s is not None and s > bs: best, bs = w, s
            out.append(best or (inw[0] if inw else w0))
        else: out.append(t)
    s = " ".join(x for x in out if x); s = re.sub(r"\s+([,.!?…])", r"\1", s); s = re.sub(r"\s+", " ", s).strip()
    return s[:1].upper() + s[1:] if s else s

def reply(inp, ctx, name, k=5, rng=None, used=None):
    v = sem(toks(inp)); sk = skeleton(inp)
    if v is None: v = np.zeros(4096, np.float32)
    sims = CV @ v
    # скелет: совпадение типа реплики (вопрос/я/прошлое/эмоция) — грамматика поверх смысла
    bonus = np.array([0.08 * len(set(sk) & set(c.skel)) - 0.05 * len(set(sk) ^ set(c.skel)) for c in cases], np.float32)
    score = sims + bonus + np.array([0.02 * c.w for c in cases], np.float32)
    if used: score = score - np.array([0.3 if i in used else 0.0 for i in range(len(cases))], np.float32)   # разнообразие: не повторять один случай
    top = np.argsort(-score)[:k]
    i = int(top[0]) if rng is None else int(rng.choice(top[:3]))
    c = cases[i]
    text = fill(c.cons, inp, name, c)
    return text, i, float(sims[i])

# ---------- 4. оценка на отложенных диалогах ----------
def topic_sim(a, b):
    va, vb = sem(toks(a)), sem(toks(b)); return float(va @ vb) if va is not None and vb is not None else 0.0

rng = random.Random(1); nprng = np.random.default_rng(1)
res = collections.defaultdict(list); examples = []; used = set(); all_out = []
rand_pool = [c.out for c in cases]
for d in test:
    turns = d["turns"]; name = None; ctx = []
    for i in range(len(turns) - 1):
        if turns[i]["role"] == "user" and turns[i + 1]["role"] == "bot":
            inp, gold = turns[i]["text"], turns[i + 1]["text"]; name = extract_name(inp) or name
            ours, ci, s = reply(inp, ctx, name, rng=nprng, used=used); used.add(ci); all_out.append(ours)
            copy = cases[int(np.argmax(CV @ (sem(toks(inp)) if sem(toks(inp)) is not None else np.zeros(4096))))].out
            rnd = rng.choice(rand_pool)
            res["ours_topic"].append(topic_sim(ours, gold)); res["copy_topic"].append(topic_sim(copy, gold)); res["rand_topic"].append(topic_sim(rnd, gold))
            res["ours_in"].append(topic_sim(ours, inp)); res["rand_in"].append(topic_sim(rnd, inp))
            if name and name in gold: res["name_gold"].append(1); res["name_ours"].append(1 if name in ours else 0)
            if len(examples) < 10: examples.append((inp, gold, ours))
        ctx.append(turns[i]["text"])
div = len(set(all_out)) / max(1, len(all_out))
out = {k: round(float(np.mean(v)), 3) for k, v in res.items() if v}; out["diversity"] = round(div, 3); out["n"] = len(all_out)
print(f"[eval] n={out['n']}  topic↔gold: ours {out['ours_topic']}  copy {out['copy_topic']}  random {out['rand_topic']}", flush=True)
print(f"[eval] on-topic↔input: ours {out['ours_in']}  random {out['rand_in']}   diversity {out['diversity']}   name recall {out.get('name_ours', 'n/a')} (gold uses name in {len(res['name_gold'])} replies)", flush=True)
for inp, gold, ours in examples: print(f"  Ю: {inp}\n  ИИ: {ours}\n  (эталон: {gold})")
json.dump(dict(out, examples=examples), open(OUT, "w"), ensure_ascii=False, indent=1)
