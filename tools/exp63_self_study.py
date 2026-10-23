"""exp63 — САМООБУЧЕНИЕ БЕЗ УЧИТЕЛЯ: она сама задаёт себе вопросы по прочитанному и сама себя проверяет.

Идея (новая структура, не настройка): учитель-человек даёт ~45 исправлений на 200 вопросов — мало. Но у неё есть
проверяемый источник правды — её собственная память фактов. Из каждого прочитанного факта (X —rel→ Y) она делает вопрос
(«What is X called?», «What does X rel?», «What rel Y?»), отвечает на него ВСЕМ конвейером (арбитр, отрывок, deep…),
и сверяет с Y, который ей известен. Промах → «no, it's Y» (всё, что учится от исправлений: solver, cases, reform, growth).
Это RL на проверяемой награде без человека и без градиента. Гипотеза: после самообучения точность на НОВЫХ вопросах
SciQ (которые она никогда не видела и не генерировала) выше, чем у холодной, при нуле человеческих меток.

Критерий (до запуска): mean(self-studied − cold) ≥ +0.025 на 3 срезах, lost ≤ won/2. Контроль: cold; второй контроль —
столько же «вопросов», но с случайными ответами-метками (шум) — чтобы отличить пользу правды от простого прогона.
Запуск: tools/light.sh tools/exp63_self_study.py [slices=2000,5000,8000] [n_self=300] [test=80]
"""
import sys, os, re, json, time, shutil, gc, random, collections
sys.path.insert(0, ".")
import pandas as pd
from mini_ai.experimental import ExperimentalMind, content_words, toks

args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
slices = [int(x) for x in args.get("slices", "2000,5000,8000").split(",")]; N_SELF = int(args.get("n_self", 300)); TEST = int(args.get("test", 80)); CTX = int(args.get("ctx", 200))
MODE = args.get("mode", "both")            # both | self | noise
OUT = os.path.expanduser("~/Developer/mini-ai-runs/learning-core/exp63_self_study.json")
tr = pd.read_parquet("data/school/sciq/data/train-00000-of-00001.parquet").dropna(subset=["support"])
STOP = {"the", "a", "an", "this", "that", "these", "those", "it", "they", "its", "their", "there", "which", "who", "what", "also", "called", "known", "such"}

def hit(o, g):
    a = ((o.get("text") or "") + " " + str(o.get("answer") or "")).lower(); return str(g).lower() in a

def _clean_x(X):
    X = re.sub(r"^(?:if|when|while|because|although|the|a|an|these|those|this|that|most|many|some|all|each|both)\s+", "", X.strip(), flags=re.I)
    X = re.sub(r"\s+(?:usually|often|also|always|sometimes|generally|typically|then|still)$", "", X, flags=re.I)
    return X.strip()

def _clean_y(Y):
    Y = Y.strip().rstrip(".,;")
    if re.search(r"\b(?:in|of|by|to|from|with|for|on|at|into|as|that|which)$", Y, re.I): return None       # оборванный хвост «a compound in»
    Y = re.sub(r"^(?:a|an|the)\s+", "", Y, flags=re.I)
    return Y if 1 <= len(toks(Y)) <= 4 else None

def self_questions(m, limit):
    """вопросы из её же фактов. Берём только факты с содержательными X и Y (2–4 слова, не стоп, не местоимения)."""
    qs = []; seen = set()
    for (Xw, rel, Yw, X, Y, src) in m.kb.facts:
        if src == "user": continue
        X = _clean_x(X); Y = _clean_y(Y)
        if not X or not Y: continue
        if re.match(r"^(?:measured|found|used|made|located|known|called|formed|produced)\b", Y, re.I): continue     # «be measured in joules» — не определение
        # Y для «what is an example of Y» должен быть классом-существительным: не прилагательное/наречие, не оборот с предлогом
        yw = toks(Y); xw = toks(X)
        if any(w in ("so", "very", "right", "same", "important", "different", "on", "in", "at", "several", "many") for w in yw[:1]): continue
        if len(xw) == 1 and xw[0] in ("volume", "it", "one", "which"): continue
        if Y[0].isupper() and len(yw) > 1 and not re.match(r"^[A-Z][a-z]+$", Y.split()[0]): continue
        xs = [w for w in toks(X) if w not in STOP]; ys = [w for w in toks(Y) if w not in STOP]
        if not (1 <= len(xs) <= 3 and 1 <= len(ys) <= 4): continue
        if set(xs) & set(ys): continue
        # X должен быть термином: не начинается с заглавной внутри предложения-обрывка, не наречие, не число, встречается в памяти ≥2 раз
        if re.search(r"\b(?:ly|ing)$", xs[-1]) or any(w.isdigit() for w in xs + ys) or xs[0] in ("most", "many", "some", "other", "these", "all", "each", "both"): continue
        if len(m.kb.by_word.get(xs[-1], [])) < 2: continue
        if ys[-1].endswith(("ly",)) or ys[0] in ("will", "can", "may", "also", "usually", "often", "not"): continue
        key = (X.lower(), rel, Y.lower())
        if key in seen: continue
        seen.add(key)
        r = rel.lower()
        if r in ("be", "is", "are"):
            if re.search(r"\bcalled\b|\bknown as\b", Y, re.I) or len(ys) > 3: continue
            qs.append((f"What is {X}?", Y, "be"))
            if len(yw) <= 2 and not yw[-1].endswith(("ed", "ing")) and m.kb.idf(yw[-1]) < 4.0: qs.append((f"What is an example of {Y}?", X, "be-rev"))   # класс — частое слово
        elif r in ("called", "is called", "known as", "termed"):
            qs.append((f"What is {X} called?", Y, "called"))
        elif r in ("contain", "contains", "produce", "produces", "use", "uses", "have", "has", "cause", "causes", "need", "needs", "form", "forms", "release", "releases", "make", "makes"):
            qs.append((f"What do {X} {r.rstrip('s')}?", Y, "rel"))
    random.Random(63).shuffle(qs); return qs[:limit]

def run_slice(lo, mode):
    rows = list(tr.iloc[lo:lo + CTX + TEST].iterrows()); ctx, test = rows[:CTX], rows[CTX:]
    res = {}
    for variant in (["cold", "self", "noise"] if mode == "both" else ["cold", mode]):
        sd = f"/tmp/e63_{variant}_{lo}"; shutil.rmtree(sd, ignore_errors=True)
        m = ExperimentalMind(school=False, wiki=False, state_dir=sd)
        for _, r in rows: m.kb.read_text(str(r["support"]), "учебник")
        stats = collections.Counter(); t0 = time.time()
        if variant != "cold":
            qs = self_questions(m, N_SELF); rnd = random.Random(lo)
            if lo == slices[0] and variant == "self": print("  sample self-questions:", [(q, g) for q, g, _ in qs[:6]], "| facts", len(m.kb.facts), flush=True)
            for q, gold, kind in qs:
                o = m.ask(q, internet=False)
                if variant == "noise":
                    # контроль: тот же прогон, но «правда» — случайный ответ из другого факта
                    gold = rnd.choice(qs)[1]
                ok = hit(o, gold); stats[("self-hit", kind, ok)] += 1
                if ok: m.ask("yes", internet=False)
                elif o.get("kind") in ("answer", "clarify", "unknown"): m.ask(f"no, it's {gold}", internet=False); stats["corr"] += 1
            stats["n_self"] = len(qs)
        h = 0
        for _, r in test: h += hit(m.ask(r["question"], internet=False), r["correct_answer"])
        acc = h / len(test); res[variant] = dict(acc=acc, self={str(k): v for k, v in stats.items()} if variant != "cold" else None, solver=m.solver.report(), sec=round(time.time() - t0))
        sh = sum(v for k, v in stats.items() if isinstance(k, tuple) and k[2]); sn = sum(v for k, v in stats.items() if isinstance(k, tuple))
        print(f"[{lo}] {variant:5s} test {acc:.3f}  self-q {stats.get('n_self', 0)} self-acc {sh / max(1, sn):.2f} corrections {stats.get('corr', 0)}  {res[variant]['solver'][:60]}  {res[variant]['sec']}s", flush=True)
        del m; gc.collect()
    return res

all_res = {}; t0 = time.time()
for lo in slices:
    all_res[lo] = run_slice(lo, MODE)
    json.dump(all_res, open(OUT, "w"), indent=1, default=str)
d_self = [all_res[lo]["self"]["acc"] - all_res[lo]["cold"]["acc"] for lo in slices if "self" in all_res[lo]]
d_noise = [all_res[lo]["noise"]["acc"] - all_res[lo]["cold"]["acc"] for lo in slices if "noise" in all_res[lo]]
print(f"SELF-STUDY mean {sum(d_self) / max(1, len(d_self)):+.3f}  NOISE-control mean {sum(d_noise) / max(1, len(d_noise)):+.3f}  over {len(slices)} slices ({int(time.time() - t0)}s)", flush=True)
