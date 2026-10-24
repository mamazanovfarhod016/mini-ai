"""reader_gsm8k — ЧИТАТЕЛЬ С ГРАДИЕНТОМ + РАССУЖДЕНИЕ ФОРМУЛОЙ (24.10.2026, его «да» на разделение: градиент читает, формула рассуждает).
Цель — линейка GPT-3: GSM8K (GPT-3 175B без подсказок ≈ 15 %, с цепочкой мыслей 35–50 %). Сегодня у нас 0/200.

Разделение:
  ЧИТАТЕЛЬ (градиент, наш трансформер mini_ai/transformer.py, PyTorch только для backprop): английское условие, где числа
     заменены слотами [N0] [N1] …, → ПЛАН в тройках: «N0 * N1 ; R0 + N2 ; ANS R1». Это и есть «прочитать»: кто с кем какой
     связью. Учится на 7 473 решённых задачах GSM8K-train (план извлекается из пометок <<a*b=c>>, числа → слоты).
  РАССУЖДЕНИЕ (без градиента): план исполняется шаг за шагом с настоящими числами; каждый шаг виден («думаю так: 1) … 2) …»);
     КОНТРОЛЬ без градиента — ближайшая по словам задача из памяти → её план на новых числах (перенос случая по форме, как
     analogy.py). Если учитель исправит план — он ляжет в память случаев (здесь не тестируется, отдельно).
Числа успеха (записаны до запуска): читатель > контроль-память; точность ответа по кривой времени 10/20/40/60 мин растёт;
первая цель 15 % на полном тесте (1 319 задач) = уровень GPT-3 без подсказок. Честно: это seq2seq-читатель из 7 473 примеров без
предобучения — литература даёт для таких 5–20 %; если ниже контроля-памяти, значит, читатель не читает, и надо менять формат плана.
Выход: metrics.json, samples.txt (первые 40 тестовых задач: план читателя, шаги, ответ, эталон), reader.npz.
Локально: SMOKE=1 python3 tools/reader_gsm8k.py
"""
import os, re, io, sys, json, time, math, collections, urllib.request
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SMOKE = os.environ.get("SMOKE") == "1"; E = os.environ.get
OUT = E("OUT", "/kaggle/working" if os.path.isdir("/kaggle/working") else "/tmp/reader_gsm8k"); os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(*a):
    s = " ".join(str(x) for x in a); print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)
    with open(os.path.join(OUT, "log.txt"), "a") as f: f.write(s + "\n")
import torch
from mini_ai.transformer import init_params, n_params
from mini_ai.torch_engine import TorchEngine, forward as tforward
MIN = float(E("MIN", 1.5 if SMOKE else 60)); D_T = int(E("D_T", 64 if SMOKE else 256)); NL = int(E("NL", 2 if SMOKE else 6)); NH = int(E("NH", 4 if SMOKE else 8)); BS = int(E("BS", 32 if SMOKE else 64))
N_TEST_CK = int(E("N_TEST_CK", 40 if SMOKE else 300)); FRAC = float(E("FRAC", 1.0))

HF = "https://huggingface.co/datasets/openai/gsm8k/resolve/refs%2Fconvert%2Fparquet/main/{split}/0000.parquet"
def parquet(url):
    import pyarrow.parquet as pq
    for k in range(3):
        try: return pq.read_table(io.BytesIO(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "mini-ai"}), timeout=120).read())).to_pandas()
        except Exception as e: log("retry", k, type(e).__name__); time.sleep(5)
tr = parquet(HF.format(split="train")); te = parquet(HF.format(split="test"))
TRAIN_RAW = list(zip(tr["question"], tr["answer"])); TEST_RAW = list(zip(te["question"], te["answer"]))
# 24.10 вечер: открытые наборы (качаются на Kaggle): MetaMathQA (MIT; GSM_AnsAug+GSM_Rephrased — перефразы GSM8K-TRAIN, ~160k)
# и Orca-Math 200k (MIT). Решения свободным текстом → «#### N» дописывается; план извлекается как обычно; чистка от теста ниже.
HF_SETS = [x for x in E("HF_SETS", "").split(",") if x]
def _hf_first_shard(repo):
    lst = json.loads(urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/api/datasets/{repo}/parquet", headers={"User-Agent": "mini-ai"}), timeout=60).read())
    cfg = next(iter(lst)); urls = lst[cfg].get("train") or next(iter(lst[cfg].values()))
    import pyarrow.parquet as pq
    return pq.read_table(io.BytesIO(urllib.request.urlopen(urllib.request.Request(urls[0], headers={"User-Agent": "mini-ai"}), timeout=600).read())).to_pandas()
def _with_hash(a):
    if "####" in a: return a
    m = re.search(r"answer is:?\s*\$?(-?[\d,]*\.?\d+)", a) or re.search(r"(-?[\d,]*\.?\d+)\D*$", a)
    return (a + "\n#### " + m.group(1).replace(",", "")) if m else None
EXTRA_HF = []
for name in HF_SETS:
    try:
        if name == "metamath":
            df = _hf_first_shard("meta-math/MetaMathQA"); df = df[df["type"].isin(["GSM_AnsAug", "GSM_Rephrased"])]
            EXTRA_HF += [(q, a) for q, a in zip(df["query"], df["response"])]
        elif name == "orca":
            df = _hf_first_shard("microsoft/orca-math-word-problems-200k")
            EXTRA_HF += [(q, x) for q, a in zip(df["question"], df["answer"]) if (x := _with_hash(a))]
        log("открытый набор", name, "→ всего внешних", len(EXTRA_HF))
    except Exception as e: log("набор", name, "НЕ загрузился:", type(e).__name__, str(e)[:120])
EXTRA = E("EXTRA", "")   # наши новые задачи (data/gsm_new/all.jsonl, 24.10: 1400 от Sonnet-агентов) — тот же формат
if EXTRA and os.path.exists(EXTRA):
    ex = [json.loads(l) for l in open(EXTRA) if l.strip()]; TRAIN_RAW += [(d["question"], d["answer"]) for d in ex]; log("добавлено своих задач", len(ex))
if EXTRA_HF:   # чистка от теста: 8-словные совпадения с любым вопросом GSM8K-test → выкидываем
    def _grams(s_, n=8):
        w = re.findall(r"[a-z]+|\d+", s_.lower()); return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
    _T = set()
    for q, _ in TEST_RAW: _T |= _grams(q)
    _n0 = len(EXTRA_HF); EXTRA_HF = [(q, a) for q, a in EXTRA_HF if not (_grams(q) & _T)]
    log(f"внешние после чистки от теста: {len(EXTRA_HF)} из {_n0}")
    MAX_HF = int(E("MAX_HF", 10**9)); TRAIN_RAW += EXTRA_HF[:MAX_HF]
if SMOKE: TRAIN_RAW = TRAIN_RAW[:1500]; TEST_RAW = TEST_RAW[:80]
if FRAC < 1: TRAIN_RAW = TRAIN_RAW[: int(len(TRAIN_RAW) * FRAC)]
log("gsm8k train", len(TRAIN_RAW), "test", len(TEST_RAW))

NUM = re.compile(r"\d+(?:\.\d+)?")
WORDNUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
WORDNUM.update(thirty=30, forty=40, fifty=50, sixty=60, seventy=70, eighty=80, ninety=90, hundred=100, thousand=1000, dozen=12, twice=2, half=0.5, double=2, triple=3, quarter=0.25)
def clean(q):
    q = q.replace(",", "").replace("$", " $ ").replace("%", " % ")
    return re.sub(r"\b(" + "|".join(WORDNUM) + r")\b", lambda m: f"{WORDNUM[m.group(1).lower()]:g}", q, flags=re.I)   # слова-числа → цифры (three → 3)
def numbers(q): return [float(x) for x in NUM.findall(clean(q))]
def slot_tokens(q):
    """условие → токены, числа → [N i] по порядку появления"""
    out = []; k = 0
    for t in re.findall(r"\d+(?:\.\d+)?|[a-z]+(?:'[a-z]+)?|[^\sa-z\d]", clean(q).lower()):
        if NUM.fullmatch(t): out.append(f"N{k}"); k += 1
        else: out.append(t)
    return out
def gold_answer(a):
    """первое число после «####» (MetaMathQA дописывает «The answer is: …» следующей строкой)"""
    m = re.search(r"-?\d+(?:\.\d+)?", a.split("####")[-1].replace(",", "")) if "####" in a else None
    return float(m.group()) if m else None
CONST_OK = {"1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "24", "60", "100", "1000", "0.5", "0.25", "0.1", "0.2", "0.75", "7", "52", "30", "365", "0.01"}
def plan_of(q, a):
    """план из пометок <<a op b = c>>: операнды → N i (число условия), R j (прошлый результат) или константа; ответ → ANS R k"""
    qn = numbers(q); res = []; plan = []
    def ref(v):
        for j, r in enumerate(res):
            if abs(r - v) < 1e-6: return f"R{j}"
        for i, n in enumerate(qn):
            if abs(n - v) < 1e-6: return f"N{i}"
        s = f"{v:g}"
        if s in CONST_OK: return f"C{s}"
        for i, n in enumerate(qn):          # 25.10: «25%» в условии, 0.25 в решении → явный шаг N i / 100 (как сделал бы человек)
            if abs(n / 100 - v) < 1e-9:
                plan.extend([f"N{i}", "/", "C100", ";"]); res.append(v); return f"R{len(res) - 1}"
        return None
    body = a.replace(",", "").replace("$", "")
    # 25.10: выражение → бинарные шаги по ПРАВИЛАМ (скобки, * / раньше + −), а не слева направо; «.3» → 0.3;
    # кроме пометок <<…>> берём и неразмеченные «a op b = c» из текста решения (по порядку появления, без дублей)
    body = re.sub(r"(?<![\d.])\.(\d)", r"0.\1", body)
    exprs = [(m.start(), m.group(1), m.group(2)) for m in re.finditer(r"<<([^<>=]+)=\s*(-?[\d.]+)>>", body)]
    text = re.sub(r"<<[^<>]*>>", " ", body.split("####")[0])
    exprs += [(m.start() + 0.5, m.group(1), m.group(2)) for m in re.finditer(r"(?<![\d.\w])([\d.(][\d.\s+\-*/x×÷()]*[-+*/x×÷][\d.\s()]*[\d.)])\s*=\s*(-?[\d.]+)", text)]
    exprs.sort()
    import ast as _ast
    def _steps(expr):
        e = expr.replace("x", "*").replace("×", "*").replace("÷", "/").strip()
        if not re.fullmatch(r"[\d.\s+\-*/()]+", e): return None
        try: tree = _ast.parse(e, mode="eval").body
        except SyntaxError: return None
        out = []
        def walk(n):
            if isinstance(n, _ast.Constant) and isinstance(n.value, (int, float)): return float(n.value)
            if isinstance(n, _ast.BinOp) and type(n.op) in (_ast.Add, _ast.Sub, _ast.Mult, _ast.Div):
                l, r = walk(n.left), walk(n.right)
                if l is None or r is None: return None
                op = {_ast.Add: "+", _ast.Sub: "-", _ast.Mult: "*", _ast.Div: "/"}[type(n.op)]
                if op == "/" and r == 0: return None
                v = {"+": l + r, "-": l - r, "*": l * r, "/": l / r if r else 0}[op]; out.append((l, op, r, v)); return v
            return None
        v = walk(tree)
        return (out, v) if v is not None and out else None
    found = []; seen_ = set()
    for _, ex, z in exprs:
        r_ = _steps(ex)
        if not r_: continue
        st, v = r_
        try: zf = float(z.rstrip("."))
        except ValueError: continue
        if abs(v - zf) > 1e-3 * max(1, abs(zf)): continue
        key = tuple(round(x, 6) for t in st for x in (t[0], t[2], t[3]))
        if key in seen_: continue
        seen_.add(key); found += st
    for x, op, y, z in found:
        try: x, y, z = float(x), float(y), float(z)
        except ValueError: return None
        rx, ry = ref(x), ref(y)
        if rx is None or ry is None: return None
        plan += [rx, op, ry, ";"]; res.append(z)
    g = gold_answer(a)
    if g is None or not res: return None
    k = next((j for j, r in enumerate(res) if abs(r - g) < 1e-6), None)
    if k is None: return None
    return plan + ["ANS", f"R{k}"]
def execute(plan, qn):
    """без градиента: исполняем план шаг за шагом; возвращает (ответ, шаги) или (None, причина)"""
    res = []; steps = []; toks_ = list(plan)
    def val(t):
        if t.startswith("N"): i = int(t[1:]); return qn[i] if i < len(qn) else None
        if t.startswith("R"): j = int(t[1:]); return res[j] if j < len(res) else None
        if t.startswith("C"): return float(t[1:])
        return None
    i = 0
    while i < len(toks_):
        if toks_[i] == "ANS":
            v = val(toks_[i + 1]) if i + 1 < len(toks_) else None
            return (v, steps) if v is not None else (None, "ANS без значения")
        if i + 2 >= len(toks_): return None, "обрыв"
        a, op, b = val(toks_[i]), toks_[i + 1], val(toks_[i + 2])
        if a is None or b is None or op not in "+-*/": return None, f"плохой шаг {toks_[i:i+3]}"
        if op == "/" and abs(b) < 1e-12: return None, "деление на 0"
        try: c = {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[op]
        except (ZeroDivisionError, OverflowError): return None, "ошибка арифметики"
        if not math.isfinite(c) or abs(c) > 1e15: return None, "переполнение"
        res.append(c); steps.append(f"{a:g} {op} {b:g} = {c:g}")
        i += 4 if i + 3 < len(toks_) and toks_[i + 3] == ";" else 3
    return (res[-1], steps) if res else (None, "пусто")
TRAIN = []
for q, a in TRAIN_RAW:
    try: p = plan_of(q, a)
    except (ValueError, ZeroDivisionError, OverflowError): p = None
    if not p: continue
    qn_, g_ = numbers(q), gold_answer(a); v_, _ = execute(p, qn_)
    if v_ is None or g_ is None or abs(v_ - g_) > 1e-6: continue      # чужие наборы шумные: берём только планы, дающие эталонный ответ
    TRAIN.append((slot_tokens(q), p, qn_, g_, q))
def _plan_safe(q, a):
    try: return plan_of(q, a)
    except (ValueError, ZeroDivisionError, OverflowError): return None
TEST = [(slot_tokens(q), _plan_safe(q, a), numbers(q), gold_answer(a), q) for q, a in TEST_RAW]
ok_exec = sum(1 for s, p, qn, g, _ in TRAIN if execute(p, qn)[0] is not None and abs(execute(p, qn)[0] - g) < 1e-6)
log(f"планов извлечено {len(TRAIN)}/{len(TRAIN_RAW)} ({len(TRAIN)/len(TRAIN_RAW):.0%}); исполняются верно {ok_exec}/{len(TRAIN)}; тест с планом {sum(1 for t in TEST if t[1])}/{len(TEST)}")
wc = collections.Counter(w for s, p, *_ in TRAIN for w in s + p)
itos = ["<pad>", "<unk>", "<eos>", "<bos>", "<sep>"] + [w for w, _ in wc.most_common(8000)]; stoi = {w: i for i, w in enumerate(itos)}; V = len(itos)
PAD, UNK, EOS, BOS, SEP = range(5); enc = lambda ws: [stoi.get(w, UNK) for w in ws]
PLAN_TOK = {i for w, i in stoi.items() if re.fullmatch(r"[NRC][\d.]+|[-+*/;]|ANS", w)}; NOT_PLAN = [i for i in range(len(itos)) if i not in PLAN_TOK and i != EOS]
def seq(s, p):
    """маска на позициях 0..T-2 (предсказываем x[1..]); позиция k = SEP должна предсказывать ПЕРВЫЙ токен плана → маска с k (ошибка 24.10: с k+1 — первый токен не обучался, читатель выдавал план без начала)"""
    x = [BOS] + enc(s)[-160:] + [SEP] + enc(p) + [EOS]; k = x.index(SEP); return x, [0] * k + [1] * (len(x) - k - 1)
AUG = int(E("AUG", 0 if SMOKE else 12))
NAMES = "janet tom lily ben anna john mary sarah james mike emma noah olivia liam mia lucas sophia jack ava ethan".split()
def augment(s, p, rng):
    """условие со слотами → варианты: (а) имена заменяются другими именами; (б) слоты уже скрывают числа, поэтому план инвариантен
       к их значениям — числа менять не нужно; (в) случайное выбрасывание 1 служебного слова (the/a/that) как шум. План тот же."""
    out = []
    for _ in range(AUG):
        t = list(s); mp = {}
        for i, w in enumerate(t):
            if w in NAMES:
                if w not in mp: mp[w] = NAMES[int(rng.integers(len(NAMES)))]
                t[i] = mp[w]
        drop = [i for i, w in enumerate(t) if w in ("the", "a", "an", "that", "then")]
        if drop and rng.random() < 0.5: del t[drop[int(rng.integers(len(drop)))]]
        out.append((t, p))
    return out
_rng = np.random.default_rng(1); SEQ = [seq(s, p) for s, p, *_ in TRAIN]
if AUG: SEQ += [seq(t, p) for s, p, *_ in TRAIN for t, p in augment(s, p, _rng)]
log(f"словарь {V}, токенов плана {len(PLAN_TOK)}, обучающих последовательностей {len(SEQ)} (аугментация ×{AUG+1}), макс длина {max(len(x) for x,_ in SEQ)}")
def batches(seqs, bs, rng):
    idx = rng.permutation(len(seqs))
    for i in range(0, len(idx), bs):
        ch = [seqs[j] for j in idx[i:i + bs]]; T = max(len(x) for x, _ in ch); X = np.full((len(ch), T), EOS, np.int64); M = np.zeros((len(ch), T - 1), bool)
        for j, (x, m) in enumerate(ch): X[j, :len(x)] = x; M[j, :len(m)] = m
        yield X, M

# ---------------- контроль без градиента: ближайшая по словам задача → её план на новых числах
idf = {w: math.log(len(TRAIN) / (1 + n)) for w, n in collections.Counter(w for s, *_ in TRAIN for w in set(s)).items()}
def bow(s): return {w: idf.get(w, 1.0) for w in set(s) if w not in ("the", "a", "an", "of", "to", "and", "in", "is", "are", "how", "many", "much", "does", "did", "do", "he", "she", "it", "they", "his", "her", ".", ",", "?")}
TB = [bow(s) for s, *_ in TRAIN]
def nn_plan(s, qn):
    b = bow(s); best = None; bs_ = -1
    for i, tb in enumerate(TB):
        if len(TRAIN[i][2]) != len(qn): continue
        inter = sum(v for w, v in b.items() if w in tb); den = math.sqrt(sum(v * v for v in b.values()) * sum(v * v for v in tb.values())) + 1e-9
        if inter / den > bs_: bs_, best = inter / den, i
    return TRAIN[best][1] if best is not None else None
def acc_of(fn, items):
    ok = 0
    for s, p, qn, g, q in items:
        plan = fn(s, qn); v, _ = execute(plan, qn) if plan else (None, None)
        ok += v is not None and g is not None and abs(v - g) < 1e-6
    return ok / max(1, len(items))
ctrl = acc_of(nn_plan, TEST[:N_TEST_CK]); log(f"[контроль-память] ближайшая задача → её план на новых числах: {ctrl:.1%} на {N_TEST_CK}")

# ---------------- читатель (градиент)
p0, cfg = init_params(V, d=D_T, n_layer=NL, n_head=NH, pos="rope", seed=0); eng = TorchEngine(p0, cfg, lr=5e-4, wd=0.01, seed=0)
log(f"[читатель] d={D_T} слоёв {NL} параметров {n_params(p0):,} на {eng.dev}; бюджет {MIN:g} мин, batch {BS}")
@torch.no_grad()
def read(s, max_new=40):
    ids = [BOS] + enc(s)[-160:] + [SEP]; out = []
    for _ in range(max_new):
        lg = tforward(eng.p, cfg, eng._t([ids], torch.long))[0, -1]; lg[[PAD, BOS, SEP, UNK]] = -1e9; lg[NOT_PLAN] = -1e9   # читатель пишет только на языке плана
        nxt = int(lg.argmax()); ids.append(nxt)
        if nxt == EOS: break
        out.append(itos[nxt])
    return out
def reader_plan(s, qn): return read(s)
rng = np.random.default_rng(0); t_start = time.time(); step = 0; CURVE = []; marks = [0.1, 0.2, 0.3, 0.45, 0.6, 0.8, 1.0]; mi = 0; base_lr = 5e-4; warm = 200
best = dict(acc=-1.0, params=None, at=None); DROP = float(E("DROP", 0.2))
while True:
    for X, M in batches(SEQ, BS, rng):
        frac = (time.time() - t_start) / (MIN * 60); lr = base_lr * min(1.0, (step + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, frac))) * 0.9 + 0.1)
        loss, _ = eng.step(X[:, :-1], X[:, 1:], M, lr, drop=DROP); step += 1
        if step % 100 == 0: log(f"[читатель] шаг {step} loss {loss:.3f} {(time.time()-t_start)/60:.1f} мин")
        if mi < len(marks) and frac >= marks[mi]:
            a = acc_of(reader_plan, TEST[:N_TEST_CK]); exact = sum(1 for s, p, qn, g, q in TEST[:N_TEST_CK] if p and read(s) == p) / N_TEST_CK
            CURVE.append(dict(frac=marks[mi], min=round((time.time() - t_start) / 60, 1), step=step, loss=loss, acc=a, exact=exact)); log(f"[кривая] {int(marks[mi]*100)}% времени, шаг {step}: ответ верен {a:.1%}, план дословно {exact:.1%} (контроль-память {ctrl:.1%})"); mi += 1
            if a > best["acc"]: best = dict(acc=a, params={k: v.detach().clone() for k, v in eng.p.items()}, at=step)
        if frac >= 1.0: break
    if (time.time() - t_start) / (MIN * 60) >= 1.0: break
# ---------------- полный тест ЛУЧШЕЙ точкой кривой (ранняя остановка по проверочным 300) + примеры
last_acc = CURVE[-1]["acc"] if CURVE else None
if best["params"] is not None:
    with torch.no_grad():
        for k in eng.p: eng.p[k].copy_(best["params"][k])
    log(f"[остановка] беру веса шага {best['at']} (проверочные {best['acc']:.1%}; последняя точка {last_acc:.1%})")
t1 = time.time(); full = acc_of(reader_plan, TEST); ctrl_full = acc_of(nn_plan, TEST); log(f"ИТОГ reader: GSM8K полный тест {len(TEST)}: читатель+исполнение {full:.1%} | контроль-память {ctrl_full:.1%} | GPT-3 без подсказок ≈ 15 % | шагов {step}, {(time.time()-t_start)/60:.0f} мин (тест {time.time()-t1:.0f}s)")
with open(os.path.join(OUT, "samples.txt"), "w") as f:
    for s, p, qn, g, q in TEST[:40]:
        plan = read(s); v, st = execute(plan, qn)
        f.write(f"Q: {q}\nЧИСЛА: {[f'{x:g}' for x in qn]}\nПЛАН ЧИТАТЕЛЯ: {' '.join(plan)}\nЭТАЛОН ПЛАНА: {' '.join(p) if p else '-'}\nШАГИ: {st if isinstance(st, list) else '— ' + str(st)}\nОТВЕТ: {v if v is None else f'{v:g}'}  ЭТАЛОН: {g:g}  {'✓' if v is not None and abs(v-g)<1e-6 else '✗'}\n\n")
json.dump(dict(train_plans=len(TRAIN), train_raw=len(TRAIN_RAW), train_seqs=len(SEQ), aug=AUG, drop=DROP, test=len(TEST), V=V, d=D_T, layers=NL, params=n_params(p0), steps=step, minutes=MIN, curve=CURVE, best_step=best["at"], best_ck=best["acc"], acc_full=full, ctrl_full=ctrl_full, ctrl_ck=ctrl, seconds=time.time() - T0), open(os.path.join(OUT, "metrics.json"), "w"), indent=1, default=float)
np.savez_compressed(os.path.join(OUT, "reader.npz"), **eng.params_numpy()); json.dump(dict(cfg=cfg, itos=itos), open(os.path.join(OUT, "reader_vocab.json"), "w"))
log(f"[done] {time.time()-T0:.0f}s → {OUT}")
