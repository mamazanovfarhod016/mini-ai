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
    try: return float(a.split("####")[-1].strip().replace(",", ""))
    except Exception: return None
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
        return f"C{s}" if s in CONST_OK else None
    body = a.replace(",", "").replace("$", "")
    found = re.findall(r"<<([\d.]+)\s*([-+*/])\s*([\d.]+)\s*=\s*([\d.]+)>>", body) or re.findall(r"(?<![\d.])([\d.]+)\s*([-+*/x×÷])\s*([\d.]+)\s*=\s*([\d.]+)", body.split("####")[0])
    for x, op, y, z in found:
        op = {"x": "*", "×": "*", "÷": "/"}.get(op, op)
        x, y, z = float(x), float(y), float(z); rx, ry = ref(x), ref(y)
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
    p = plan_of(q, a)
    if p: TRAIN.append((slot_tokens(q), p, numbers(q), gold_answer(a), q))
TEST = [(slot_tokens(q), plan_of(q, a), numbers(q), gold_answer(a), q) for q, a in TEST_RAW]
ok_exec = sum(1 for s, p, qn, g, _ in TRAIN if execute(p, qn)[0] is not None and abs(execute(p, qn)[0] - g) < 1e-6)
log(f"планов извлечено {len(TRAIN)}/{len(TRAIN_RAW)} ({len(TRAIN)/len(TRAIN_RAW):.0%}); исполняются верно {ok_exec}/{len(TRAIN)}; тест с планом {sum(1 for t in TEST if t[1])}/{len(TEST)}")
wc = collections.Counter(w for s, p, *_ in TRAIN for w in s + p)
itos = ["<pad>", "<unk>", "<eos>", "<bos>", "<sep>"] + [w for w, _ in wc.most_common(8000)]; stoi = {w: i for i, w in enumerate(itos)}; V = len(itos)
PAD, UNK, EOS, BOS, SEP = range(5); enc = lambda ws: [stoi.get(w, UNK) for w in ws]
PLAN_TOK = {i for w, i in stoi.items() if re.fullmatch(r"[NRC][\d.]+|[-+*/;]|ANS", w)}; NOT_PLAN = [i for i in range(len(itos)) if i not in PLAN_TOK and i != EOS]
def seq(s, p): x = [BOS] + enc(s)[-160:] + [SEP] + enc(p) + [EOS]; k = x.index(SEP); return x, [0] * (k + 1) + [1] * (len(x) - k - 2)
SEQ = [seq(s, p) for s, p, *_ in TRAIN]; log(f"словарь {V}, токенов плана {len(PLAN_TOK)}, макс длина {max(len(x) for x,_ in SEQ)}")
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
rng = np.random.default_rng(0); t_start = time.time(); step = 0; CURVE = []; marks = [m for m in (0.17, 0.33, 0.67, 1.0)]; mi = 0; base_lr = 5e-4; warm = 200
while True:
    for X, M in batches(SEQ, BS, rng):
        frac = (time.time() - t_start) / (MIN * 60); lr = base_lr * min(1.0, (step + 1) / warm) * (0.5 * (1 + math.cos(math.pi * min(1.0, frac))) * 0.9 + 0.1)
        loss, _ = eng.step(X[:, :-1], X[:, 1:], M, lr, drop=0.1); step += 1
        if step % 100 == 0: log(f"[читатель] шаг {step} loss {loss:.3f} {(time.time()-t_start)/60:.1f} мин")
        if mi < len(marks) and frac >= marks[mi]:
            a = acc_of(reader_plan, TEST[:N_TEST_CK]); exact = sum(1 for s, p, qn, g, q in TEST[:N_TEST_CK] if p and read(s) == p) / N_TEST_CK
            CURVE.append(dict(frac=marks[mi], min=round((time.time() - t_start) / 60, 1), step=step, loss=loss, acc=a, exact=exact)); log(f"[кривая] {int(marks[mi]*100)}% времени, шаг {step}: ответ верен {a:.1%}, план дословно {exact:.1%} (контроль-память {ctrl:.1%})"); mi += 1
        if frac >= 1.0: break
    if (time.time() - t_start) / (MIN * 60) >= 1.0: break
# ---------------- полный тест + примеры
t1 = time.time(); full = acc_of(reader_plan, TEST); ctrl_full = acc_of(nn_plan, TEST); log(f"ИТОГ reader: GSM8K полный тест {len(TEST)}: читатель+исполнение {full:.1%} | контроль-память {ctrl_full:.1%} | GPT-3 без подсказок ≈ 15 % | шагов {step}, {(time.time()-t_start)/60:.0f} мин (тест {time.time()-t1:.0f}s)")
with open(os.path.join(OUT, "samples.txt"), "w") as f:
    for s, p, qn, g, q in TEST[:40]:
        plan = read(s); v, st = execute(plan, qn)
        f.write(f"Q: {q}\nЧИСЛА: {[f'{x:g}' for x in qn]}\nПЛАН ЧИТАТЕЛЯ: {' '.join(plan)}\nЭТАЛОН ПЛАНА: {' '.join(p) if p else '-'}\nШАГИ: {st if isinstance(st, list) else '— ' + str(st)}\nОТВЕТ: {v if v is None else f'{v:g}'}  ЭТАЛОН: {g:g}  {'✓' if v is not None and abs(v-g)<1e-6 else '✗'}\n\n")
json.dump(dict(train_plans=len(TRAIN), train_raw=len(TRAIN_RAW), test=len(TEST), V=V, d=D_T, layers=NL, params=n_params(p0), steps=step, minutes=MIN, curve=CURVE, acc_full=full, ctrl_full=ctrl_full, ctrl_ck=ctrl, seconds=time.time() - T0), open(os.path.join(OUT, "metrics.json"), "w"), indent=1, default=float)
np.savez_compressed(os.path.join(OUT, "reader.npz"), **eng.params_numpy()); json.dump(dict(cfg=cfg, itos=itos), open(os.path.join(OUT, "reader_vocab.json"), "w"))
log(f"[done] {time.time()-T0:.0f}s → {OUT}")
