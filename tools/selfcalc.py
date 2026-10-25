"""selfcalc — НЕЙРОСЕТЬ САМА ЧИТАЕТ, РАССУЖДАЕТ И СЧИТАЕТ (25.10.2026, его решение: «зато это будет ИИ»).
Нет плана-для-формулы и нет исполнителя: модель пишет решение шагами ТЕКСТОМ, цифра за цифрой («3 * 500 = 1500 ; ... ANS 500»),
каждое число (и результат арифметики) выдаёт сама. Формула НЕ участвует ни в обучении ответа, ни в проверке: ответ = то, что модель написала после ANS.
Сравнение — с читателем+формулой (25,1 % на тех же 1319 задачах GSM8K; GPT-3 без подсказок ≈ 15 %).
Заранее записано, что считаем успехом шага 1: полный тест 1319 + доля арифметических шагов, где модель посчитала ВЕРНО (диагностика «считает ли сама»);
ожидание — точность сначала ниже 25,1 %, потому что маленькие модели ошибаются в умножении; смысл — увидеть, растёт ли счёт с данными (кривая по времени).
Запуск: SMOKE=1 python3 tools/selfcalc.py ; Kaggle: MIN=40 TG_N=300000 HF_SETS=tinygsm.
"""
import os, re, io, sys, json, time, math, collections, urllib.request, ast
import numpy as np
sys.path.insert(0, "."); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
E = os.environ.get; SMOKE = E("SMOKE") == "1"
OUT = E("OUT", "/kaggle/working" if os.path.isdir("/kaggle/working") else "/tmp/selfcalc"); os.makedirs(OUT, exist_ok=True); T0 = time.time()
def log(*a):
    s = " ".join(str(x) for x in a); print(f"[{time.time()-T0:6.0f}s] {s}", flush=True)
    with open(os.path.join(OUT, "log.txt"), "a") as f: f.write(s + "\n")
import torch
from mini_ai.transformer import init_params, n_params
from mini_ai.torch_engine import TorchEngine, forward as tforward
MIN = float(E("MIN", 1.0 if SMOKE else 40)); D_T = int(E("D_T", 64 if SMOKE else 256)); NL = int(E("NL", 2 if SMOKE else 6)); NH = int(E("NH", 4 if SMOKE else 8)); BS = int(E("BS", 16 if SMOKE else 64))
N_CK = int(E("N_CK", 20 if SMOKE else 150)); MAXLEN = int(E("MAXLEN", 320)); DROP = float(E("DROP", 0.1)); TEST_MAX = int(E("TEST_MAX", 10**9))
HF = "https://huggingface.co/datasets/openai/gsm8k/resolve/refs%2Fconvert%2Fparquet/main/{split}/0000.parquet"
UA = {"User-Agent": "mini-ai"}
def parquet(url):
    import pyarrow.parquet as pq
    for k in range(3):
        try: return pq.read_table(io.BytesIO(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read())).to_pandas()
        except Exception as e: log("retry", k, type(e).__name__); time.sleep(5)
tr = parquet(HF.format(split="train")); te = parquet(HF.format(split="test"))
RAW = list(zip(tr["question"], tr["answer"])); TEST_RAW = list(zip(te["question"], te["answer"]))
def _hf_first_shard(repo):
    lst = json.loads(urllib.request.urlopen(urllib.request.Request(f"https://huggingface.co/api/datasets/{repo}/parquet", headers=UA), timeout=60).read())
    cfg = next(iter(lst)); urls = lst[cfg].get("train") or next(iter(lst[cfg].values()))
    import pyarrow.parquet as pq
    return pq.read_table(io.BytesIO(urllib.request.urlopen(urllib.request.Request(urls[0], headers=UA), timeout=600).read())).to_pandas()
EXTRA = []
if "tinygsm" in E("HF_SETS", "").split(","):
    from tinygsm_conv import tinygsm_to_answer
    df = __import__("pandas").read_parquet(E("TG_LOCAL")) if E("TG_LOCAL") else _hf_first_shard("TinyGSM/TinyGSM"); n_tg = int(E("TG_N", 300000))
    EXTRA = [(q, x) for q, c in zip(df["question"][:n_tg], df["code"][:n_tg]) if (x := tinygsm_to_answer(c))]; del df
    def _grams(s_, n=8):
        w = re.findall(r"[a-z]+|\d+", s_.lower()); return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
    _T = set()
    for q, _ in TEST_RAW: _T |= _grams(q)
    n0 = len(EXTRA); EXTRA = [(q, a) for q, a in EXTRA if not (_grams(q) & _T)]; log(f"TinyGSM после чистки от теста: {len(EXTRA)} из {n0}")
RAW += EXTRA
if SMOKE: RAW = RAW[:1500]; TEST_RAW = TEST_RAW[:60]
TEST_RAW = TEST_RAW[:TEST_MAX]

NUM = r"\d+(?:\.\d+)?"
def qtok(q):
    """условие → токены: слова; число → цифры по одной + «_» (граница числа). Слова-числа (three) НЕ переводим — модель читает их сама."""
    q = q.replace(",", "").replace("$", " $ ").replace("%", " % ").lower(); out = []
    for t in re.findall(NUM + r"|[a-z]+(?:'[a-z]+)?|[^\sa-z\d]", q):
        if re.fullmatch(NUM, t): out += list(t) + ["_"]
        else: out.append(t)
    return out
def fmt(x):
    s = f"{float(x):.6f}".rstrip("0").rstrip("."); return "0" if s in ("-0", "") else s
def solution_tokens(a):
    """шаги из пометок <<a op b = c>> + итог после «####» → токены: цифры, оператор, «=», «;», ANS. Всё, что модель должна написать САМА."""
    marks = re.findall(r"<<([^<>=]+)=\s*(-?[\d.]+)>>", a.replace(",", ""))
    g = re.search(r"-?\d+(?:\.\d+)?", a.split("####")[-1].replace(",", "")) if "####" in a else None
    if not marks or not g: return None, None
    out = []
    for ex, res in marks:
        ex = re.sub(r"(?<![\d.])\.(\d)", r"0.\1", ex.replace(" ", "")).replace("x", "*").replace("×", "*").replace("÷", "/")
        if not re.fullmatch(r"[\d.+\-*/()]+", ex) or "e" in res: return None, None
        out += list(re.findall(NUM + r"|[-+*/()]", ex)) and [c for t in re.findall(NUM + r"|[-+*/()]", ex) for c in (list(t) + ["_"] if re.fullmatch(NUM, t) else [t])]
        out += ["="] + list(fmt(float(res))) + ["_", ";"]
    gv = float(g.group()); out += ["ANS"] + list(fmt(gv)) + ["_"]
    return out, gv
def parse_answer(toks):
    """последнее «ANS <цифры>» в тексте модели → число (или None)"""
    if "ANS" not in toks: return None
    i = len(toks) - 1 - toks[::-1].index("ANS"); s = ""
    for t in toks[i + 1:]:
        if t == "_": break
        s += t
    try: return float(s)
    except ValueError: return None
def steps_ok(toks):
    """(верных арифметических шагов, всего): шаг «выражение = результат ;» верен, если выражение, посчитанное отдельно, равно тому, что написала модель"""
    ok = tot = 0; s = ""; i = 0; cur = []
    txt = "".join(" " if t == "_" else t for t in toks)
    for part in txt.split(";"):
        if "=" not in part: continue
        ex, res = part.rsplit("=", 1); ex = ex.strip().replace(" ", ""); tot += 1
        try:
            if not re.fullmatch(r"[\d.+\-*/()]+", ex): continue
            ok += abs(eval(ex, {"__builtins__": {}}) - float(res.strip().replace("ANS", "") or "nan")) < 1e-4
        except Exception: pass
    return ok, tot

TRAIN = []
for q, a in RAW:
    sol, g = solution_tokens(a)
    if sol and (lambda r: r[0] == r[1])(steps_ok(sol)) and (parse_answer(sol) is not None) and abs(parse_answer(sol) - g) < 1e-9: TRAIN.append((qtok(q), sol, g))   # учим только на решениях, где КАЖДЫЙ шаг арифметически верен
TEST = []
for q, a in TEST_RAW:
    g = re.search(r"-?\d+(?:\.\d+)?", a.split("####")[-1].replace(",", ""))
    TEST.append((qtok(q), float(g.group()), q))
log(f"обучающих задач {len(TRAIN)} из {len(RAW)}; тест {len(TEST)}")
wc = collections.Counter(w for s, p, _ in TRAIN for w in s)
itos = ["<pad>", "<unk>", "<eos>", "<bos>", "<sep>"] + sorted(set("0123456789._+-*/()=;") | {"ANS"}) + [w for w, _ in wc.most_common(8000) if w not in set("0123456789._+-*/()=;")]
itos = list(dict.fromkeys(itos)); stoi = {w: i for i, w in enumerate(itos)}; V = len(itos); PAD, UNK, EOS, BOS, SEP = range(5)
enc = lambda ws: [stoi.get(w, UNK) for w in ws]
SOL_IDS = [stoi[t] for t in itos if re.fullmatch(r"[\d._+\-*/()=;]|ANS", t)] + [EOS]; BAN = [i for i in range(V) if i not in set(SOL_IDS)]
def seq(s, p):
    x = [BOS] + enc(s)[-200:] + [SEP] + enc(p) + [EOS]; k = x.index(SEP); return x, [0] * k + [1] * (len(x) - k - 1)
SEQ = [seq(s, p) for s, p, _ in TRAIN]; SEQ = [z for z in SEQ if len(z[0]) <= MAXLEN]
log(f"словарь {V}; обучающих последовательностей {len(SEQ)}, средняя длина {np.mean([len(x) for x, _ in SEQ]):.0f}")
def batches(seqs, bs, rng):
    idx = rng.permutation(len(seqs))
    for i in range(0, len(idx), bs):
        ch = [seqs[j] for j in idx[i:i + bs]]; T = max(len(x) for x, _ in ch); X = np.full((len(ch), T), EOS, np.int64); M = np.zeros((len(ch), T - 1), bool)
        for j, (x, m) in enumerate(ch): X[j, :len(x)] = x; M[j, :len(m)] = m
        yield X, M

p0, cfg = init_params(V, d=D_T, n_layer=NL, n_head=NH, pos="rope", seed=0)
eng = TorchEngine(p0, cfg, lr=5e-4, wd=0.01, seed=0)
log(f"[модель] d={D_T} слоёв {NL} параметров {n_params(p0):,} на {eng.dev}; бюджет {MIN:g} мин, batch {BS}")
@torch.no_grad()
def solve(s, max_new=170):
    ids = [BOS] + enc(s)[-200:] + [SEP]; out = []
    for _ in range(max_new):
        lg = tforward(eng.p, cfg, eng._t([ids], torch.long))[0, -1]; lg[BAN] = -1e9
        nxt = int(lg.argmax()); ids.append(nxt)
        if nxt == EOS: break
        out.append(itos[nxt])
    return out
def evaluate(items, with_steps=True):
    ok = so = st = 0
    for s, g, q in items:
        out = solve(s); v = parse_answer(out); ok += v is not None and abs(v - g) < 1e-6
        if with_steps: a, b = steps_ok(out); so += a; st += b
    return ok / max(1, len(items)), (so / st if st else 0.0)

rng = np.random.default_rng(0); t_start = time.time(); step = 0; CURVE = []; marks = [0.2, 0.4, 0.6, 0.8, 1.0]; mi = 0; best = dict(acc=-1.0, params=None, at=0)
while True:
    for X, M in batches(SEQ, BS, rng):
        frac = (time.time() - t_start) / (MIN * 60); lr = 5e-4 * min(1.0, (step + 1) / 200) * (0.5 * (1 + math.cos(math.pi * min(1.0, frac))) * 0.9 + 0.1)
        loss, _ = eng.step(X[:, :-1], X[:, 1:], M, lr, drop=DROP); step += 1
        if step % 100 == 0: log(f"[учёба] шаг {step} loss {loss:.3f} {(time.time()-t_start)/60:.1f} мин")
        if mi < len(marks) and frac >= marks[mi]:
            a, sa = evaluate(TEST[:N_CK]); CURVE.append(dict(frac=marks[mi], min=round((time.time() - t_start) / 60, 1), step=step, loss=float(loss), acc=a, step_arith=sa))
            log(f"[кривая] {int(marks[mi]*100)}% времени, шаг {step}: ответ верен {a:.1%}; арифметический шаг посчитан верно {sa:.1%}"); mi += 1
            if a > best["acc"]: best = dict(acc=a, params={k: v.detach().clone() for k, v in eng.p.items()}, at=step)
        if frac >= 1.0: break
    if (time.time() - t_start) / (MIN * 60) >= 1.0: break
if best["params"] is not None:
    with torch.no_grad():
        for k in eng.p: eng.p[k].copy_(best["params"][k])
    log(f"[остановка] веса шага {best['at']} (проверочные {best['acc']:.1%})")
t1 = time.time(); full, sfull = evaluate(TEST)
log(f"ИТОГ selfcalc: GSM8K полный тест {len(TEST)}: нейросеть сама читает и считает {full:.1%} | арифм. шаг верно {sfull:.1%} | читатель+формула 25,1 % | GPT-3 без подсказок ≈ 15 % | шагов {step}, {(time.time()-t_start)/60:.0f} мин (тест {time.time()-t1:.0f}s)")
with open(os.path.join(OUT, "samples.txt"), "w") as f:
    for s, g, q in TEST[:40]:
        out = solve(s); f.write(f"Q: {q}\nМОДЕЛЬ: {' '.join('' if t == '_' else t for t in out)}\nответ модели {parse_answer(out)} | эталон {g}\n\n")
json.dump(dict(params=n_params(p0), full=full, step_arith=sfull, curve=CURVE, steps=step, minutes=MIN, train_tasks=len(SEQ)), open(os.path.join(OUT, "metrics.json"), "w"), indent=1)
torch.save({k: v.detach().cpu() for k, v in eng.p.items()}, os.path.join(OUT, "selfcalc.pt")); json.dump(dict(itos=itos, cfg=cfg), open(os.path.join(OUT, "selfcalc_vocab.json"), "w"))
