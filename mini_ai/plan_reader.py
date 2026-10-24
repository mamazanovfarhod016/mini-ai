"""plan_reader — ГИБРИД в приложении (24.10.2026): ЧИТАТЕЛЬ (градиент, наш трансформер) + РАССУЖДЕНИЕ (формула, без градиента).

Читатель — наш трансформер, обученный на Kaggle (tools/reader_gsm8k.py): английское условие задачи, числа → слоты [N0] [N1] …,
→ ПЛАН на языке «N0 * N1 ; R0 + N2 ; ANS R1». Исполнение плана — без градиента, шаг за шагом, каждый шаг виден.
Здесь только вывод (numpy, без torch): веса берутся из лучшего доступного прогона (максимум «полный тест GSM8K» в metrics.json).
Сильнейшая версия выбирается автоматически при загрузке: появится лучший прогон — подхватится после перезапуска сервера.
Путь можно задать явно: MINI_AI_READER=/путь/к/папке (там reader.npz + reader_vocab.json + metrics.json).
"""
import os, re, json, glob, math
import numpy as np

RUNS = os.path.expanduser("~/Developer/mini-ai-runs/learning-core")
NUM = re.compile(r"\d+(?:\.\d+)?")
WORDNUM = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())}
WORDNUM.update(thirty=30, forty=40, fifty=50, sixty=60, seventy=70, eighty=80, ninety=90, hundred=100, thousand=1000, dozen=12, twice=2, half=0.5, double=2, triple=3, quarter=0.25)
OPS = {"+": "плюс", "-": "минус", "*": "умножить на", "/": "разделить на"}


def clean(q):
    q = q.replace(",", "").replace("$", " $ ").replace("%", " % ")
    return re.sub(r"\b(" + "|".join(WORDNUM) + r")\b", lambda m: f"{WORDNUM[m.group(1).lower()]:g}", q, flags=re.I)


def numbers(q): return [float(x) for x in NUM.findall(clean(q))]


def slot_tokens(q):
    out = []; k = 0
    for t in re.findall(r"\d+(?:\.\d+)?|[a-z]+(?:'[a-z]+)?|[^\sa-z\d]", clean(q).lower()):
        if NUM.fullmatch(t): out.append(f"N{k}"); k += 1
        else: out.append(t)
    return out


def execute(plan, qn):
    """без градиента: план → шаги с настоящими числами; (ответ, шаги) или (None, причина)"""
    res = []; steps = []; t = list(plan)
    def val(x):
        if x.startswith("N"): i = int(x[1:]); return qn[i] if i < len(qn) else None
        if x.startswith("R"): j = int(x[1:]); return res[j] if j < len(res) else None
        if x.startswith("C"): return float(x[1:])
        return None
    i = 0
    while i < len(t):
        if t[i] == "ANS":
            v = val(t[i + 1]) if i + 1 < len(t) else None
            return (v, steps) if v is not None else (None, "ответ без значения")
        if i + 2 >= len(t): return None, "план оборвался"
        a, op, b = val(t[i]), t[i + 1], val(t[i + 2])
        if a is None or b is None or op not in "+-*/": return None, f"плохой шаг {' '.join(t[i:i+3])}"
        if op == "/" and abs(b) < 1e-12: return None, "деление на 0"
        try: c = {"+": a + b, "-": a - b, "*": a * b, "/": a / b}[op]
        except (ZeroDivisionError, OverflowError): return None, "ошибка арифметики"
        if not math.isfinite(c) or abs(c) > 1e15: return None, "переполнение"
        res.append(c); steps.append((a, op, b, c))
        i += 4 if i + 3 < len(t) and t[i + 3] == ";" else 3
    return (res[-1], steps) if res else (None, "пустой план")


def _ln(x, g, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * g + b


def _gelu(x): return 0.5 * x * (1 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))


def best_run():
    """папка прогона с максимальным «полный тест GSM8K» (acc_full), где лежат веса"""
    env = os.environ.get("MINI_AI_READER")
    if env: return env
    best = None
    for m in glob.glob(os.path.join(RUNS, "reader_*", "metrics.json")) + glob.glob(os.path.join(RUNS, "reader_*", "*", "metrics.json")):
        d = os.path.dirname(m)
        if not (os.path.exists(os.path.join(d, "reader.npz")) and os.path.exists(os.path.join(d, "reader_vocab.json"))): continue
        try: j = json.load(open(m))
        except Exception: continue
        a = j.get("acc_full")
        if a is not None and (best is None or a > best[0]): best = (a, d)
    return best[1] if best else None


class PlanReader:
    def __init__(self, path=None):
        self.path = path or best_run()
        if not self.path: raise FileNotFoundError("нет обученного читателя в " + RUNS)
        z = np.load(os.path.join(self.path, "reader.npz")); self.p = {k: z[k].astype(np.float32) for k in z.files}
        v = json.load(open(os.path.join(self.path, "reader_vocab.json"))); self.cfg = v["cfg"]; self.itos = v["itos"]
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        try: self.metrics = json.load(open(os.path.join(self.path, "metrics.json")))
        except Exception: self.metrics = {}
        self.PAD, self.UNK, self.EOS, self.BOS, self.SEP = range(5)
        plan_tok = {i for w, i in self.stoi.items() if re.fullmatch(r"[NRC][\d.]+|[-+*/;]|ANS", w)}
        self.not_plan = np.array([i for i in range(len(self.itos)) if i not in plan_tok and i != self.EOS])
        d, H = self.cfg["d"], self.cfg["n_head"]; dh = d // H
        theta = self.cfg.get("rope_base", 10000.0) ** (-np.arange(0, dh, 2) / dh)
        ang = np.arange(512)[:, None] * theta[None, :]; self.cos, self.sin = np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)

    def info(self):
        m = self.metrics
        return dict(path=self.path, params=m.get("params"), layers=self.cfg["n_layer"], d=self.cfg["d"], train=m.get("train_plans"),
                    gsm8k_full=m.get("acc_full"), control=m.get("ctrl_full"))

    def _rope(self, x):
        T = x.shape[-2]; c, s = self.cos[:T], self.sin[:T]
        x0, x1 = x[..., 0::2], x[..., 1::2]
        out = np.empty_like(x); out[..., 0::2] = x0 * c - x1 * s; out[..., 1::2] = x0 * s + x1 * c
        return out

    def _logits_last(self, ids):
        p, cfg = self.p, self.cfg; d, H = cfg["d"], cfg["n_head"]; dh = d // H; T = len(ids)
        x = p["wte"][np.array(ids)]
        mask = np.tril(np.ones((T, T), bool))
        for l in range(cfg["n_layer"]):
            h = _ln(x, p[f"{l}.ln1.g"], p[f"{l}.ln1.b"])
            qkv = (h @ p[f"{l}.attn.wqkv"] + p[f"{l}.attn.bqkv"]).reshape(T, 3, H, dh).transpose(1, 2, 0, 3)
            q, k, v = self._rope(qkv[0]), self._rope(qkv[1]), qkv[2]
            S = q @ k.transpose(0, 2, 1) / math.sqrt(dh); S = np.where(mask, S, -1e9)
            S = np.exp(S - S.max(-1, keepdims=True)); A = S / S.sum(-1, keepdims=True)
            y = (A @ v).transpose(1, 0, 2).reshape(T, d)
            x = x + y @ p[f"{l}.attn.wo"] + p[f"{l}.attn.bo"]
            h = _ln(x, p[f"{l}.ln2.g"], p[f"{l}.ln2.b"])
            x = x + _gelu(h @ p[f"{l}.mlp.w1"] + p[f"{l}.mlp.b1"]) @ p[f"{l}.mlp.w2"] + p[f"{l}.mlp.b2"]
        hf = _ln(x[-1], p["lnf.g"], p["lnf.b"])
        return hf @ p["wte"].T + p["bout"]

    def read(self, q, max_new=40):
        """условие → план (жадно; читатель пишет только на языке плана)"""
        s = slot_tokens(q); ids = [self.BOS] + [self.stoi.get(w, self.UNK) for w in s][-160:] + [self.SEP]; out = []
        for _ in range(max_new):
            lg = self._logits_last(ids); lg[[self.PAD, self.BOS, self.SEP, self.UNK]] = -1e9; lg[self.not_plan] = -1e9
            nxt = int(lg.argmax()); ids.append(nxt)
            if nxt == self.EOS: break
            out.append(self.itos[nxt])
        unk = [w for w in s if w not in self.stoi and not re.fullmatch(r"N\d+", w)]
        return out, unk

    def solve(self, q):
        qn = numbers(q)
        plan, unk = self.read(q)
        v, steps = execute(plan, qn)
        return dict(numbers=qn, plan=plan, answer=v, steps=steps if isinstance(steps, list) else [], fail=None if isinstance(steps, list) else steps, unknown_words=unk)


def looks_like_word_problem(q):
    """английская текстовая задача: латиница, ≥2 числа, есть вопрос"""
    if re.search(r"[а-яё]", q.lower()) or not re.search(r"[a-z]{3}", q.lower()): return False
    if len(numbers(q)) < 2 or len(q.split()) < 6: return False
    return q.strip().endswith("?") or bool(re.search(r"\bhow (?:many|much|long|far|old)\b|\bwhat is the total\b", q.lower()))


def looks_like_numbers_statement(q):
    """английское условие с ≥2 числами и действием, но без вопроса («Keyli has 6 apples, her brother ate 2») → переспросить"""
    ql = q.lower()
    if re.search(r"[а-яё]", ql) or "?" in q or looks_like_word_problem(q): return False
    if re.match(r"^\s*(?:i|i'm|i am|my|we)\b", ql): return False          # рассказ о себе — не задача
    return len(numbers(q)) >= 2 and bool(re.search(r"\b(?:has|have|had|ate|eats|gave|gives|bought|buys|sold|sells|lost|found|got|gets|took|spent|costs?|each|left|more|less|times)\b", ql))


def fmt(x): return f"{x:g}" if abs(x - round(x)) > 1e-9 else f"{int(round(x))}"


def say(res, info):
    """ответ своими словами: план → шаги → ответ; честная подпись, что это читатель с ~4 % на GSM8K"""
    acc = info.get("gsm8k_full")
    tag = f"(читатель: наш трансформер {(info.get('params') or 0) / 1e6:.1f} млн параметров, GSM8K {acc:.1%}; GPT-3 ≈ 15 %)" if acc is not None else ""
    if res["answer"] is None:
        return f"Прочитала задачу, но план не сработал: {res['fail']}. План: {' '.join(res['plan']) or '—'}. {tag}"
    lines = [f"{i+1}) {fmt(a)} {OPS[op]} {fmt(b)} = {fmt(c)}" for i, (a, op, b, c) in enumerate(res["steps"])]
    return "Думаю так:\n" + "\n".join(lines) + f"\nОтвет: {fmt(res['answer'])}.\nПроверь меня — я часто ошибаюсь в чтении. {tag}"
