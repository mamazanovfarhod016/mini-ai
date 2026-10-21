"""Эксперимент 38: ЧЕСТНАЯ ЦИФРА — WebQuestions (Berant et al. 2013), 2032 test-вопросов, не мои.
Память: факты из дампа Simple Wikipedia. Ответ — по фактам с рамкой ВНУТРИ Y:
  «discovered by Z» → agent Z; «of Z» / «in Z» → Z; иначе Y целиком.
Один счёт, настроенный на TRAIN (первые 300), измеренный на TEST (все 2032). Метрика WebQ: ответ-строка
содержит любой из gold-ответов (или gold содержит ответ) — мягкое совпадение, как принято для WebQ (F1/acc).
Контроль: BM25 по предложениям (та же память, без фактов) → ответ = предложение целиком, та же метрика.
Честно: WebQ — Freebase-вопросы (актёры, фильмы, "what did X do") — Simple Wikipedia многого не содержит.
Поэтому отдельно считаем «покрытие»: доля вопросов, где хотя бы один gold встречается в памяти вообще.
"""
import sys, os, json, time, re, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from mini_ai.memory_v2 import toks, content_words, C
from mini_ai.binary_store import pack, hamming

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
STORE = os.environ.get("STORE", f"{OUT}/simplewiki_store")
idf = json.load(open(f"{STORE}/idf.json")); facts = [json.loads(l) for l in open(f"{STORE}/facts.jsonl")]
sents = [json.loads(l)[0] for l in open(f"{STORE}/text.jsonl")]
print(f"память: {len(sents)} предложений, {len(facts)} фактов", flush=True)

def bag(ws):
    cw = content_words(ws); return sum(idf.get(w, 1.0) * C(w) for w in cw) if cw else None

# --- рамка внутри Y: подответы ---
BY = re.compile(r"\bby\s+((?:[A-Z][\w'\-\.]*\s*){1,4})"); OF = re.compile(r"\b(?:of|in|from|for)\s+((?:[A-Z][\w'\-]*\s*){1,4})")
def sub_answers(Y):
    out = [Y]
    for rx in (BY, OF):
        for m in rx.finditer(Y): out.append(m.group(1).strip())
    return out

t0 = time.time(); keys, keep = [], []
for i, f in enumerate(facts):
    X, rel, Y = f[0], f[1], f[2]
    v = bag(toks(X) + toks(rel.replace("_", " ")) + toks(Y))
    if v is not None: keys.append(pack(v)[0]); keep.append(i)
K = np.stack(keys); facts = [facts[i] for i in keep]
print(f"ключи {K.nbytes/1e6:.0f} МБ, {time.time()-t0:.0f}s", flush=True)

QW = {"what", "who", "where", "when", "how", "many", "much", "which", "does", "do", "did", "is", "was", "are", "were", "s", "the"}
def norm(s): return " ".join(toks(s))
def match(ans, golds):
    a = norm(ans)
    return any(g and (norm(g) in a or (len(norm(g)) > 3 and a in norm(g))) for g in golds)

def answer(q, pre=400, topk=3):
    qw = [w for w in toks(q) if w not in QW]; qset = set(content_words(qw)); qidf = sum(idf.get(w, 1.0) for w in qset) or 1
    qv = bag(qw)
    if qv is None: return []
    cand = np.argsort(hamming(K, pack(qv)[0]))[:pre]; scored = []
    for i in cand:
        X, rel, Y = facts[i][0], facts[i][1], facts[i][2]; xw, yw = set(toks(X)), set(toks(Y))
        cov = sum(idf.get(w, 1.0) for w in qset & (xw | set(toks(rel.replace("_", " "))) | yw)) / qidf
        if len(qset & xw) > len(qset & yw) or (len(qset & xw) == len(qset & yw) and xw & qset):
            for a in sub_answers(Y): scored.append((cov - 0.005 * len(toks(a)), a, f"{X} {rel} {Y}"))
        else: scored.append((cov, X, f"{X} {rel} {Y}"))
    scored.sort(key=lambda t: -t[0]); return scored[:topk]

# --- BM25 по предложениям (контроль) ---
tok_s = [content_words(toks(s)) for s in sents]; N = len(sents); avg = sum(map(len, tok_s)) / N
dfc = collections.Counter(w for ws in tok_s for w in set(ws)); inv = collections.defaultdict(list)
for i, ws in enumerate(tok_s):
    for w in set(ws): inv[w].append(i)
def bm25(q, k1=1.5, b=0.75, topk=3):
    qs = content_words(toks(q)); sc = collections.defaultdict(float)
    for w in qs:
        if w not in inv: continue
        idf_w = np.log(1 + (N - dfc[w] + 0.5) / (dfc[w] + 0.5))
        for i in inv[w]:
            tf = tok_s[i].count(w); sc[i] += idf_w * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(tok_s[i]) / avg))
    return [sents[i] for i, _ in sorted(sc.items(), key=lambda t: -t[1])[:topk]]

def run(df, label, limit=None):
    rows = df.head(limit) if limit else df; n = len(rows)
    f1 = f3 = b1 = b3 = cov = 0; t0 = time.time()
    for _, r in rows.iterrows():
        q, golds = r["question"], list(r["answers"])
        top = answer(q); hits = [match(a, golds) for _, a, _ in top]
        f1 += bool(hits and hits[0]); f3 += any(hits)
        bt = bm25(q); bh = [match(s, golds) for s in bt]; b1 += bool(bh and bh[0]); b3 += any(bh)
        cov += any(norm(g) in " ".join(s.lower() for s in bt) for g in golds) or any(hits)
    dt = time.time() - t0
    print(f"{label}: n={n}  ФАКТЫ R@1 {f1/n*100:.1f}% R@3 {f3/n*100:.1f}%  |  BM25-предложения R@1 {b1/n*100:.1f}% R@3 {b3/n*100:.1f}%  "
          f"| gold где-то в top-3 {cov/n*100:.1f}%  | {dt/n*1000:.0f} мс/вопрос", flush=True)
    return dict(n=n, f1=f1/n, f3=f3/n, b1=b1/n, b3=b3/n, cov=cov/n)

train = pd.read_parquet("data/webq/data/train-00000-of-00001.parquet"); test = pd.read_parquet("data/webq/data/test-00000-of-00001.parquet")
rt = run(train, "TRAIN (300, для настройки)", 300)
print("--- примеры (test) ---")
for _, r in test.head(12).iterrows():
    top = answer(r["question"]); a = top[0][1] if top else "-"
    print(f"  {'OK ' if top and match(a, r['answers']) else 'MISS'} {r['question']} → «{a[:40]}» (gold: {list(r['answers'])[0][:30]})  [{top[0][2][:60] if top else ''}]")
rs = run(test, "TEST (2032)")
json.dump(dict(train=rt, test=rs, store=STORE, facts=len(facts), sents=len(sents)), open(f"{OUT}/exp38_webq.json", "w"), indent=1)
