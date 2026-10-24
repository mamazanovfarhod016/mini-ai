"""Открытые наборы задач: доступны ли и сколько из них наш читатель может взять (план извлекается из решения).
Печатает: набор, строк в первом шарде, доля задач с извлекаемым планом (на 2000 примерах), пример."""
import io, json, re, sys, urllib.request
import pyarrow.parquet as pq
src = open("tools/reader_gsm8k.py").read()
blk = src[src.index("NUM = re.compile"): src.index("TRAIN = []")]
ns = {"re": re, "math": __import__("math")}; exec(blk, ns)
plan_of, execute, numbers, gold_answer = ns["plan_of"], ns["execute"], ns["numbers"], ns["gold_answer"]

def get(url, t=90): return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "mini-ai"}), timeout=t).read()
def shard(repo, cfg=None, split="train"):
    lst = json.loads(get(f"https://huggingface.co/api/datasets/{repo}/parquet"))
    if isinstance(lst, dict):
        cfg = cfg or next(iter(lst)); urls = lst[cfg].get(split) or next(iter(lst[cfg].values()))
    else: urls = lst
    return pq.read_table(io.BytesIO(get(urls[0], 300))).to_pandas(), len(urls)

def probe(name, repo, qcol, acol, conv=None, cfg=None, filt=None):
    try:
        df, nsh = shard(repo, cfg)
    except Exception as e:
        print(f"{name:22} НЕДОСТУПЕН: {type(e).__name__} {str(e)[:80]}"); return
    if filt: df = df[filt(df)]
    n = len(df); ok = 0; ex = None; k = 0
    for q, a in zip(df[qcol][:2000], df[acol][:2000]):
        a = conv(a) if conv else a; k += 1
        try: p = plan_of(q, a) if a else None
        except (ValueError, ZeroDivisionError, OverflowError): p = None
        if p:
            v, _ = execute(p, numbers(q)); g = gold_answer(a)
            if v is not None and g is not None and abs(v - g) < 1e-6:
                ok += 1; ex = ex or (q[:110], " ".join(p))
    print(f"{name:22} шардов {nsh:3}  строк в 1-м {n:7}  план извлекается {ok}/{k} ({ok/max(1,k):.0%})")
    if ex: print(f"{'':22} пример: {ex[0]} → {ex[1]}")

def ans_hash(a):
    """свободное решение → добавляем '#### <число>' из 'The answer is: X' / последнего числа (для MetaMath/Orca)"""
    if "####" in a: return a
    m = re.search(r"answer is:?\s*\$?(-?[\d,.]+)", a) or re.search(r"(-?[\d,]*\.?\d+)\D*$", a)
    return (a + "\n#### " + m.group(1).replace(",", "").rstrip(".")) if m else None

probe("MetaMathQA (GSM)", "meta-math/MetaMathQA", "query", "response", conv=ans_hash, filt=lambda d: d["type"].isin(["GSM_AnsAug", "GSM_Rephrased"]))
probe("Orca-Math 200k", "microsoft/orca-math-word-problems-200k", "question", "answer", conv=ans_hash)
probe("GSM8K-aug (calc)", "MU-NLPC/Calc-gsm8k", "question", "chain", conv=lambda a: re.sub(r"<gadget id=\"calculator\">([^<]*)</gadget>\s*<output>([^<]*)</output>", lambda m: f"<<{m.group(1).replace(' ','')}={m.group(2).strip()}>>", a) + "\n#### 0")
probe("SVAMP", "ChilleD/SVAMP", "question_concat", "Equation", conv=None)
