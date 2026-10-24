"""Собрать все наши задачи (part_a..n + пачки part_n_b*), проверить каждую строку, убрать дубли и совпадения с тестом GSM8K → clean2.jsonl"""
import glob, io, json, os, re, urllib.request
import pyarrow.parquet as pq
os.chdir("/home/hermes/Developer/mini-ai/data/gsm_new")
files = sorted(f for f in glob.glob("part_*.jsonl") if re.fullmatch(r"part_[a-n](_b\d+)?\.jsonl", f))
if os.path.exists("part_n.jsonl"):
    files = [f for f in files if not f.startswith("part_n_b")]      # агент уже собрал свой итоговый файл
def ok(d):
    q, a = d.get("question", ""), d.get("answer", "")
    st = re.findall(r"<<([\d.]+)([-+*/])([\d.]+)=([\d.]+)>>", a.replace(",", ""))
    if not q or not st or "####" not in a: return False
    for x, op, y, z in st:
        x, y, z = float(x), float(y), float(z)
        v = {"+": x + y, "-": x - y, "*": x * y, "/": x / y if y else None}[op]
        if v is None or abs(v - z) > 1e-6: return False
    try: fin = float(a.split("####")[-1].strip().replace(",", ""))
    except ValueError: return False
    return abs(fin - float(st[-1][3])) < 1e-6
te = pq.read_table(io.BytesIO(urllib.request.urlopen("https://huggingface.co/datasets/openai/gsm8k/resolve/refs%2Fconvert%2Fparquet/main/test/0000.parquet", timeout=60).read())).to_pandas()
def grams(s, n=8):
    w = re.findall(r"[a-z]+|\d+", s.lower()); return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
T = set()
for q in te["question"]: T |= grams(q)
seen, out = set(), []; stat = dict(lines=0, bad=0, dup=0, contam=0)
for f in files:
    for line in open(f):
        if not line.strip(): continue
        stat["lines"] += 1
        try: d = json.loads(line)
        except json.JSONDecodeError: stat["bad"] += 1; continue
        if not ok(d): stat["bad"] += 1; continue
        k = re.sub(r"\s+", " ", d["question"].strip().lower())
        if k in seen: stat["dup"] += 1; continue
        if grams(d["question"]) & T: stat["contam"] += 1; continue
        seen.add(k); out.append({"question": d["question"], "answer": d["answer"]})
with open("clean2.jsonl", "w") as fo:
    for d in out: fo.write(json.dumps(d, ensure_ascii=False) + "\n")
print("файлов", len(files), stat, "→ чистых", len(out))
