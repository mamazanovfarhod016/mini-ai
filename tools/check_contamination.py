import json, re, io, urllib.request
import pyarrow.parquet as pq
te = pq.read_table(io.BytesIO(urllib.request.urlopen("https://huggingface.co/datasets/openai/gsm8k/resolve/refs%2Fconvert%2Fparquet/main/test/0000.parquet", timeout=60).read())).to_pandas()
def grams(s, n=8):
    w = re.findall(r"[a-z]+|\d+", s.lower()); return {" ".join(w[i:i+n]) for i in range(len(w) - n + 1)}
T = set()
for q in te["question"]: T |= grams(q)
ours = [json.loads(l) for l in open("data/gsm_new/all.jsonl") if l.strip()]
hit = [d["question"][:90] for d in ours if grams(d["question"]).intersection(T)]
print("наших задач с 8-словным совпадением с тестом GSM8K:", len(hit), "/", len(ours))
for h in hit[:5]: print("  ", h)
