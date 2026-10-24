import json, re, io, urllib.request
import pyarrow.parquet as pq
te = pq.read_table(io.BytesIO(urllib.request.urlopen("https://huggingface.co/datasets/openai/gsm8k/resolve/refs%2Fconvert%2Fparquet/main/test/0000.parquet", timeout=60).read())).to_pandas()
def grams(s, n=8):
    w = re.findall(r"[a-z]+|\d+", s.lower()); return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}
T = set()
for q in te["question"]: T |= grams(q)
ours = [json.loads(l) for l in open("data/gsm_new/all.jsonl") if l.strip()]
clean = [d for d in ours if not grams(d["question"]).intersection(T)]
with open("data/gsm_new/clean.jsonl", "w") as f:
    for d in clean: f.write(json.dumps(d, ensure_ascii=False) + "\n")
print("чистых", len(clean), "из", len(ours))
