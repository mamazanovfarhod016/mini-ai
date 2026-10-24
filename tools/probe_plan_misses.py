"""Почему план не извлекается: для GSM8K-test (и train) классифицируем причину отказа plan_of.
Причины: no_steps (нет <<…>>), operand_missing (операнд не найден среди чисел условия / прошлых результатов / констант),
answer_not_step (ответ не равен ни одному результату шага), other. Для operand_missing печатаем самые частые «потерянные» числа."""
import io, re, collections, urllib.request, sys
import pyarrow.parquet as pq
src = open("tools/reader_gsm8k.py").read(); blk = src[src.index("NUM = re.compile"): src.index("def execute")]
ns = {"re": re, "math": __import__("math")}; exec(blk, ns)
numbers, gold_answer, CONST_OK = ns["numbers"], ns["gold_answer"], ns["CONST_OK"]
split = sys.argv[1] if len(sys.argv) > 1 else "test"
df = pq.read_table(io.BytesIO(urllib.request.urlopen(f"https://huggingface.co/datasets/openai/gsm8k/resolve/refs%2Fconvert%2Fparquet/main/{split}/0000.parquet", timeout=60).read())).to_pandas()
why = collections.Counter(); lost = collections.Counter(); ex = collections.defaultdict(list)
for q, a in zip(df["question"], df["answer"]):
    try: p = ns["plan_of"](q, a)
    except Exception: p = None
    if p: why["ok"] += 1; continue
    body = a.replace(",", "").replace("$", "")
    steps = re.findall(r"<<([^<>=]+)=\s*([\d.]+)>>", body)
    if not steps: why["no_steps"] += 1; ex["no_steps"].append(q[:90]); continue
    qn = numbers(q); res = []; reason = None
    for exp, z in steps:
        for t in re.findall(r"\d+(?:\.\d+)?", exp):
            v = float(t)
            if not (any(abs(v - n) < 1e-6 for n in qn) or any(abs(v - r) < 1e-6 for r in res) or f"{v:g}" in CONST_OK):
                reason = "operand_missing"; lost[f"{v:g}"] += 1
        if re.search(r"[()]", exp): reason = reason or "parentheses"
        try: res.append(float(z))
        except ValueError: pass
    g = gold_answer(a)
    if reason is None and (g is None or not any(abs(g - r) < 1e-6 for r in res)): reason = "answer_not_step"
    reason = reason or "other"; why[reason] += 1
    if len(ex[reason]) < 4: ex[reason].append(q[:100] + " || " + steps[0][0])
n = len(df); print(split, n, {k: f"{v} ({v/n:.0%})" for k, v in why.most_common()})
print("чаще всего теряются числа:", lost.most_common(25))
for k, v in ex.items():
    print("--", k); [print("   ", x) for x in v]
