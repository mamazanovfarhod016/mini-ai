import json, sys, glob, collections, re
parts = sorted(glob.glob('/home/hermes/Developer/mini-ai/data/dialogs/_rr_parts/p*.jsonl'))
ids = set(); sigs = set(); errs = []; topics = collections.Counter(); total = 0
bad_starts = ("Конечно", "Отлично", "Я понимаю", "Понимаю")
for p in parts:
    for i, line in enumerate(open(p, encoding='utf-8'), 1):
        line = line.strip()
        if not line: continue
        total += 1
        try:
            d = json.loads(line)
        except Exception as e:
            errs.append(f"{p}:{i} JSON: {e}"); continue
        for k in ("id", "lang", "topic", "memory", "turns"):
            if k not in d: errs.append(f"{d.get('id')} missing {k}")
        if d["id"] in ids: errs.append(f"dup id {d['id']}")
        ids.add(d["id"])
        t = d["turns"]
        if not (4 <= len(t) <= 10) or len(t) % 2: errs.append(f"{d['id']} turns={len(t)}")
        for j, x in enumerate(t):
            if x["role"] != ("user" if j % 2 == 0 else "bot"): errs.append(f"{d['id']} role order")
            if x["role"] == "bot" and x["text"].startswith(bad_starts): errs.append(f"{d['id']} bad start: {x['text'][:30]}")
            # arithmetic check a op b = c
            for m in re.finditer(r'(?<![\d,.])(\d+(?:[,.]\d+)?(?:\s*[+−\-×*/]\s*\d+(?:[,.]\d+)?)+)\s*=\s*(\d+(?:[,.]\d+)?)', x["text"]):
                expr = m.group(1).replace('−', '-').replace('×', '*').replace(',', '.')
                try:
                    r = eval(expr)
                except Exception:
                    continue
                c = float(m.group(2).replace(',', '.'))
                if abs(r - c) > 0.01: errs.append(f"{d['id']} arith {m.group(0)}")
        sig = tuple(x["text"] for x in t)
        if sig in sigs: errs.append(f"{d['id']} dup content")
        sigs.add(sig)
        if d["topic"].startswith("память") and not d["memory"]: errs.append(f"{d['id']} memory flag")
        topics[d["topic"].split("-")[0]] += 1
print("total", total, "errors", len(errs))
for e in errs: print(" ", e)
for k, v in topics.most_common(): print(f"  {k}: {v}")
