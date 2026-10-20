import json, re, collections

path = "/home/hermes/Developer/mini-ai/data/dialogs/en.jsonl"
lines = open(path, encoding="utf-8").read().splitlines()

errors = []
ids = set()
bot_replies_all = []
topic_counts = collections.Counter()
cyr_re = re.compile(r'[\u0400-\u04FF]')
arith_re = re.compile(r'(-?\d+)\s*([+\-x])\s*(-?\d+)\s*=\s*(-?\d+)')

n = len(lines)
for i, line in enumerate(lines):
    try:
        d = json.loads(line)
    except Exception as e:
        errors.append(f"line {i}: invalid JSON: {e}")
        continue
    did = d.get("id")
    if did in ids:
        errors.append(f"{did}: duplicate id")
    ids.add(did)
    topic_counts[d.get("topic")] += 1
    turns = d.get("turns", [])
    if not (4 <= len(turns) <= 10):
        errors.append(f"{did}: turn count {len(turns)} not in 4..10")
    if turns:
        if turns[0]["role"] != "user":
            errors.append(f"{did}: doesn't start with user")
        if turns[-1]["role"] != "bot":
            errors.append(f"{did}: doesn't end with bot")
        for j, t in enumerate(turns):
            expected = "user" if j % 2 == 0 else "bot"
            if t["role"] != expected:
                errors.append(f"{did}: turn {j} role mismatch")
            if t["role"] == "bot":
                txt = t["text"]
                bot_replies_all.append(txt)
                wc = len(txt.split())
                if wc > 30:
                    errors.append(f"{did}: bot turn {j} too long ({wc} words)")
                if cyr_re.search(txt):
                    errors.append(f"{did}: bot turn {j} contains Cyrillic")
                for m in arith_re.finditer(txt):
                    a, op, b, c = m.groups()
                    a, b, c = int(a), int(b), int(c)
                    if op == "+":
                        ok = a + b == c
                    elif op == "-":
                        ok = a - b == c
                    elif op == "x":
                        ok = a * b == c
                    else:
                        ok = True
                    if not ok:
                        errors.append(f"{did}: arithmetic wrong in '{txt}'")
    for mc in d.get("memory_checks", []):
        turn_idx = mc["turn"]
        words = mc["must_contain"]
        if turn_idx >= len(turns):
            errors.append(f"{did}: memory_check turn idx out of range")
            continue
        target = turns[turn_idx]
        if target["role"] != "bot":
            errors.append(f"{did}: memory_check turn is not bot")
        for w in words:
            if w not in target["text"]:
                errors.append(f"{did}: memory_check word '{w}' not substring of bot turn {turn_idx}")
            found_earlier = any(w in turns[k]["text"] for k in range(turn_idx) if turns[k]["role"]=="user")
            if not found_earlier:
                errors.append(f"{did}: memory_check word '{w}' not found in earlier user turn")

uniq_ratio = len(set(bot_replies_all)) / len(bot_replies_all) if bot_replies_all else 0

print("total dialogs:", n)
print("unique ids:", len(ids))
print("bot replies:", len(bot_replies_all), "unique:", len(set(bot_replies_all)), "ratio:", round(uniq_ratio,4))
print("errors:", len(errors))
for e in errors[:60]:
    print(" -", e)
print("\ntopic distribution:")
for t, c in topic_counts.most_common():
    print(f"  {t}: {c}")
