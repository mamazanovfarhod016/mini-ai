import json, re, collections

path = "/home/hermes/Developer/mini-ai/data/dialogs/ru.jsonl"
lines = open(path, encoding="utf-8").read().splitlines()

errors = []
ids = set()
dupe_dialog_check = set()
bot_replies = []
topic_counter = collections.Counter()

num_words = {
    "ноль":0,"один":1,"два":2,"три":3,"четыре":4,"пять":5,"шесть":6,"семь":7,"восемь":8,"девять":9,"десять":10
}

def check_arith(text):
    # pattern: N op N = N  (op words or symbols)
    m = re.search(r'(\d+)\s*(плюс|минус|умножить на|разделить на|\+|\-|\*|/)\s*(\d+)\s*=\s*(-?\d+(?:\.\d+)?)', text)
    if not m:
        return True
    a, op, b, res = m.group(1), m.group(2), m.group(3), m.group(4)
    a = float(a); b = float(b); res = float(res)
    if op in ("плюс","+"):
        expected = a + b
    elif op in ("минус","-"):
        expected = a - b
    elif op in ("умножить на","*"):
        expected = a * b
    elif op in ("разделить на","/"):
        expected = a / b
    else:
        return True
    return abs(expected - res) < 0.01

for i, line in enumerate(lines, 1):
    try:
        d = json.loads(line)
    except Exception as e:
        errors.append(f"line {i}: invalid JSON: {e}")
        continue
    did = d.get("id")
    if did in ids:
        errors.append(f"line {i}: duplicate id {did}")
    ids.add(did)
    turns = d.get("turns", [])
    if not (4 <= len(turns) <= 10):
        errors.append(f"{did}: turn count {len(turns)} out of range")
    if len(turns) % 2 != 0:
        errors.append(f"{did}: odd number of turns")
    for j, t in enumerate(turns):
        expected_role = "user" if j % 2 == 0 else "bot"
        if t.get("role") != expected_role:
            errors.append(f"{did}: turn {j} role mismatch, got {t.get('role')}")
    if turns and turns[0]["role"] != "user":
        errors.append(f"{did}: does not start with user")
    if turns and turns[-1]["role"] != "bot":
        errors.append(f"{did}: does not end with bot")
    full_text_sig = tuple(t["text"] for t in turns)
    if full_text_sig in dupe_dialog_check:
        errors.append(f"{did}: duplicate dialog content")
    dupe_dialog_check.add(full_text_sig)

    for j, t in enumerate(turns):
        if t["role"] == "bot":
            wc = len(t["text"].split())
            if wc > 30:
                errors.append(f"{did} turn {j}: bot reply too long ({wc} words): {t['text']}")
            bot_replies.append(t["text"])
            if not check_arith(t["text"]):
                errors.append(f"{did} turn {j}: arithmetic error: {t['text']}")

    for mc in d.get("memory_checks", []):
        tn = mc["turn"]
        if tn < 0 or tn >= len(turns):
            errors.append(f"{did}: memory_check turn {tn} out of range")
            continue
        bot_text = turns[tn]["text"]
        for w in mc["must_contain"]:
            if w not in bot_text:
                errors.append(f"{did}: memory word '{w}' not found in turn {tn} text: {bot_text}")
            found_earlier = any(w in turns[k]["text"] for k in range(tn) if turns[k]["role"] == "user")
            if not found_earlier:
                errors.append(f"{did}: memory word '{w}' not present in any earlier user turn")

    topic_counter[d.get("topic","?")] += 1

print("Total dialogs:", len(lines))
print("Unique ids:", len(ids))
print("Errors found:", len(errors))
for e in errors[:50]:
    print(" -", e)

unique_bot = len(set(bot_replies))
print(f"Bot replies total: {len(bot_replies)}, unique: {unique_bot} ({unique_bot/len(bot_replies)*100:.1f}%)")

print("\nTopic distribution:")
for t, c in topic_counter.most_common():
    print(f"  {t}: {c}")
