"""convert DailyDialog (en) → data/dialogs/en.jsonl in our format (user/bot alternate; drop dialogs with odd junk)"""
import pandas as pd, json, re, glob
p = glob.glob("data/dialogs/en_src/**/train-*.parquet", recursive=True)[0]
d = pd.read_parquet(p)
def clean(u):
    u = re.sub(r"\s+([,.!?;:'])", r"\1", u.strip()); u = re.sub(r"\s+", " ", u); u = u.replace(" ’ ", "'").replace(" ' ", "'"); u = re.sub(r"\s'\s?(s|t|re|ve|ll|d|m)\b", r"'\1", u)
    return u
out = []; n_turns = 0
for did, g in d.groupby("dialog_id", sort=False):
    utts = [clean(u) for u in g["utterance"].tolist() if u and u.strip()]
    if len(utts) < 2: continue
    turns = [dict(role=("user" if i % 2 == 0 else "bot"), text=u) for i, u in enumerate(utts[:10])]
    if turns[-1]["role"] == "user": turns = turns[:-1]
    if len(turns) < 2: continue
    out.append(dict(id=f"en-{did}", lang="en", topic="daily", memory=False, turns=turns)); n_turns += len(turns)
with open("data/dialogs/en.jsonl", "w", encoding="utf-8") as f:
    for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(len(out), "dialogs", n_turns, "turns"); print(out[0]["turns"][:3])
