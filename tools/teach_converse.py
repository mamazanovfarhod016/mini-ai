"""teach_converse.py — предобучение собеседника из диалогов в СОСТОЯНИЕ сервера (как живой разговор: ответ → эталон как исправление).
Запуск: .venv/bin/python tools/teach_converse.py [state=~/Developer/mini-ai-runs/memory-core/experimental_state]"""
import sys, os, json, glob, collections, time
sys.path.insert(0, ".")
from mini_ai.meaning import Meaning
from mini_ai.converse import Converse
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
state = os.path.expanduser(args.get("state", "~/Developer/mini-ai-runs/memory-core/experimental_state")); os.makedirs(state, exist_ok=True)

class Mind: pass
m = Mind(); m.meaning = Meaning(None)
EN_BASE = os.path.expanduser("~/Developer/mini-ai-runs/learning-core/meaning_base.npz")
if os.path.exists(EN_BASE): m.meaning.load(EN_BASE)
C = Converse(m, state); C.cases = []; C.schemes = []; C.stats.clear()
rows = []
for pat in ["data/dialogs/en.jsonl", "data/dialogs/ru.jsonl", "data/dialogs/_rr_parts/*.jsonl", "data/dialogs/ru_reasoning.jsonl"]:   # англ. первыми: при любой обрезке русские (ценнее) остаются
    for f in sorted(glob.glob(pat)):
        rs = [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
        if "en.jsonl" in f: rs = rs[:int(args.get("en_limit", 4000))]
        rows += rs
# английские пары из intents (q→a, только английские и без эмодзи-шаблонов «чем могу помочь»)
try:
    d = json.load(open("data/intents_pairs.json"))["train"]
    en = [x for x in d if (x.get("lang") == "en" or (not x.get("lang") and x["q"].isascii())) and x["a"].isascii() and len(x["a"]) > 8]
    seen = set(); en_rows = []
    for x in en:
        k = (x["q"].lower(), x["a"].lower())
        if k in seen: continue
        seen.add(k); en_rows.append(dict(turns=[dict(role="user", text=x["q"]), dict(role="bot", text=x["a"])]))
    rows += en_rows[:3000]; print("en pairs", len(en_rows[:3000]))
except Exception as e: print("intents skipped", e)
t0 = time.time(); tot = collections.Counter()
for dlg in rows: tot.update(C.teach_dialog(dlg["turns"]))
C.profile = {}; C.facts = []; C.save()
print(f"taught {len(rows)} dialogs in {time.time()-t0:.0f}s: {dict(tot)}\n{C.report()}\nstate → {C.path}")
