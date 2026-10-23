"""exp66 — converse.py: учится из диалогов как из разговора. Критерий: после обучения на N диалогов ответы на отложенных
диалогах ближе к эталону, чем холодные (копия/случайный), схемы рассуждения срабатывают на новых задачах той же формы,
память о собеседнике отвечает точно. Контроль — exp65 (копия ближайшего): topic↔gold 0.151."""
import sys, os, json, random, glob, collections, time
sys.path.insert(0, ".")
import numpy as np
from mini_ai.meaning import Meaning
from mini_ai.converse import Converse, toks, STOP

args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
files = args.get("files", "data/dialogs/ru.jsonl,data/dialogs/_rr_parts/*.jsonl").split(",")
rows = []
for pat in files:
    for f in sorted(glob.glob(pat)):
        rows += [json.loads(l) for l in open(f, encoding="utf-8") if l.strip()]
random.Random(66).shuffle(rows); n_test = max(30, len(rows) // 6); test, train = rows[:n_test], rows[n_test:]
print(f"[data] {len(rows)} dialogs → train {len(train)} test {len(test)}", flush=True)

class Mind: pass
m = Mind(); m.meaning = Meaning(None)
RU_BASE = os.path.expanduser("~/Developer/mini-ai-runs/learning-core/meaning_ru.npz")
if os.path.exists(RU_BASE): m.meaning.load(RU_BASE)
for d in rows:
    for t in d["turns"]: m.meaning.read(t["text"])
print(f"[meaning] {len(m.meaning.sem)} words", flush=True)

C = Converse(m, None); t0 = time.time(); tot = collections.Counter()
for d in train: tot.update(C.teach_dialog(d["turns"]))
print(f"[teach] {dict(tot)}  {time.time()-t0:.0f}s | {C.report()}", flush=True)

def sim(a, b):
    va, vb = C.vec(a), C.vec(b); return float(va @ vb) if va is not None and vb is not None else 0.0
res = collections.defaultdict(list); ex = []; outs = []
pool = [c["out"] for c in C.cases]
for d in test:
    C.profile = {}; C.facts = []
    for i in range(len(d["turns"]) - 1):
        if d["turns"][i]["role"] != "user" or d["turns"][i + 1]["role"] != "bot": continue
        inp, gold = d["turns"][i]["text"], d["turns"][i + 1]["text"]
        r = C.respond(inp) or dict(text="", how="none"); outs.append(r["text"])
        res["ours"].append(sim(r["text"], gold)); res["random"].append(sim(random.choice(pool), gold))
        how = r["how"].split(":")[0].split(" (")[0]; res["how_" + how].append(1)
        if "Думаю так" in gold: res["steps_gold"].append(1); res["steps_ours"].append(1 if "Думаю так" in r["text"] else 0)
        if d.get("memory") and any(k in inp.lower() for k in ("помнишь", "как меня", "сколько у меня", "в каком я", "как зовут мо")):
            res["mem_q"].append(1); res["mem_hit"].append(1 if r["how"].startswith("память") and any(w in r["text"] for w in toks(gold) if len(w) > 2 and w not in STOP) else 0)
        if len(ex) < 14 and (("Думаю так" in gold) or d.get("memory") or random.random() < 0.15): ex.append((inp, r["text"], gold, r["how"]))
print(f"[eval] n={len(outs)}  sim↔gold ours {np.mean(res['ours']):.3f}  random {np.mean(res['random']):.3f}  diversity {len(set(outs))/len(outs):.2f}")
print(f"[eval] steps: gold had {len(res['steps_gold'])}, ours gave steps {np.mean(res['steps_ours']) if res['steps_ours'] else 0:.2f} | memory questions {len(res['mem_q'])}, hit {np.mean(res['mem_hit']) if res['mem_hit'] else 0:.2f}")
print("[eval] how:", {k[4:]: len(v) for k, v in res.items() if k.startswith("how_")})
for inp, ours, gold, how in ex: print(f"  Ю: {inp}\n  ИИ: {ours}   [{how}]\n  (эталон: {gold})")
