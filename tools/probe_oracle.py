"""probe_oracle.py — ПОТОЛОК механизма «игнорировать уточнители»: на свежих 80 вопросах среза (холодный ум) сравниваем
base / oracle (все отсутствующие слова — уточнители) / heur (только куски после предлога/союза/местоимения).
    tools/light.sh tools/probe_oracle.py slices=2000,5000
"""
import sys, shutil, gc, json; sys.path.insert(0, ".")
import pandas as pd
from mini_ai.experimental import ExperimentalMind
args = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a)
slices = [int(x) for x in args.get("slices", "2000,5000").split(",")]; TEACH, TEST = 120, 80
tr = pd.read_parquet("data/school/sciq/data/train-00000-of-00001.parquet").dropna(subset=["support"])
def hit(o, g):
    a = ((o.get("text") or "") + " " + str(o.get("answer") or "")).lower(); return str(g).lower() in a
PREP = set("in on at of for from by with within without during before after while whether because despite through across along between among into onto under over such like via per than when where if unless until since although though except including about you we they do does did".split())
tot = {}
for lo in slices:
    rows = list(tr.iloc[lo:lo + TEACH + TEST].iterrows()); h2 = rows[TEACH:]
    sd = f"/tmp/wtb/orc_{lo}"; shutil.rmtree(sd, ignore_errors=True); m = ExperimentalMind(school=False, wiki=False, state_dir=sd)
    for _, r in rows: m.kb.read_text(str(r["support"]), "учебник")
    R = m.reform
    def heur(q, absent):
        out = set()
        for intro, words, ab in R.runs(q, lambda w: w not in absent):
            if ab and intro in PREP: out |= set(words)
        return out
    variants = dict(base=(lambda q, a: set(), lambda q: set()),
                    oracle=(lambda q, a: set(a), lambda q: set()),
                    heur=(heur, lambda q: set()),
                    heur_typ=(heur, lambda q: set(R.typeword(q))))
    res = {}
    for name, (qf, tf) in variants.items():
        R.qualifiers = qf; R.type_words = tf
        m.new_session(); m._last_out = None
        outs = [m.ask(r["question"], internet=False) for _, r in h2]
        res[name] = [hit(o, r["correct_answer"]) for o, (_, r) in zip(outs, h2)]
        print(f"[{lo}] {name}: {sum(res[name])}/{len(h2)}", flush=True)
    for name in variants:
        if name == "base": continue
        won = [r["question"][:70] for (_, r), a, b in zip(h2, res["base"], res[name]) if b and not a]
        lost = [r["question"][:70] for (_, r), a, b in zip(h2, res["base"], res[name]) if a and not b]
        print(f"[{lo}] {name}: won {len(won)} lost {len(lost)}"); 
        for w in won: print("   +", w)
        for w in lost: print("   -", w)
    del m; gc.collect()
