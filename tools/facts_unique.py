"""Синтетические уникальные факты (exp4–6)."""
import random
MODEL = "/home/hermes/Developer/mini-ai/model_transformer_school.json"
OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
SYL = ["ba", "zo", "ra", "mi", "do", "sha", "ke", "nu", "li", "to", "far", "jon", "gul", "nur", "sher", "bek", "din", "ol", "ser", "ka"]


def make_unique(N, seed=11):  # те же факты, что в exp4
    rng = random.Random(seed); seen = set(); facts = []
    while len(facts) < N:
        n = "".join(rng.choice(SYL) for _ in range(rng.choice([2, 3]))); nm = n[0].upper() + n[1:]
        if nm in seen: continue
        seen.add(nm)
        lang = ["uz", "ru", "en"][len(facts) % 3]; num = rng.randrange(2, 99); obj = rng.choice(["olma", "kitob", "qalam"])
        fact = {"uz": f"{nm}da {num} ta {obj} bor.", "ru": f"У {nm} {num} {obj}.", "en": f"{nm} has {num} {obj}."}[lang]
        q = {"uz": f"{nm}da nechta {obj} bor?", "ru": f"Сколько {obj} у {nm}?", "en": f"How many {obj} does {nm} have?"}[lang]
        facts.append({"name": nm, "obj": obj, "fact": fact, "q": q, "gold": str(num), "lang": lang})
    return facts
