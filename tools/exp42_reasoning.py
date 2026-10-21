"""Эксперимент 42: рассуждение с разбором вопроса и схемами связей (mini_ai/reason.py).
Те же 20 двухшаговых вопросов (exp40: 0/20, exp41: 2/20) + 10 одношаговых (контроль, что не сломали простое).
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mini_ai.reason import Reasoner

OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
STORE = os.environ.get("STORE", f"{OUT}/simplewiki_full")
t0 = time.time(); facts = [json.loads(l) for l in open(f"{STORE}/facts.jsonl")]
R = Reasoner(facts)
print(f"фактов {len(facts)}, типов {len(R.etype)}, схем {len(R.scheme_n)}, {time.time()-t0:.0f}s", flush=True)
for k in (("city", "country"), ("country", "sea"), ("country", "language"), ("country", "continent"), ("work", "person")):
    if k in R.scheme: print(f"  схема {k[0]}→{k[1]} (n={R.scheme_n[k]}): {[w for w, _ in R.scheme[k].most_common(6)]}")

Q2 = [("What is the capital of the country where Mount Everest is?", ["kathmandu", "beijing"]),
      ("What is the currency of the country whose capital is Paris?", ["euro", "franc"]),
      ("What language is spoken in the country whose capital is Tashkent?", ["uzbek"]),
      ("What is the largest planet in the system that contains Earth?", ["jupiter"]),
      ("Who wrote the play in which Hamlet is the main character?", ["shakespeare"]),
      ("What river flows through the capital of France?", ["seine"]),
      ("What is the capital of the country that Istanbul is in?", ["ankara"]),
      ("What sea is next to the city of Istanbul?", ["marmara", "black sea", "bosporus"]),
      ("What continent is the country of Kazakhstan in?", ["asia"]),
      ("Who discovered the antibiotic that is made by Penicillium?", ["fleming"]),
      ("What is the chemical symbol of the metal used for coins and teeth?", ["au"]),
      ("What is the capital of the country that borders Uzbekistan to the north?", ["astana", "nur-sultan"]),
      ("What ocean is next to the country whose capital is Tokyo?", ["pacific"]),
      ("What is the largest city of the country where the Nile is?", ["cairo"]),
      ("Who painted the picture that hangs in the Louvre and shows a smiling woman?", ["leonardo"]),
      ("What is the capital of the state where Hollywood is?", ["sacramento"]),
      ("What planet is closest to the star at the center of the Solar System?", ["mercury"]),
      ("What is the highest mountain on the continent where Kenya is?", ["kilimanjaro"]),
      ("What is the capital of the country where the Taj Mahal is?", ["delhi"]),
      ("What language do people speak in the country where Berlin is?", ["german"])]
Q1 = [("What is the capital of Uzbekistan?", ["tashkent"]), ("What country is Istanbul in?", ["turkey"]),
      ("What continent is Kenya in?", ["africa"]), ("What is the capital of Japan?", ["tokyo"]),
      ("What language is spoken in Germany?", ["german"]), ("What river is Cairo on?", ["nile"]),
      ("What is the capital of Kazakhstan?", ["astana", "nur-sultan"]), ("What sea is Turkey next to?", ["black", "mediterranean", "aegean", "marmara"]),
      ("What country is Berlin the capital of?", ["germany"]), ("What planet is Jupiter?", ["gas", "giant", "largest"])]

def run(Q, label):
    ok = 0; t0 = time.time()
    for q, golds in Q:
        r = R.reason(q); h = bool(r and any(g in r["answer"].lower() for g in golds)); ok += h
        p = R.parse(q)
        print(f"  {'OK ' if h else 'MISS'} {q}\n       → «{r['answer'] if r else '-'}» (альт. {r['alternatives'] if r else ''})  разбор: {p[1] if p else None}\n       {' | '.join(r['chain']) if r else ''}")
    print(f"  {label}: {ok}/{len(Q)}, {(time.time()-t0)/len(Q)*1000:.0f} мс/вопрос\n"); return ok
o2 = run(Q2, "ДВУХШАГОВЫЕ"); o1 = run(Q1, "ОДНОШАГОВЫЕ")
print(f"ИТОГО: 2 шага {o2}/20 (exp41: 2/20, exp40: 0/20); 1 шаг {o1}/10")
json.dump(dict(two=o2, one=o1), open(f"{OUT}/exp42_reasoning.json", "w"))
