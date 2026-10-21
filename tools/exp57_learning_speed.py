"""exp57 — скорость развития: сколько НОВОГО она усваивает за минуту и выводит сама.

Гипотеза: ядро без градиента усваивает 200 новых фактов (о вымышленных сущностях — их точно нет ни
в памяти, ни в Википедии) за секунды, отвечает на ≥80% прямых вопросов сразу, ≥80% после перезапуска
(persist), и ≥40% на вопросы-ВЫВОДЫ (ответ не написан ни в одном предложении, следует из двух).
Контроль: трансформер-говорилка — те же 200 фактов ей можно дать только дообучением; замеряем,
что она отвечает на те же вопросы БЕЗ дообучения (ожидаем ~0) — и честно пишем, сколько минут стоило
бы дообучение (из meta модели: train_time_sec на 12861 пар).
Вымышленный мир генерируется с seed=0: N стран → столицы, реки, валюты; люди → родились в городе, основали компанию.
Вопросы: прямые («What is the capital of Zorvania?»), выводы («Where was Kelmar born?» нет; есть:
«Kelmar was born in Tashi.» + «Tashi is a city in Zorvania.» → «In which country was Kelmar born?»).
Числа до запуска: прямые ≥0.8, после перезапуска ≥0.8, выводы ≥0.4, время усвоения ≤10 с на 200 фактов.
"""
import os, sys, time, json, re, random, shutil
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.expanduser("~/Developer/mini-ai-runs/learning-core"); STATE = "/tmp/exp57_state"
rng = random.Random(0)
SYL = ["zor", "va", "ni", "kel", "mar", "ta", "shi", "lun", "dra", "vek", "o", "ri", "sun", "bel", "ka", "mo", "re", "th", "an", "ul"]
def name(k=3): return "".join(rng.choice(SYL) for _ in range(k)).capitalize()

def world(n=50):
    facts, direct, infer = [], [], []
    for _ in range(n):
        C, cap, riv, cur = name(), name(2), name(2), name(2)
        P, co = name(2) + " " + name(2), name(2) + " Corp"
        facts += [f"{C} is a country in the Vendar continent.", f"{cap} is the capital of {C}.", f"The {riv} River is the longest river in {C}.",
                  f"The {cur} is the currency of {C}.", f"{P} is a scientist who was born in {cap}.", f"{P} founded {co}.", f"{co} is a company in {cap}."]
        direct += [(f"What is the capital of {C}?", cap), (f"What is the currency of {C}?", cur), (f"Who founded {co}?", P.split()[0]),
                   (f"What is the longest river in {C}?", riv), (f"Where is {co}?", cap)]
        infer += [(f"In which country was {P} born?", C), (f"In which country is {co}?", C)]
    return facts, direct, infer

def hit(ans, gold): return bool(ans) and gold.lower() in ans.lower()

if __name__ == "__main__":
    from mini_ai.experimental import ExperimentalMind
    shutil.rmtree(STATE, ignore_errors=True)
    facts, direct, infer = world(); text = " ".join(facts)
    rng.shuffle(direct); rng.shuffle(infer); direct, infer = direct[:120], infer[:60]
    print(f"мир: {len(facts)} фактов, прямых вопросов {len(direct)}, выводов {len(infer)}")
    res = {"facts": len(facts)}
    m = ExperimentalMind(school=True, wiki=False, state_dir=STATE)
    # до обучения
    before = sum(hit(m.ask(q, internet=False).get("text"), g) for q, g in direct[:40]); print(f"до обучения (40 прямых): {before}/40")
    t = time.time(); m.learn(text); res["learn_sec"] = round(time.time() - t, 2); print(f"усвоила {len(facts)} фактов за {res['learn_sec']} с")
    t = time.time(); d_ok = sum(hit(m.ask(q, internet=False).get("text"), g) for q, g in direct); res["direct"] = round(d_ok / len(direct), 3); res["ask_ms"] = round(1000 * (time.time() - t) / len(direct), 1)
    i_ok = sum(hit(m.ask(q, internet=False).get("text"), g) for q, g in infer); res["infer"] = round(i_ok / len(infer), 3)
    print(f"сразу: прямые {d_ok}/{len(direct)} = {res['direct']:.2f} ({res['ask_ms']} мс/вопрос) | выводы {i_ok}/{len(infer)} = {res['infer']:.2f}")
    # перезапуск
    del m; m2 = ExperimentalMind(school=True, wiki=False, state_dir=STATE)
    d2 = sum(hit(m2.ask(q, internet=False).get("text"), g) for q, g in direct); res["direct_after_restart"] = round(d2 / len(direct), 3)
    print(f"после перезапуска: прямые {d2}/{len(direct)} = {res['direct_after_restart']:.2f}")
    # примеры вывода
    print("примеры:")
    for q, g in infer[:4]:
        r = m2.ask(q, internet=False); print(f"  {q} -> {r.get('text')}  [{r.get('how')}]  (верно: {g})")
    for q, g in direct[:2]:
        r = m2.ask(q, internet=False); print(f"  {q} -> {r.get('text')}  (верно: {g})")
    # контроль: говорилка
    try:
        from mini_ai.hybrid import Hybrid
        h = Hybrid(mind=m2); s_ok = sum(hit(h.speak(q), g) for q, g in direct[:40]); res["speaker_direct"] = round(s_ok / 40, 3)
        meta = json.loads(open(os.path.join(os.path.dirname(__file__), "..", "app/www/model_transformer.js"), encoding="utf-8").read().split("=", 1)[1].rstrip(";\n"))
        res["speaker_retrain_sec"] = meta.get("train_time_sec"); print(f"контроль трансформер без дообучения: {s_ok}/40; дообучение стоило бы ~{meta.get('train_time_sec')} с на {meta.get('train_pairs_aug')} пар (и риск забыть старое)")
    except Exception as e: print("контроль-трансформер пропущен:", e)
    json.dump(res, open(os.path.join(OUT, "exp57_learning_speed.json"), "w"), indent=1); print(res)
