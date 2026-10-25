"""exp73 — «не знает → ищет → читает → понял → умеет использовать → объясняет» (цель 2, 25.10.2026).

Линейка: MuSiQue (вопросы, где ответ надо СОБРАТЬ из двух фактов из разных статей; ответа нет целиком ни в одном предложении),
100 вопросов «2 шага», выбраны случайно до запуска. Ум стартует СВЕЖИМ (без Википедии в памяти) → она их не знает.
Уровни (записано до запуска, цель Фархода):
  0 до учёбы       — полный вопрос без интернета (ожидаем ≈ 0)
  учёба            — полный вопрос с интернетом: её собственная петля «не знаю → ищу → читаю»
  1 запомнила      — после ПЕРЕЗАПУСКА (новый ум из сохранённого состояния = «через день»), без интернета: шаг-1 и шаг-2 по отдельности
  3 применяет      — после перезапуска, без интернета: полный вопрос (надо соединить два факта)
  5 объясняет      — «why?» после ответа 3: в объяснении есть и мост (ответ шага 1), и итог
  2 поняла (перефраз) и 4 (сама понимает, когда нужно) — следующим шагом (нужны написанные вручную вопросы)
Засчитано = эталон или его вариант встречается в ответе (без регистра/артиклей). Успех цели 2 = уровни 3–5.
Запуск: tools/light.sh tools/exp73_learn_ruler.py n=100 tag=base   (MINI_AI_RECURSE=1 — с рекурсивным чтением, когда будет)
"""
import os, re, sys, json, time, shutil, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
A = dict(a.split("=", 1) for a in sys.argv[1:] if "=" in a); N = int(A.get("n", 100)); TAG = A.get("tag", "base")
OUT = os.path.expanduser(f"~/Developer/mini-ai-runs/learning-core/exp73_{TAG}"); STATE = os.path.join(OUT, "state")
shutil.rmtree(OUT, ignore_errors=True); os.makedirs(STATE)
T0 = time.time()
def log(*a):
    s = " ".join(str(x) for x in a); print(f"[{time.time()-T0:5.0f}s] {s}", flush=True); open(os.path.join(OUT, "log.txt"), "a").write(s + "\n")

REL = {"performer": "Who performed {x}?", "educated at": "Where was {x} educated?", "mother": "Who is the mother of {x}?", "father": "Who is the father of {x}?",
       "part of": "What is {x} part of?", "instance of": "What is {x}?", "country": "In which country is {x}?", "located in the administrative territorial entity": "Where is {x} located?",
       "spouse": "Who is the spouse of {x}?", "author": "Who wrote {x}?", "director": "Who directed {x}?", "producer": "Who produced {x}?", "record label": "What is the record label of {x}?",
       "place of birth": "Where was {x} born?", "capital": "What is the capital of {x}?", "creator": "Who created {x}?", "developer": "Who developed {x}?",
       "headquarters location": "Where is the headquarters of {x}?", "employer": "Who employed {x}?", "member of sports team": "Which team did {x} play for?",
       "publisher": "Who published {x}?", "manufacturer": "Who made {x}?", "owned by": "Who owns {x}?", "located in or next to body of water": "Which body of water is {x} on?",
       "child": "Who is the child of {x}?", "sibling": "Who is the sibling of {x}?", "composer": "Who composed {x}?", "cast member": "Who starred in {x}?", "genre": "What genre is {x}?",
       "continent": "On which continent is {x}?", "place of death": "Where did {x} die?", "founded by": "Who founded {x}?", "original broadcaster": "Which network aired {x}?",
       "country of citizenship": "What country is {x} a citizen of?", "languages spoken, written or signed": "What language does {x} speak?", "religion": "What is the religion of {x}?"}
def natural(q, bridge=None):
    q = q.replace("#1", bridge) if bridge else q
    m = re.match(r"^(.+?)\s*>>\s*(.+)$", q)
    if m:
        t = REL.get(m.group(2).strip().lower()); return t.format(x=m.group(1).strip()) if t else None
    q = q.strip(); q = q[0].upper() + q[1:]
    return q if q.endswith("?") else q + "?"
d = pd.read_parquet(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/multihop/musique_val.parquet"))
d = d[d.answerable & d.id.str.startswith("2hop")]
items = []
for _, r in d.sample(frac=1, random_state=73).iterrows():
    qd = list(r.question_decomposition); b = qd[0]["answer"]
    q1 = natural(qd[0]["question"]); q2 = natural(qd[1]["question"], b)
    if not q1 or not q2: continue
    items.append(dict(q=r.question, a=r.answer, al=list(r.answer_aliases), q1=q1, a1=b, q2=q2))
    if len(items) >= N: break
json.dump(items, open(os.path.join(OUT, "items.json"), "w"), ensure_ascii=False, indent=1)
log(f"вопросов {len(items)}; пример: «{items[0]['q']}» → {items[0]['a']} (шаг 1: «{items[0]['q1']}» → {items[0]['a1']}; шаг 2: «{items[0]['q2']}»)")

def norm(s): return " " + re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", re.sub(r"\b(the|a|an)\b", " ", str(s or "").lower()))).strip() + " "
def hit(out, golds):
    txt = norm((out.get("text") or "") + " " + str(out.get("answer") or "")); return any(norm(g).strip() and norm(g) in txt for g in golds if g)

from mini_ai.experimental import ExperimentalMind
def fresh(): return ExperimentalMind(school=True, wiki=False, state_dir=STATE, dialog=True, seed=0)
m = fresh(); n0 = len(m.kb.facts); log(f"свежий ум: фактов {n0} (без Википедии), загрузка {time.time()-T0:.0f} с")

res = [dict() for _ in items]
for i, it in enumerate(items):
    res[i]["before"] = hit(m.ask(it["q"], internet=False), [it["a"]] + it["al"])
log(f"0 до учёбы (без интернета): {sum(r['before'] for r in res)}/{len(items)}")
t1 = time.time(); reads = 0
for i, it in enumerate(items):
    out = m.ask(it["q"], internet=True); res[i]["learn"] = hit(out, [it["a"]] + it["al"]); res[i]["read"] = out.get("read") or []; reads += len(res[i]["read"])
    if i % 20 == 19: log(f"  учёба {i+1}/{len(items)}: верно {sum(r.get('learn', 0) for r in res[:i+1])}, прочитано статей {reads}, {time.time()-t1:.0f} с")
log(f"учёба (с интернетом, её петля): {sum(r['learn'] for r in res)}/{len(items)}; прочитала статей {reads}; фактов стало {len(m.kb.facts)} (+{len(m.kb.facts)-n0}); {time.time()-t1:.0f} с")
del m; import gc; gc.collect()
m = fresh(); log(f"ПЕРЕЗАПУСК («через день»): фактов {len(m.kb.facts)}")
for i, it in enumerate(items):
    g = [it["a"]] + it["al"]
    res[i]["l1_step1"] = hit(m.ask(it["q1"], internet=False), [it["a1"]]); res[i]["l1_step2"] = hit(m.ask(it["q2"], internet=False), g)
    o3 = m.ask(it["q"], internet=False); res[i]["l3"] = hit(o3, g); res[i]["ans3"] = (o3.get("text") or "")[:200]
    ex = m.ask("why?", internet=False); et = norm(ex.get("text")); res[i]["expl"] = (ex.get("text") or "")[:300]
    res[i]["l5"] = bool(res[i]["l3"] and norm(it["a1"]) in et and any(norm(x) in et for x in g))
S = lambda k: sum(1 for r in res if r.get(k))
log(f"ИТОГ exp73 [{TAG}] из {len(items)}: до учёбы {S('before')} | во время учёбы {S('learn')} | "
    f"1 запомнила: шаг-1 {S('l1_step1')}, шаг-2 {S('l1_step2')} | 3 применяет (собрать 2 факта) {S('l3')} | 5 объясняет {S('l5')} | статей {reads}")
json.dump(res, open(os.path.join(OUT, "res.json"), "w"), ensure_ascii=False, indent=1)
for i in [k for k, r in enumerate(res) if r.get("l3")][:2] + [k for k, r in enumerate(res) if not r.get("l3")][:2]:
    log(f"  пример: «{items[i]['q']}» эталон {items[i]['a']} → {res[i]['ans3'][:150]} | объяснение: {res[i]['expl'][:150]}")
