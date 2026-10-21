"""exp40 — слой ДИАЛОГА для памяти без градиента (переспрос, нить разговора, разнообразие речи, инициатива).

ГИПОТЕЗА (написана 21.10.2026 ДО кода mini_ai/dialog.py):
  Если поверх рамок вопросов добавить состояние разговора (последние N рамок и сущностей), переспрос из
  реально имеющихся фактов и сборку ответа несколькими грамматическими способами, то на зафиксированном
  наборе из 30 мини-диалогов (tools/exp_dialog_set.json) доля диалогов, где ВСЕ реплики отвечены верно,
  вырастет с контроля (experimental.py как есть, ожидаю ≈ 3–6/30 — верны только те диалоги, где каждая
  реплика сама по себе полный вопрос) до ≥ 20/30, при этом:
   • ≥ 70 % случаев «не знаю» на 20 контрольных вопросах превратятся в осмысленный переспрос
     (kind='clarify', в тексте есть распознанная сущность вопроса, варианты — только из фактов памяти);
   • на 20 одинаковых вопросах про один факт — ≥ 3 различных формулировки (контроль: 1);
   • ответы на обычные вопросы не портятся (те же 30 диалогов — первые реплики).
ЧИСЛО УСПЕХА: диалоги ≥ 20/30, переспрос ≥ 70 %, формулировок ≥ 3. Набор диалогов после начала кода не меняется.

Запуск:  cd ~/Developer/mini-ai-memory/wt && timeout 600 python3 tools/exp40_dialog.py [--control] [--tag before|after]
Лог:     ~/Developer/mini-ai-runs/memory-core/exp40_dialog_<tag>.log / .json
Контроль (--control): ExperimentalMind без слоя диалога (dialog=False, либо старый код без такого параметра).
"""
import os, sys, json, time, shutil, argparse, tempfile, random
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RUNS = os.path.expanduser("~/Developer/mini-ai-runs/memory-core")
SET = json.load(open(os.path.join(ROOT, "tools/exp_dialog_set.json")))


def make_mind(control):
    from mini_ai.experimental import ExperimentalMind
    sd = tempfile.mkdtemp(prefix="exp40_state_")             # чистое состояние: learned.txt не трогаем
    try: m = ExperimentalMind(wiki=False, state_dir=sd, dialog=not control)
    except TypeError: m = ExperimentalMind(wiki=False, state_dir=sd)      # старый код: параметра dialog нет
    m.kb.read_text(" ".join(SET["seed"]), SET["seed_source"])
    return m


def new_session(m):
    if hasattr(m, "new_session"): m.new_session()
    else: m.history = []


def turn(m, q):
    """→ dict(kind, text, answer). Для контроля kind = answer|unknown."""
    r = m.ask(q, internet=False)
    kind = r.get("kind") or ("answer" if r.get("answer") else "unknown")
    return dict(kind=kind, text=r.get("text") or "", answer=r.get("answer"), how=r.get("how"))


def ok_turn(exp, got):
    kinds = exp["kind"] if isinstance(exp["kind"], list) else [exp["kind"]]
    if got["kind"] not in kinds: return False
    t = got["text"].lower()
    if any(s.lower() not in t for s in exp.get("contains", [])): return False
    if exp.get("contains_any") and not any(s.lower() in t for s in exp["contains_any"]): return False
    return True


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--control", action="store_true"); ap.add_argument("--tag", default=None)
    a = ap.parse_args(); tag = a.tag or ("control" if a.control else "dialog")
    os.makedirs(RUNS, exist_ok=True)
    log = open(os.path.join(RUNS, f"exp40_dialog_{tag}.log"), "w")
    def P(*xs):
        s = " ".join(str(x) for x in xs); print(s, flush=True); log.write(s + "\n")
    t0 = time.time(); m = make_mind(a.control); P(f"# exp40 {tag}: boot {m.boot}, +seed → {len(m.kb.facts)} фактов, {time.time()-t0:.1f}s")

    # 1. «не знаю» → переспрос (каждый вопрос в свежей сессии)
    unk = []
    for q in SET["unknown_probe"]:
        new_session(m); g = turn(m, q)
        ents = [w for w in q.rstrip("?").split() if w[:1].isupper() and w.lower() not in ("what", "who", "where", "when", "how", "which")]
        meaningful = g["kind"] == "clarify" and (not ents or any(e.lower().strip(",.'s") in g["text"].lower() for e in ents))
        unk.append(dict(q=q, kind=g["kind"], text=g["text"], meaningful=meaningful))
        P(f"[unk] {q} -> {g['kind']}: {g['text']}")
    n_unk = sum(1 for u in unk if u["kind"] in ("unknown", "clarify")); n_clar = sum(1 for u in unk if u["meaningful"])
    P(f"## не знаю/переспрос: {n_unk}/20 без ответа, из них осмысленный переспрос {n_clar} ({100*n_clar/max(1,n_unk):.0f}%)")

    # 2. 30 мини-диалогов
    res = []; good = 0
    for d in SET["dialogs"]:
        new_session(m); turns = []; all_ok = True
        for tq in d["turns"]:
            g = turn(m, tq["q"]); o = ok_turn(tq, g); all_ok &= o
            turns.append(dict(q=tq["q"], **g, ok=o)); P(f"[d{d['id']:02d}] {'✓' if o else '✗'} {tq['q']} -> ({g['kind']}) {g['text']}")
        good += all_ok; res.append(dict(id=d["id"], type=d["type"], ok=all_ok, turns=turns))
    first_ok = sum(1 for r in res if r["turns"][0]["ok"])
    by_type = {}
    for r in res: by_type.setdefault(r["type"], [0, 0]); by_type[r["type"]][1] += 1; by_type[r["type"]][0] += r["ok"]
    P(f"## диалоги целиком верны: {good}/30; первые реплики верны: {first_ok}/30; по типам: {by_type}")

    # 3. разнообразие: 20 раз один факт — свежие сессии и одна сессия
    q = SET["variety_question"]; fresh, same = [], []
    for i in range(SET["variety_repeats"]):
        new_session(m); fresh.append(turn(m, q)["text"])
    new_session(m)
    for i in range(SET["variety_repeats"]): same.append(turn(m, q)["text"])
    P(f"## формулировок на 20 одинаковых фактов: свежие сессии {len(set(fresh))}, одна сессия {len(set(same))}")
    for s in sorted(set(fresh) | set(same)): P("   •", s)

    out = dict(tag=tag, control=a.control, facts=len(m.kb.facts), dialogs_ok=good, first_turn_ok=first_ok, by_type=by_type,
               unknown_total=n_unk, clarify_meaningful=n_clar, variety_fresh=len(set(fresh)), variety_same=len(set(same)),
               variety_texts=sorted(set(fresh) | set(same)), unknown=unk, dialogs=res, seconds=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(RUNS, f"exp40_dialog_{tag}.json"), "w"), ensure_ascii=False, indent=1)
    P(f"# ИТОГ {tag}: диалоги {good}/30, переспрос {n_clar}/{n_unk}, формулировок {len(set(fresh))}/{len(set(same))}, {time.time()-t0:.0f}s")
    shutil.rmtree(m.state_dir, ignore_errors=True)


if __name__ == "__main__": main()
