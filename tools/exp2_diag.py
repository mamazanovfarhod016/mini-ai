"""Диагностика: ПОЧЕМУ память путает факты. Смотрим, на что похожи ошибочные ответы (тот же предмет?
тот же язык? другое имя?) и пробуем ключ = «отличительные» токены (имя), а не вся фраза.
Теория T1-name: усреднение по всей фразе топит имя (1 токен из ~10). Ключ только по имени должен резко
поднять точность. Проверяем: ключ = mean hidden-state токенов ИМЕНИ в факте и в вопросе.
Теория T1-contrast: ключ = hidden − средний hidden фраз ТОГО ЖЕ шаблона (убираем «форму», остаётся «содержание»).
"""
import json, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.transformer import load_model
from mini_ai.oneshot_bench import make_facts
from mini_ai.memory_core import Encoder, unit
from train_transformer import clean_question

MODEL = "/home/hermes/Developer/mini-ai/model_transformer_school.json"
OUT = "/home/hermes/Developer/mini-ai-runs/memory-core"
p, cfg, tok, meta = load_model(MODEL)
facts = make_facts(100, seed=7)
L = 1
enc = Encoder(p, cfg, tok, L)


def name_span(text, name):
    """Индексы BPE-токенов, покрывающих имя (по символьным позициям в очищенном тексте)."""
    ct = clean_question(text); nm = clean_question(name)
    i = ct.find(nm)
    ids = tok.encode(ct)
    # восстанавливаем символьные границы токенов
    pos, spans = 0, []
    for t in ids:
        s = tok.decode([t]); spans.append((pos, pos + len(s))); pos += len(s)
    if i < 0:
        return ids, list(range(len(ids)))
    sel = [k for k, (a, b) in enumerate(spans) if b > i and a < i + len(nm)]
    return ids, sel or list(range(len(ids)))


t0 = time.time()
KF_all, KQ_all, KF_name, KQ_name = [], [], [], []
for f in facts:
    ids_f, sel_f = name_span(f["fact"], f["name"]); ids_q, sel_q = name_span(f["q"], f["name"])
    sf, sq = enc.states(ids_f), enc.states(ids_q)
    KF_all.append(sf.mean(0)); KQ_all.append(sq.mean(0))
    KF_name.append(sf[sel_f].mean(0)); KQ_name.append(sq[sel_q].mean(0))
KF_all, KQ_all, KF_name, KQ_name = map(np.stack, (KF_all, KQ_all, KF_name, KQ_name))
print(f"encode {time.time()-t0:.0f} s")


def evaluate(KF, KQ, label):
    S = unit(KF) @ unit(KQ).T          # S[i, j] = похожесть факта i на вопрос j
    pred = S.argmax(0)
    ok = (pred == np.arange(len(facts))).mean()
    # разбор ошибок
    same_kind = same_lang = same_obj = 0; wrong = 0
    for j, i in enumerate(pred):
        if i == j: continue
        wrong += 1
        same_kind += facts[i]["kind"] == facts[j]["kind"]; same_lang += facts[i]["lang"] == facts[j]["lang"]
    print(f"{label:14} acc={ok*100:5.1f}%  ошибок {wrong}: тот же тип {same_kind}, тот же язык {same_lang}")
    return ok


evaluate(KF_all, KQ_all, "mean (всё)")
evaluate(KF_name, KQ_name, "mean (имя)")
# contrast: вычитаем средний вектор фраз того же языка и типа (убираем «форму шаблона»)
def contrast(K):
    K2 = K.copy()
    for lang in ("uz", "ru", "en"):
        for kind in ("count", "city"):
            m = np.array([f["lang"] == lang and f["kind"] == kind for f in facts])
            if m.sum() > 1:
                K2[m] -= K[m].mean(0)
    return K2
evaluate(contrast(KF_all), contrast(KQ_all), "contrast(всё)")
evaluate(contrast(KF_name), contrast(KQ_name), "contrast(имя)")
# то же при токенах напрямую (эмбеддинг wte, слой 0 — без контекста)
enc0 = Encoder(p, cfg, tok, -1) if False else None
E = np.asarray(p["wte"], dtype=np.float64)
KF_e = np.stack([E[name_span(f["fact"], f["name"])[0]][name_span(f["fact"], f["name"])[1]].mean(0) for f in facts])
KQ_e = np.stack([E[name_span(f["q"], f["name"])[0]][name_span(f["q"], f["name"])[1]].mean(0) for f in facts])
evaluate(KF_e, KQ_e, "wte (имя)")
# сколько уникальных имён среди 100 фактов — потолок для ключа «только имя»
names = {(f["lang"], f["name"]) for f in facts}
print("уникальных (язык, имя):", len(names), "из", len(facts), "→ потолок ключа «только имя» ≈", f"{len(names)/len(facts)*100:.0f}%")
