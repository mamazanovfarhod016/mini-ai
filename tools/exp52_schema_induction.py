"""exp52 — индукция схем (понятий) из памяти фактов без градиента.

Гипотеза: сущности группируются по ПРОФИЛЮ отношений (какие rel и какие «головы» Y у них бывают),
и группа = схема: «страна → be_country, have_population, be_located, border…». Из схемы можно
предсказывать, какие отношения у новой сущности ЕСТЬ, но ещё не прочитаны (→ она знает, чего не знает).

Метод: ключ отношения = rel, а для rel=be — be:<голова Y> (первое содержательное слово Y: country/city/actor).
Профиль сущности = множество ключей. Кластеризация: k-средних по биполярным HDC-кодам профилей
(bundle кодов ключей), k=40, 10 итераций — без градиента, только средние. Схема кластера = ключи с частотой ≥0.5.
Тест: отложенные сущности (20%), у каждой скрываем один случайный ключ; предсказываем скрытые ключи как
«ключи схемы, которых у сущности не видно». Precision/recall по скрытым ключам.
Контроль: глобальная частота ключей (top-N ключей по всей базе, без кластеров).
Успех: precision ≥0.7 при recall ≥0.5 и ≥3 схемы, читаемые текстом.
"""
import json, os, re, time, collections, numpy as np
rng = np.random.default_rng(0)
SRC = os.path.expanduser("~/Developer/mini-ai-runs/memory-core/simplewiki_full/facts.jsonl")
OUT = os.path.expanduser("~/Developer/mini-ai-runs/learning-core")
STOP = set("a an the of in on at to for from by with and or as is was are were its his her their one two first second".split())
ADV = set("also very not often most usually mostly currently now still sometimes only about more less best better well known named called considered located found used made born".split())
LIMIT = 400000

def head(Y):
    """голова Y = первое существительное-категория: пропускаем артикли, наречия, причастия, прилагательные-порядковые"""
    ws = [w for w in re.findall(r"[a-z]+", Y.lower()) if w not in STOP and w not in ADV and not w.endswith("ly")]
    for w in ws[:4]:
        if w.endswith(("ed", "ing")) and len(w) > 5: continue
        if w in ("largest", "smallest", "biggest", "oldest", "famous", "small", "large", "big", "important", "popular", "major"): continue
        return w
    return None

def load():
    prof = collections.defaultdict(set); n = 0
    with open(SRC) as f:
        for line in f:
            X, rel, Y, title = json.loads(line); n += 1
            if n > LIMIT: break
            if X != title or "@" in rel: continue                 # только факты о сущности-заголовке статьи
            h = head(Y)
            if rel == "be":
                if not h: continue
                key = "be:" + h
            elif rel == "have": key = "have:" + (h or "?")
            else: key = rel
            prof[X].add(key)
    return prof

if __name__ == "__main__":
    t0 = time.time(); prof = load()
    ents = [e for e, ks in prof.items() if len(ks) >= 3]
    cnt = collections.Counter(k for e in ents for k in prof[e])
    keys = [k for k, c in cnt.most_common() if c >= 20]; kid = {k: i for i, k in enumerate(keys)}
    ents = [e for e in ents if sum(k in kid for k in prof[e]) >= 3]
    print(f"сущностей {len(ents)}, ключей отношений {len(keys)}, {time.time()-t0:.0f}s")
    rng.shuffle(ents); n_te = len(ents) // 5; te, tr = ents[:n_te], ents[n_te:]
    # профили → биполярные коды (bundle случайных кодов ключей), D=4096
    D = 4096; KC = rng.choice([-1.0, 1.0], size=(len(keys), D)).astype(np.float32)
    def code(e, hide=None):
        v = sum(KC[kid[k]] for k in prof[e] if k in kid and k != hide); return v / (np.linalg.norm(v) + 1e-9)
    Xtr = np.stack([code(e) for e in tr])
    # k-средних (без градиента)
    k = 40; C = Xtr[rng.choice(len(Xtr), k, replace=False)].copy()
    for it in range(10):
        a = np.argmax(Xtr @ C.T, 1)
        for j in range(k):
            m = Xtr[a == j]
            if len(m): C[j] = m.mean(0); C[j] /= np.linalg.norm(C[j]) + 1e-9
    a = np.argmax(Xtr @ C.T, 1)
    # схемы кластеров
    schema = []
    for j in range(k):
        mem = [tr[i] for i in np.where(a == j)[0]]
        if len(mem) < 10: schema.append({}); continue
        c = collections.Counter(kk for e in mem for kk in prof[e] if kk in kid)
        schema.append({kk: c[kk] / len(mem) for kk in c})
    # тест: скрыть один ключ, предсказать
    glob = [kk for kk, _ in cnt.most_common()]
    def evaluate(pred_fn, thr):
        tp = fp = fn = 0
        for e in te:
            vis = [kk for kk in prof[e] if kk in kid]
            if len(vis) < 3: continue
            hide = vis[int(rng.integers(len(vis)))]; shown = set(vis) - {hide}
            pred = set(pred_fn(e, hide, shown, thr))
            tp += hide in pred; fp += len(pred - {hide}); fn += hide not in pred
        P = tp / max(1, tp + fp); R = tp / max(1, tp + fn); return P, R
    def by_schema(e, hide, shown, thr):
        j = int(np.argmax(C @ code(e, hide))); return [kk for kk, f in schema[j].items() if f >= thr and kk not in shown]
    def by_global(e, hide, shown, thr):
        n = int(thr); return [kk for kk in glob if kk not in shown][:n]
    # метод Б: ассоциации «ключ → ключ» (условная частота по обучающим сущностям): P(b | a) = n(a&b)/n(a)
    co = collections.Counter(); single = collections.Counter()
    for e in tr:
        ks = [kk for kk in prof[e] if kk in kid]
        for a_ in ks:
            single[a_] += 1
            for b_ in ks:
                if a_ != b_: co[(a_, b_)] += 1
    def by_assoc(e, hide, shown, thr):
        sc = collections.defaultdict(float)
        for a_ in shown:
            for b_ in keys:
                if b_ in shown or single[a_] < 5: continue
                p = co[(a_, b_)] / single[a_]
                if p > sc[b_]: sc[b_] = p
        return [b_ for b_, p in sc.items() if p >= thr]
    res = {"entities": len(ents), "keys": len(keys)}
    print("ассоциации ключ→ключ: P/R при порогах P(b|a)")
    for thr in (0.2, 0.3, 0.4, 0.5, 0.6):
        P, R = evaluate(by_assoc, thr); res[f"assoc_thr{thr}"] = (round(P, 3), round(R, 3)); print(f"  thr={thr}: precision {P:.3f} recall {R:.3f}")
    rules = sorted(((co[(a_, b_)] / single[a_], a_, b_, co[(a_, b_)]) for (a_, b_) in co if single[a_] >= 30 and co[(a_, b_)] >= 15), reverse=True)[:15]
    print("сильнейшие правила: " + "; ".join(f"{a_}→{b_} {p:.2f} (n={n})" for p, a_, b_, n in rules))
    print("схемы: P/R при порогах частоты")
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        P, R = evaluate(by_schema, thr); res[f"schema_thr{thr}"] = (round(P, 3), round(R, 3)); print(f"  thr={thr}: precision {P:.3f} recall {R:.3f}")
    print("контроль (глобальные top-N ключей):")
    for n in (1, 2, 3, 5):
        P, R = evaluate(by_global, n); res[f"global_top{n}"] = (round(P, 3), round(R, 3)); print(f"  top{n}: precision {P:.3f} recall {R:.3f}")
    # показать схемы текстом
    print("\nПримеры схем (кластер → типичные отношения, доля членов):")
    shown = 0
    for j in np.argsort([-len(s) for s in schema]):
        s = schema[j]
        if not s: continue
        top = sorted(s.items(), key=lambda t: -t[1])[:6]
        if top[0][1] < 0.6: continue
        mem = [tr[i] for i in np.where(a == j)[0]][:5]
        print(f"  [{j}] n={int((a==j).sum())} " + ", ".join(f"{kk} {f:.2f}" for kk, f in top) + f"   напр.: {', '.join(mem)}")
        shown += 1
        if shown >= 8: break
    res["seconds"] = round(time.time() - t0, 1); json.dump(res, open(os.path.join(OUT, "exp52_schema_induction.json"), "w"), indent=1); print("sec", res["seconds"])
