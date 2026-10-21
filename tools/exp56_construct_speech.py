"""exp56 — речь как сборка из смысла: конструкции, выученные из корпуса, + слоты из реплики.

Идея (без градиента): ответ = ЧТО сказать (акт: поздороваться / отразить тему / спросить дальше /
вспомнить имя) + КАК сказать (конструкция). Конструкции не пишутся руками, а ИЗВЛЕКАЮТСЯ из ответов
бота в data/dialogs/ru.jsonl: слова ответа, совпадающие (по основе) со словами реплики пользователя,
→ слот [U]; числа → [N]; имя пользователя → [NAME]. Остаётся конструкция «давно [U]?», «сколько [U] в неделю?».
Выбор конструкции: Хебб-таблица «основы слов пользователя → конструкция» (счётчики), + акт реплики
(вопрос / приветствие / знакомство / утверждение), + заполняемость слотов. Заполнение [U] — самое редкое
(idf) слово пользователя с тем же окончанием, что было у слова в исходном слоте (грубая морфология из данных).
Память: «меня зовут X» → NAME; «как меня зовут» → ответ из памяти.

Числа (ДО запуска), тест = 20% диалогов: (1) по теме — доля ответов, где есть слово по основе из реплики
пользователя или из памяти; (2) разнообразие — уникальных ответов / всех; (3) число выученных конструкций.
Контроль: поиск ближайшей реплики (classic retrieval-бот: ответ бота на самую похожую реплику из train).
Успех: по теме ≥0.6 при разнообразии ≥0.7 и конструкций ≥100; 20 примеров печатаются без отбора.
"""
import json, os, re, time, collections, numpy as np
rng = np.random.default_rng(0)
ROOT = os.path.expanduser("~/Developer/mini-ai"); OUT = os.path.expanduser("~/Developer/mini-ai-runs/learning-core")
STOP = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только её мне было вот от меня еще нет о из ему
теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их
чем была сам чтоб без будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один почти мой тем чтобы нее сейчас были куда
зачем всех никогда можно при наконец два об другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя впрочем хорошо свою этой перед
иногда лучше чуть том нельзя такой им более всегда конечно всю между это мне тебе тебя у меня очень просто вроде типа ещё хочу хочешь нужно мой моя""".split())
NUM = re.compile(r"^\d+([.,]\d+)?$")
def toks(s): return re.findall(r"[а-яёa-z0-9]+(?:[.,]\d+)?|[,.!?—:;]", s.lower())
def stem(w): return w[:5] if len(w) > 5 else (w[:4] if len(w) > 4 else w)
def content(ws): return [w for w in ws if re.match(r"[а-яёa-z]", w) and w not in STOP and len(w) > 2]
def act(ws):
    s = " ".join(ws)
    if re.search(r"\b(привет|здравствуй|салам|салом|хай|добрый)\b", s): return "greet"
    if re.search(r"меня зовут|моё имя|мое имя|я —|зови меня", s): return "intro"
    if "?" in ws or re.search(r"\b(как|что|сколько|почему|где|когда|какой|какая|какие|можно|посоветуй|подскажи)\b", s): return "ask"
    return "say"

class Speaker:
    def __init__(self):
        self.cons = collections.Counter(); self.assoc = collections.defaultdict(collections.Counter)   # stem → конструкция
        self.by_act = collections.defaultdict(collections.Counter); self.df = collections.Counter(); self.n = 0
        self.slot_end = collections.defaultdict(collections.Counter)                                  # конструкция → окончания слова в [U]
    def abstract(self, user, reply, name):
        us = {stem(w): w for w in content(user)}; out = []
        for w in reply:
            if NUM.match(w): out.append("[N]")
            elif name and w == name.lower(): out.append("[NAME]")
            elif re.match(r"[а-яё]", w) and stem(w) in us and w not in STOP: out.append("[U]"); self.slot_end[None][w[-2:]] += 1
            else: out.append(w)
        return out
    def learn(self, user, reply, name):
        self.n += 1
        for s in set(stem(w) for w in content(user)): self.df[s] += 1
        # конструкции — по предложениям ответа
        sent, cur = [], []
        for w in reply:
            cur.append(w)
            if w in ".!?": sent.append(cur); cur = []
        if cur: sent.append(cur)
        a = act(user)
        for s in sent:
            ab = tuple(self.abstract(user, s, name))
            if len(ab) < 2 or len(ab) > 14: continue
            self.cons[ab] += 1; self.by_act[a][ab] += 1
            for st in set(stem(w) for w in content(user)): self.assoc[st][ab] += 1
            ends = [w[-2:] for w, x in zip(s, ab) if x == "[U]"]
            for e in ends: self.slot_end[ab][e] += 1
    def idf(self, st): return np.log((1 + self.n) / (1 + self.df[st])) + 1
    def reply(self, user, mem):
        a = act(user); cw = content(user); nums = [w for w in user if NUM.match(w)]
        if a == "intro":
            m = re.search(r"(?:зовут|имя|зови меня)\s+(\w+)", " ".join(user))
            if m: mem["name"] = m.group(1)
        if re.search(r"как меня зовут|помнишь.*(имя|зовут)", " ".join(user)) and mem.get("name"): return f"тебя зовут {mem['name']}.", "memory"
        # оценка конструкций: ассоциация со словами пользователя (взвешенная idf) + акт + заполняемость
        score = collections.Counter()
        for w in cw:
            st = stem(w)
            for ab, c in self.assoc[st].items(): score[ab] += c * self.idf(st) / (self.cons[ab] ** 0.5)
        for ab, c in self.by_act[a].items(): score[ab] += 0.3 * c / (self.cons[ab] ** 0.5)
        ranked = sorted(score.items(), key=lambda t: -t[1])
        for ab, _ in ranked[:40]:
            has_u = "[U]" in ab; has_n = "[N]" in ab; has_name = "[NAME]" in ab
            if has_n and not nums: continue
            if has_name and not mem.get("name"): continue
            if has_u and not cw: continue
            words = []
            for x in ab:
                if x == "[U]":
                    ends = self.slot_end[ab]
                    cand = sorted(cw, key=lambda w: (-(ends[w[-2:]] > 0), -self.idf(stem(w))))
                    words.append(cand[0])
                elif x == "[N]": words.append(nums[0])
                elif x == "[NAME]": words.append(mem["name"])
                else: words.append(x)
            return detok(words), "construct"
        return detok(list(self.cons.most_common(1)[0][0])) if self.cons else "…", "fallback"

def detok(ws):
    s = re.sub(r"\s+([,.!?:;])", r"\1", " ".join(ws)); return s[:1].upper() + s[1:]

if __name__ == "__main__":
    t0 = time.time()
    dl = [json.loads(l) for l in open(os.path.join(ROOT, "data/dialogs/ru.jsonl"), encoding="utf-8")]; rng.shuffle(dl)
    n_te = len(dl) // 5; te, tr = dl[:n_te], dl[n_te:]
    def walk(d):
        name = None; t = d["turns"]
        for i in range(1, len(t)):
            if t[i]["role"] == "bot" and t[i - 1]["role"] == "user":
                u = toks(t[i - 1]["text"]); m = re.search(r"(?:зовут|имя|зови меня)\s+(\w+)", t[i - 1]["text"].lower())
                if m: name = m.group(1)
                yield u, toks(t[i]["text"]), name
    S = Speaker(); train_pairs = []
    for d in tr:
        for u, r, nm in walk(d): S.learn(u, r, nm); train_pairs.append((u, r))
    print(f"train пар {len(train_pairs)}, конструкций {len(S.cons)} (со слотом {sum('[U]' in c or '[N]' in c or '[NAME]' in c for c in S.cons)}), {time.time()-t0:.1f}s")
    print("частые конструкции со слотом:", "; ".join(detok(list(c)) for c, _ in S.cons.most_common(400) if "[U]" in c or "[N]" in c)[:600])
    # контроль: retrieval
    vocab = {}; 
    def vec(ws):
        v = collections.Counter(stem(w) for w in content(ws)); return v
    def sim(a, b):
        return sum(min(a[k], b[k]) for k in a if k in b) / (np.sqrt(sum(a.values())) * np.sqrt(sum(b.values())) + 1e-9)
    tr_vecs = [(vec(u), r) for u, r in train_pairs]
    def retrieve(u):
        vu = vec(u); best = max(tr_vecs, key=lambda t: sim(vu, t[0])); return detok(best[1])
    # тест
    stats = {"construct": [0, 0, []], "retrieval": [0, 0, []]}; kinds = collections.Counter(); examples = []
    for d in te:
        mem = {}
        for u, r, nm in walk(d):
            g, kind = S.reply(u, mem); kinds[kind] += 1; rt = retrieve(u)
            us = set(stem(w) for w in content(u)) | ({stem(mem["name"])} if mem.get("name") else set())
            for key, txt in (("construct", g), ("retrieval", rt)):
                on = bool(us & set(stem(w) for w in content(toks(txt)))); stats[key][0] += on; stats[key][1] += 1; stats[key][2].append(txt)
            if len(examples) < 20: examples.append((detok(u), g, rt, detok(r)))
    res = {}
    for key in stats:
        on, n, txts = stats[key]; res[key] = dict(on_topic=round(on / n, 3), diversity=round(len(set(txts)) / n, 3), n=n)
        print(f"{key}: по теме {res[key]['on_topic']:.3f} | разнообразие {res[key]['diversity']:.3f} (n={n})")
    print("виды ответов:", dict(kinds)); res["constructions"] = len(S.cons); res["kinds"] = dict(kinds)
    print("\n20 примеров без отбора (Ю / ИИ-сборка / контроль-поиск / эталон):")
    for u, g, rt, r in examples: print(f"  Ю: {u}\n  ИИ: {g}\n  поиск: {rt}\n  эталон: {r}\n")
    res["examples"] = examples; res["seconds"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(OUT, "exp56_construct_speech.json"), "w"), ensure_ascii=False, indent=1); print("sec", res["seconds"])
