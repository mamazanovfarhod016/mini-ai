"""Эксперимент 12 (раунд 2, экспериментатор-Б): закрыть дыры теоретика и критика по рассуждению.

(1) ТРУДНЫЕ ФАКТЫ. Три отношения с ОДИНАКОВОЙ подписью имя→имя: dost (друг), aka (брат), ustoz (учитель),
    плюс три предмета имя→число (olma, kitob, qalam). Вопросы из 3/4/5 слов = 2/3/4 шага:
    [X, r1, obj], [X, r1, r2, obj], [X, r1, r2, r3, obj] — на каждом шаге ≥3 допустимых по памяти пар.
    Сравниваем решатели без плана:
      backtrack      — перебор пар с откатом, без типов (контроль exp9)
      dna_look       — ДНК типов (bind) + просмотр вперёд по типам (exp9 bind_look)
      field_learned  — типы как поле, таблица связей выведена из памяти (exp9)
      pos_look       — конструкция B теоретика: вопрос = ОДИН вектор Σ code(w_i)⊛p_i (p_i — код позиции);
                       решатель сам достаёт слова и их позиции через unbind + сито, затем ДНК типов +
                       правило «тратить от ближайшего к имени X»
      pos_only       — контроль: тот же вектор позиций, но БЕЗ типов: цепочка от позиции 0 подряд
    Порядки слов: uz = [X, r1, r2, obj] и ru = [obj, r2, r1, X] («olma у брата друга X»).
(2) КОНТРОЛЬ КРИТИКА: в exp8 выбор пары заменён на случайный — должно дать те же ~50%.
(3) ОТКРЫТЫЙ СЛОВАРЬ ОТВЕТОВ: ответы — не 97 чисел, а тысячи разных строк (имена + числа + слова из ru.txt);
    чтение через сито (softmax-сумма значений → ближайшее слово словаря) — не падает ли сито наоборот.
(4) МНОГОЗНАЧНОСТЬ: одно слово в двух ролях (Olma — имя и olma — предмет и т.д.) — ломается ли.
Запуск: cd ~/Developer/mini-ai-memory/wt && python3 tools/exp12_hard_facts.py
"""
import sys, os, json, random, time, re, zlib, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from mini_ai.memory_core import unit
from tools.exp3_wordcode import letter_code
from tools.facts_unique import make_unique, OUT
from tools.exp9_dna_types import (TW, Memory, bind, unbind, content, tcode, enc_plain, enc_bind, enc_concat,
                                  solve_backtrack, solve_typed, solve_exp8, reducible, allowed, make_field_solver, D)

REL = ["dost", "aka", "ustoz"]          # все три: name⊛rel → name (одинаковая подпись!)
OBJ = ["olma", "kitob", "qalam"]        # name⊛obj → num
THR = 0.5
LOG = None


def log(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True)
    if LOG: LOG.write(s + "\n"); LOG.flush()


nm = lambda s: TW("name", s); rl = lambda r: TW("rel", r); ob = lambda o: TW("obj", o); num = lambda n: TW("num", n)


# ---------- данные ----------
def build(N, seed=7, extra_names=()):
    """Имена как в exp4–9 (+ extra_names), 3 отношения-перестановки, 3 предмета со счётом. Вопросы на 2/3/4 шага."""
    names = [f["name"] for f in make_unique(N)] + list(extra_names)
    rng = random.Random(seed)
    rel = {}
    for r in REL:
        perm = names[:]; rng.shuffle(perm); rel[r] = dict(zip(names, perm))
    count = {(a, o): str(rng.randrange(2, 99)) for a in names for o in OBJ}
    qs = []
    for a in names:
        o = rng.choice(OBJ)
        r1, r2, r3 = rng.sample(REL, 3)
        b = rel[r1][a]; c = rel[r2][b]; d = rel[r3][c]
        qs.append({"name": a, "obj": o, "rels": (r1, r2, r3),
                   "gold2": count[(b, o)], "gold3": count[(c, o)], "gold4": count[(d, o)]})
    return names, rel, count, qs


def fill(mem, names, rel, count):
    for r in REL:
        for a, b in rel[r].items(): mem.write([nm(a), rl(r)], nm(b))
    for (a, o), n in count.items(): mem.write([nm(a), ob(o)], num(n))


def question(q, hops, order="uz"):
    """Список типизированных слов вопроса. uz: [X, r1, r2, obj]; ru: [obj, r2, r1, X] («olma у брата друга X»)."""
    rels = [rl(r) for r in q["rels"][:hops - 1]]
    ws = [nm(q["name"])] + rels + [ob(q["obj"])]
    return ws if order == "uz" else ws[::-1]


# ---------- конструкция B: код позиции ⊛p_i ----------
_pos = {}
def pcode(i):
    if i not in _pos: _pos[i] = unit(np.random.default_rng(1000 + i).standard_normal(D))
    return _pos[i]


def qvec(enc, ws):
    """Вопрос как ОДИН вектор: Σ_i enc(w_i) ⊛ p_i."""
    return unit(sum(bind(enc(w), pcode(i)) for i, w in enumerate(ws)))


class Vocab:
    """Сито для слов вопроса: все имена, отношения, предметы с их типами (тип — из словаря сита)."""
    def __init__(self, enc, names):
        self.items = [nm(a) for a in names] + [rl(r) for r in REL] + [ob(o) for o in OBJ]
        self.M = np.stack([enc(x) for x in self.items])

    def clean(self, v):
        s = self.M @ v; i = int(s.argmax()); return self.items[i], float(s[i])


def decode_positions(qv, vocab, maxlen=5, thr=0.3):
    """Достаём слова по позициям: unbind(q, p_i) → ближайшее слово словаря. Стоп, когда сходство падает."""
    out = []
    for i in range(maxlen):
        w, s = vocab.clean(unbind(qv, pcode(i)))
        if s < thr: break
        out.append((i, w))
    return out


def solve_pos_typed(mem, qv, vocab, gram=allowed, max_steps=6):
    """Типы + позиции: пары только допустимые по типам и сводимые по просмотру вперёд; среди них —
    та, где второе слово БЛИЖЕ ВСЕГО по позиции к текущей сущности (имени X). Результат наследует позицию."""
    dec = decode_positions(qv, vocab)
    words = [(p, w) for p, w in dec]
    for _ in range(max_steps):
        if len(words) < 2: return None
        pairs = []
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                r = gram(words[i][1].t, words[j][1].t)
                if not r: continue
                rest = [w.t for k, (p, w) in enumerate(words) if k not in (i, j)]
                if not reducible([r] + rest, gram): continue
                pairs.append((abs(words[i][0] - words[j][0]), i, j, r))
        if not pairs: return None
        _, i, j, r = min(pairs)                          # ближайшая по позиции допустимая пара
        s, a2, ta = mem.best([words[i][1], words[j][1]])
        if s < THR: return None
        if len(words) == 2: return a2
        # результат встаёт на место имени (той из двух, у кого тип name), остальные позиции не меняются
        pi = words[i][0] if words[i][1].t == "name" else words[j][0]
        words = [(pi, TW(r, a2))] + [x for k, x in enumerate(words) if k not in (i, j)]
    return None


def solve_pos_only(mem, qv, vocab):
    """Контроль без типов: читаем слова по позициям и связываем подряд от позиции 0."""
    dec = decode_positions(qv, vocab)
    if len(dec) < 2: return None
    cur = TW("?", dec[0][1].w)
    for p, w in dec[1:]:
        s, a2, ta = mem.best([cur, TW("?", w.w)])
        if s < THR: return None
        cur = TW("?", a2)
    return cur.w


# ---------- (2) контроль критика: случайный выбор пары в exp8 ----------
def solve_exp8_random(mem, words, rng, thr=THR, max_steps=5):
    words = list(words)
    for _ in range(max_steps):
        if len(words) == 2:
            s, ans, ta = mem.best(words); return ans if s >= thr else None
        cand = [(i, j) for i in range(len(words)) for j in range(i + 1, len(words)) if mem.best([words[i], words[j]])[0] >= thr]
        if not cand: return None
        i, j = rng.choice(cand)
        a2 = mem.best([words[i], words[j]])[1]
        words = [TW("?", a2)] + [w for k, w in enumerate(words) if k not in (i, j)]
    return None


# ---------- (1) основной прогон ----------
def run_hard(N, variants=("backtrack", "dna_look", "field_learned", "pos_look", "pos_only")):
    names, rel, count, qs = build(N)
    res = {}
    for var in variants:
        t0 = time.time()
        enc = enc_plain if var in ("backtrack", "field_learned", "pos_only") else enc_bind
        mem = Memory(enc); fill(mem, names, rel, count); base = len(mem.S)
        vocab = Vocab(enc, names) if var.startswith("pos") else None
        gram_f = make_field_solver(mem) if var == "field_learned" else None
        orders = ("uz", "ru") if var.startswith("pos") else ("uz",)
        r = {"records": base}
        for order in orders:
            for hops in (2, 3, 4):
                ok = 0; probes = []; posok = 0
                for q in qs:
                    ws = question(q, hops, order); gold = q[f"gold{hops}"]
                    p0 = mem.probes
                    if var == "backtrack": ans = solve_backtrack(mem, ws, THR)
                    elif var == "dna_look": ans = solve_typed(mem, ws, look=True, thr=THR)
                    elif var == "field_learned": ans = gram_f(ws)
                    else:
                        qv = qvec(enc, ws)
                        dec = decode_positions(qv, vocab)
                        posok += [w.w for p, w in dec] == [w.w for w in ws]
                        ans = solve_pos_typed(mem, qv, vocab) if var == "pos_look" else solve_pos_only(mem, qv, vocab)
                    probes.append(mem.probes - p0); ok += ans == gold
                key = f"hop{hops}" + ("" if order == "uz" else "_ru")
                r[key] = ok / len(qs); r["probes_" + key] = float(np.mean(probes))
                if var.startswith("pos"): r["posdecode_" + key] = posok / len(qs)
        r["sec"] = round(time.time() - t0, 1)
        res[var] = r
        s = f"  {var:14} N={N:4} ({base}): " + "  ".join(f"{h[-1]} шага {r[h]*100:5.1f}% ({r['probes_'+h]:.1f} проб)" for h in ("hop2", "hop3", "hop4"))
        if "hop2_ru" in r: s += " | ru-порядок: " + "  ".join(f"{r[h+'_ru']*100:5.1f}%" for h in ("hop2", "hop3", "hop4")) + \
                           f" | позиции прочитаны {min(r[k] for k in r if k.startswith('posdecode'))*100:.1f}%"
        log(s + f"  ({r['sec']} s)")
    return res


# ---------- (1b) диагностика: сколько допустимых пар и равны ли их сходства ----------
def diag_pairs(N=300):
    names, rel, count, qs = build(N)
    mem = Memory(enc_bind); fill(mem, names, rel, count)
    nadm = []; ties = 0
    for q in qs[:100]:
        ws = question(q, 4)
        sims = []
        for i in range(len(ws)):
            for j in range(i + 1, len(ws)):
                s, a2, ta = mem.best([ws[i], ws[j]])
                if s >= THR: sims.append(round(s, 3))
        nadm.append(len(sims)); ties += (len(set(sims)) < len(sims))
    log(f"  диагностика (N={N}, вопрос 5 слов): допустимых по памяти пар на 1-м шаге в среднем {np.mean(nadm):.2f}, "
        f"из них со сходством ≈1.0 одновременно в {ties}/100 вопросов")
    return {"admissible_pairs_step1": float(np.mean(nadm)), "tie_questions": ties}


# ---------- (2) exp8 со случайным выбором пары ----------
def run_random_control():
    out = {}
    for N in (100, 300, 1000):
        facts = make_unique(N); names = [f["name"] for f in facts]
        rng = random.Random(7); perm = names[:]; rng.shuffle(perm); friend = dict(zip(names, perm))
        count = {(a, o): str(rng.randrange(2, 99)) for a in names for o in OBJ}
        qs = [(a, rng.choice(OBJ)) for a in names]
        mem = Memory(enc_plain)
        for a, b in friend.items(): mem.write([nm(a), rl("dost")], nm(b))
        for (a, o), n in count.items(): mem.write([nm(a), ob(o)], num(n))
        rr = random.Random(123)
        g2 = g3 = r2 = r3 = 0
        for a, o in qs:
            g2 += solve_exp8(mem, [nm(a), rl("dost"), ob(o)]) == count[(friend[a], o)]
            g3 += solve_exp8(mem, [nm(a), rl("dost"), rl("dost"), ob(o)]) == count[(friend[friend[a]], o)]
            r2 += solve_exp8_random(mem, [nm(a), rl("dost"), ob(o)], rr) == count[(friend[a], o)]
            r3 += solve_exp8_random(mem, [nm(a), rl("dost"), rl("dost"), ob(o)], rr) == count[(friend[friend[a]], o)]
        out[N] = {"greedy_hop2": g2 / N, "greedy_hop3": g3 / N, "random_hop2": r2 / N, "random_hop3": r3 / N}
        log(f"  N={N:4}: exp8 жадный (max по сходству) 2 шага {g2/N*100:5.1f}%  3 шага {g3/N*100:5.1f}%   |   "
            f"случайная пара: 2 шага {r2/N*100:5.1f}%  3 шага {r3/N*100:5.1f}%")
    return out


# ---------- (3) открытый словарь ответов ----------
def ru_words(k):
    c = {}
    for line in open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data/corpus/ru.txt"), encoding="utf-8"):
        for w in re.findall(r"[а-яё]+", line.lower()):
            if len(w) >= 3: c[w] = c.get(w, 0) + 1
    return [w for w, _ in sorted(c.items(), key=lambda x: -x[1])][:k]


class SieveMemory(Memory):
    """Чтение через сито: softmax-сумма векторов значений → ближайшая строка словаря ответов (не argmax записи)."""
    def set_dict(self, strings, enc_s):
        self.DS = strings; self.DM = np.stack([enc_s(s) for s in strings]); self._VM = None

    def read_sieve(self, words, beta):
        self.probes += 1
        k = self.key(words); s = self.KM() @ k
        if self._VM is None: self._VM = np.stack(self.V)
        w = np.exp(beta * (s - s.max())); w /= w.sum()
        v = unit(w @ self._VM)
        i = int((self.DM @ v).argmax()); return float(s.max()), self.DS[i]


def run_open_vocab(N=1000):
    names, rel, count, qs = build(N)
    nums = [str(i) for i in range(2, 99)]
    ru = ru_words(8000)
    tiers = {"числа(97)": nums, "имена+числа": names + nums, "+ru 2000": names + nums + ru[:2000], "+ru все": names + nums + ru}
    out = {}
    for tier, dic in tiers.items():
        dic = list(dict.fromkeys(dic))
        rng = random.Random(5)
        # ответы на (имя, предмет) — случайные строки из словаря этого уровня (открытый словарь ответов)
        ans = {(a, o): rng.choice(dic) for a in names for o in OBJ}
        # сито наоборот: код → строка. Коллизии буквенного хэша и ближайшие соседи
        M = np.stack([content(s) for s in dic])
        G = M @ M.T; np.fill_diagonal(G, -1); nn = G.max(1)
        roundtrip = float(np.mean([dic[int((M @ M[i]).argmax())] == dic[i] for i in range(len(dic))]))
        r = {"dict": len(dic), "nn_cos_mean": float(nn.mean()), "nn_cos>0.9": float((nn > 0.9).mean()),
             "nn_cos>0.99": float((nn > 0.99).mean()), "roundtrip": roundtrip}
        for enc_name, enc in (("plain", enc_plain), ("bind", enc_bind)):
            mem = SieveMemory(enc)
            for rr in REL:
                for a, b in rel[rr].items(): mem.write([nm(a), rl(rr)], nm(b))
            for (a, o), s in ans.items(): mem.write([nm(a), ob(o)], TW("ans", s))
            mem.set_dict(dic, lambda s: enc(TW("ans", s)))
            # промежуточные имена тоже читаем через сито по словарю имён
            memN = SieveMemory(enc); memN.K, memN.V, memN.S, memN.KW, memN.T, memN.TA = mem.K, mem.V, mem.S, mem.KW, mem.T, mem.TA
            memN.set_dict(names, lambda s: enc(nm(s)))
            for beta in (16, 64):
                ok1 = ok2 = okarg = 0
                for q in qs:
                    a, o, r1 = q["name"], q["obj"], q["rels"][0]
                    ok1 += mem.read_sieve([nm(a), ob(o)], beta)[1] == ans[(a, o)]
                    okarg += mem.best([nm(a), ob(o)])[1] == ans[(a, o)]
                    b = memN.read_sieve([nm(a), rl(r1)], beta)[1]
                    ok2 += mem.read_sieve([nm(b), ob(o)], beta)[1] == ans[(rel[r1][a], o)]
                r[f"{enc_name}_b{beta}_hop1"] = ok1 / N; r[f"{enc_name}_b{beta}_hop2plan"] = ok2 / N
                r[f"{enc_name}_argmax_hop1"] = okarg / N
        out[tier] = r
        log(f"  словарь {tier:12} ({len(dic):5}): соседи cos>0.9 {r['nn_cos>0.9']*100:4.1f}%  cos>0.99 {r['nn_cos>0.99']*100:4.1f}%  "
            f"код→строка {roundtrip*100:5.1f}% | argmax-запись 1 шаг {r['plain_argmax_hop1']*100:5.1f}% | сито β=16: 1 шаг {r['plain_b16_hop1']*100:5.1f}% "
            f"2 шага {r['plain_b16_hop2plan']*100:5.1f}% | β=64: 1 шаг {r['plain_b64_hop1']*100:5.1f}% 2 шага {r['plain_b64_hop2plan']*100:5.1f}% "
            f"| bind β=64: {r['bind_b64_hop1']*100:5.1f}% / {r['bind_b64_hop2plan']*100:5.1f}%")
    return out


# ---------- (4) многозначность ----------
def run_polysemy(N=300):
    amb = ["Olma", "Kitob", "Qalam", "Dost", "Aka", "Ustoz"]        # те же буквы, что у предметов и отношений
    names, rel, count, qs = build(N, extra_names=amb)
    out = {}
    # коллизия ключей: (Olma:name, kitob:obj) vs (Kitob:name, olma:obj)
    kp = lambda enc, a, b: float(bind(enc(nm(a)), enc(ob(b))) @ bind(enc(nm(b.capitalize())), enc(ob(a.lower()))))
    out["key_collision_plain"] = kp(enc_plain, "Olma", "kitob"); out["key_collision_bind"] = kp(enc_bind, "Olma", "kitob")
    log(f"  сходство ключей (Olma,kitob) и (Kitob,olma): без типов {out['key_collision_plain']:.3f}, с ДНК типов {out['key_collision_bind']:.3f}")
    # вопросы: про многозначные имена (как X) и где многозначное имя — промежуточный ответ
    qa = [q for q in qs if q["name"] in amb]
    qmid = [q for q in qs if rel[q["rels"][0]][q["name"]] in amb]
    for var in ("backtrack", "exp8_greedy", "dna_look", "concat_look", "field_learned", "pos_look"):
        enc = {"backtrack": enc_plain, "exp8_greedy": enc_plain, "field_learned": enc_plain, "concat_look": enc_concat}.get(var, enc_bind)
        mem = Memory(enc); fill(mem, names, rel, count)
        vocab = Vocab(enc, names)
        gram_f = make_field_solver(mem) if var == "field_learned" else None
        def solve(ws):
            if var == "backtrack": return solve_backtrack(mem, ws, THR)
            if var == "exp8_greedy": return solve_exp8(mem, ws)
            if var == "field_learned": return gram_f(ws)
            if var == "pos_look": return solve_pos_typed(mem, qvec(enc, ws), vocab)
            return solve_typed(mem, ws, look=True, thr=THR)
        r = {}
        # 1 шаг: все 18 пар (многозначное имя, предмет), включая (Olma, kitob) против (Kitob, olma)
        r["hop1_amb"] = float(np.mean([mem.best([nm(a), ob(o)])[1] == count[(a, o)] for a in amb for o in OBJ]))
        r["hop1_all"] = float(np.mean([mem.best([nm(a), ob(o)])[1] == count[(a, o)] for a in names for o in OBJ]))
        for hops in (2, 3):
            r[f"hop{hops}_amb_as_X"] = float(np.mean([solve(question(q, hops)) == q[f"gold{hops}"] for q in qa]))
            r[f"hop{hops}_amb_mid"] = float(np.mean([solve(question(q, hops)) == q[f"gold{hops}"] for q in qmid]))
            r[f"hop{hops}_all"] = float(np.mean([solve(question(q, hops)) == q[f"gold{hops}"] for q in qs]))
        out[var] = r
        log(f"  {var:14}: 1 шаг многозн. {r['hop1_amb']*100:5.1f}% (все {r['hop1_all']*100:5.1f}%) | X многозначно: 2 шага {r['hop2_amb_as_X']*100:5.1f}% "
            f"3 шага {r['hop3_amb_as_X']*100:5.1f}% | многозн. в середине ({len(qmid)} в.): 2 шага {r['hop2_amb_mid']*100:5.1f}% 3 шага {r['hop3_amb_mid']*100:5.1f}% "
            f"| все {N+6}: 2 шага {r['hop2_all']*100:5.1f}% 3 шага {r['hop3_all']*100:5.1f}%")
    return out


# ---------- (1c) контроль на код позиции: перемешанный порядок слов и порядок «r1 X r2 obj» ----------
def run_pos_order_control(N=300):
    """Код позиции = соседство слов. Если слова вопроса перемешать случайно — правило «ближайшая пара» теряет смысл.
    Это честная граница конструкции B: порядок слов (синтаксис) и есть «план», зашитый в вопрос."""
    names, rel, count, qs = build(N)
    mem = Memory(enc_bind); fill(mem, names, rel, count); vocab = Vocab(enc_bind, names)
    rng = random.Random(3); out = {}
    for order in ("uz", "ru", "shuffle", "interleave"):
        for hops in (3, 4):
            ok = 0
            for q in qs:
                ws = question(q, hops, "uz")
                if order == "ru": ws = ws[::-1]
                elif order == "shuffle": ws = ws[:]; rng.shuffle(ws)
                elif order == "interleave": ws = [ws[1], ws[0]] + ws[2:]        # «друга X брата — olma» (r1 перед X)
                ok += solve_pos_typed(mem, qvec(enc_bind, ws), vocab) == q[f"gold{hops}"]
            out[f"{order}_hop{hops}"] = ok / N
        log(f"  pos_look, порядок {order:10}: 3 шага {out[order+'_hop3']*100:5.1f}%  4 шага {out[order+'_hop4']*100:5.1f}%")
    return out


def run_polysemy_pairs(N=300):
    """Какие именно 1-шаговые пары с многозначным именем ломаются и почему (ключи совпадают?)."""
    amb = ["Olma", "Kitob", "Qalam", "Dost", "Aka", "Ustoz"]
    names, rel, count, qs = build(N, extra_names=amb)
    out = {}
    for enc_name, enc in (("plain", enc_plain), ("bind", enc_bind), ("concat", enc_concat)):
        mem = Memory(enc); fill(mem, names, rel, count)
        bad = []
        for a in amb:
            for o in OBJ:
                s, ans, ta = mem.best([nm(a), ob(o)])
                if ans != count[(a, o)]:
                    i = int((mem.KM() @ mem.key([nm(a), ob(o)])).argmax()); bad.append(f"({a},{o})→запись {mem.KW[i]}")
        out[enc_name] = bad
        log(f"  {enc_name:6}: сломаны {len(bad)}/18: " + "; ".join(bad))
    return out


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "addenda":
    LOG = open(f"{OUT}/exp12.log", "a")
    res = json.load(open(f"{OUT}/exp12_hard_facts.json"))
    log("=== (1c) код позиции: зависимость от порядка слов (N=300) ===")
    res["pos_order_control"] = run_pos_order_control(300)
    log("=== (4b) многозначность: какие пары ломаются в 1 шаг ===")
    res["polysemy_pairs"] = run_polysemy_pairs(300)
    json.dump(res, open(f"{OUT}/exp12_hard_facts.json", "w"), indent=1, ensure_ascii=False)
    LOG.close(); sys.exit(0)

if __name__ == "__main__":
    LOG = open(f"{OUT}/exp12.log", "w")
    T0 = time.time(); res = {}
    log("=== (1) трудные факты: 3 отношения одной подписи (dost, aka, ustoz) + 3 предмета; вопросы 3/4/5 слов ===")
    res["diag"] = diag_pairs(300)
    res["hard"] = {}
    for N in (100, 300, 1000):
        log(f"N={N}"); res["hard"][N] = run_hard(N)
    log("=== (2) контроль критика: exp8 с случайным выбором пары ===")
    res["random_control"] = run_random_control()
    log("=== (3) открытый словарь ответов (N=1000 имён, 6000 записей; ответы — случайные строки словаря) ===")
    res["open_vocab"] = run_open_vocab(1000)
    log("=== (4) многозначность: имена Olma/Kitob/Qalam/Dost/Aka/Ustoz среди 300 имён ===")
    res["polysemy"] = run_polysemy(300)
    log(f"всего {time.time()-T0:.0f} s")
    json.dump(res, open(f"{OUT}/exp12_hard_facts.json", "w"), indent=1, ensure_ascii=False)
    LOG.close()
