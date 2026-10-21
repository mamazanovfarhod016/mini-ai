"""Понимание вопроса (exp45): вопрос → РАМКА с пустым слотом, не мешок слов. Без градиента.
«The electrode at which oxidation occurs is called ?»  → (X=«the electrode at which oxidation occurs», rel=be called, Y=?)
«What contains positive protons and neutral neutrons?» → (X=?, rel=contain, Y=«positive protons and neutral neutrons»)
«What type of organism is used in cheese?»            → (X=?, type=organism, rel=used in, Y=cheese)
Разбор — по служебным словам (закрытый класс, как exp18): вопросное слово определяет, где слот;
глагол (is/are/called/contains/…) делит на X и Y. Затем ОТВЕТ = факт из памяти с той же рамкой:
  слот X → ищем факты, чей Y покрывает Y вопроса И rel совпадает → ответ = X факта;
  слот Y → ищем факты, чей X покрывает X вопроса И rel совпадает → ответ = Y факта.
Покрытие считается по ОБЯЗАТЕЛЬНЫМ словам: все содержательные слова заполненной части вопроса должны
быть в факте (oxidation ≠ reduction — факт без «oxidation» отпадает), а не «похожесть».
Для вариантов (SciQ): вариант принимается, если его слова есть в найденном ответе-части факта.
"""
import re
from mini_ai.memory_v2 import toks, content_words

QWORD = re.compile(r"^(?:what|which|who|where|when|how many|how)\b\s*(?:type of|kind of|is the|is a|is an|are the|are|is|was|were|do|does|did)?\s*", re.I)
VERBS = r"(?:is called|are called|is known as|is|are|was|were|contains?|includes?|consists? of|produces?|causes?|forms?|makes?|means?|has|have|occurs? (?:in|at|when)|results? in|leads? to|comes? from|get|gets|use|uses|used|called|found in|made (?:up )?of|composed of|located in|refers? to|describes?|begins? with|ends? with|enters?|exhibits?|reproduce(?:s)? through|end their lives)"
V_RE = re.compile(rf"\b({VERBS})\b", re.I)


def parse_question(q):
    """→ dict(slot='X'|'Y', X=words, rel=str, Y=words, atype=str|None). None если не разобрали."""
    q = q.strip().rstrip("?").strip(); ql = q.lower()
    atype = None
    m = re.match(r"^(?:what|which)\s+(?:type|kind)\s+of\s+(\w+)", ql)
    m2 = re.match(r"^(?:what|which)\s+(\w+)\s+(?:is|are|was|were|do|does|did|can|has|have)\b", ql)
    if m: atype = m.group(1)
    elif m2 and m2.group(1) not in ("is", "are", "was", "were", "do", "does", "did", "can", "has", "have", "type", "kind"): atype = m2.group(1)
    body = QWORD.sub("", ql, count=1).strip()
    vm = V_RE.search(body)
    starts_with_q = bool(re.match(r"^(?:what|which|who|where|when|how)\b", ql))
    # «what is the X of Y» / «who is the X of Y» → слот X, обязательные слова = X of Y (ищем факт, чей Y содержит их)
    m_of = re.match(r"^(?:what|who|which)\s+(?:is|are|was|were)\s+(?:the\s+)?([\w\s'\-]+?\s+(?:of|in|for|from)\s+[\w\s'\-]+)$", ql)
    if m_of and not re.search(r"\b(?:called|known)\b", ql):
        return dict(slot="X", X=[], rel="be", Y=content_words(toks(m_of.group(1))), atype=atype)
    if ql.endswith(("called", "known as", "called the", "referred to as")) or re.search(r"\b(?:is|are) called$", ql):
        # «X ... is called ?» / «what is X called» → слот Y
        X = re.sub(r"\b(?:is|are) (?:called|known as)$|\bcalled$", "", body).strip()
        return dict(slot="Y", X=[w for w in content_words(toks(X)) if w != "called"], rel="be_called", Y=[], atype=atype)
    if not vm:
        return dict(slot="Y" if not starts_with_q else "X", X=content_words(toks(body)), rel="", Y=[], atype=atype)
    rel = re.sub(r"\s+", "_", vm.group(1).lower()); left, right = body[:vm.start()].strip(), body[vm.end():].strip()
    if starts_with_q and not left:                     # «What contains A?» → X=?, Y=A
        return dict(slot="X", X=[], rel=rel, Y=content_words(toks(right)), atype=atype)
    if starts_with_q and left and not right:           # «What type of organism is used in cheese» → X=?, Y=left
        return dict(slot="X", X=[], rel=rel, Y=content_words(toks(left)), atype=atype)
    if right.endswith(("of", "from", "in", "by", "to", "with", "through")) or not right or right in ("the", "a", "an"):
        return dict(slot="Y", X=content_words(toks(left)), rel=rel, Y=content_words(toks(right)), atype=atype)   # «Natural gas is made up of ?»
    if starts_with_q:                                  # «What X is Y» — слот X, Y известен
        return dict(slot="X", X=[], rel=rel, Y=content_words(toks(left + " " + right)), atype=atype)
    return dict(slot="Y", X=content_words(toks(left)), rel=rel, Y=content_words(toks(right)), atype=atype)


REL_SYN = {"be_called": {"be", "be_called", "called", "be_known_as"}, "be": {"be", "be_called"}, "contain": {"contain", "have", "include", "be_made_of", "be_composed_of"},
           "have": {"have", "contain"}, "make": {"make", "produce", "form"}, "produce": {"produce", "make", "form"}}
def rel_ok(q_rel, f_rel):
    if not q_rel: return True
    base = q_rel.split("_")[0] if q_rel not in REL_SYN else q_rel
    return f_rel in REL_SYN.get(q_rel, {q_rel, base}) or f_rel.startswith(base)


def answer_by_frame(frame, facts_idx, idf):
    """facts_idx: список (Xw:set, rel:str, Yw:set, X:str, Y:str, sent_i). Возвращает [(score, answer_text, fact)]."""
    known = set(frame["Y"] if frame["slot"] == "X" else frame["X"]); out = []
    if not known: return out
    kidf = sum(idf.get(w, 1.0) for w in known)
    for Xw, rel, Yw, X, Y, si in facts_idx:
        part = Yw if frame["slot"] == "X" else Xw; other = Xw if frame["slot"] == "X" else Yw
        allw = Xw | Yw
        cov = sum(idf.get(w, 1.0) for w in known & allw) / kidf
        if cov < 0.99: continue                                   # ВСЕ обязательные слова — иначе это другой факт
        s = 1.0 + (0.5 if rel_ok(frame["rel"], rel) else 0.0) + (0.3 if known <= part else 0.0)
        ans = X if frame["slot"] == "X" else Y
        if frame["atype"] and frame["atype"] not in toks(ans) and frame["atype"] not in toks(X + " " + Y): s -= 0.2
        out.append((s, ans, (X, rel, Y, si)))
    out.sort(key=lambda t: -t[0]); return out
