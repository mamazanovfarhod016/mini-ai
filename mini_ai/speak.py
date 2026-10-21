"""Речь своими словами (exp39/40): из ФАКТОВ, не из чужих предложений. Без градиента.
Факт (X, rel, Y) + грамматика связи → фраза; несколько фактов → абзац:
  1. тип сущности — по ОПРЕДЕЛЕНИЮ (факт be: «is a city…» → city; «is a scientist…» → person), не по россыпи слов;
  2. первый факт — полное имя; дальше местоимение (he/she/it) через раз;
  3. факты одной связи сливаются: (X, have, A), (X, have, B) → «X has A and B»;
  4. порядок: определение → свойства → глаголы; время — past для умерших людей и событий с годом;
  5. Y чистится: обрывки и повторы выбрасываются.
exp40 (диалог): ОДИН факт → НЕСКОЛЬКО грамматических форм (phrase): прямая «X is Y», инверсия «The P of Z is X»,
  с оговоркой «As far as I know…», с источником «I read in the textbook that…», неуверенная «I'm not sure, but…».
  Выбор формы — по контексту разговора (dialog.py), не копирование исходного текста и без LLM.
"""
import re, collections
from mini_ai.reader import PAST_FORM

PERSON_DEF = {"scientist", "writer", "actor", "actress", "singer", "painter", "poet", "king", "queen", "emperor", "ruler",
              "president", "politician", "philosopher", "composer", "player", "footballer", "author", "physicist",
              "mathematician", "inventor", "general", "leader", "musician", "person", "man", "woman", "prince", "princess"}
FEMALE = {"actress", "queen", "princess", "woman", "she", "her"}
CITY_DEF = {"city", "town", "capital", "village", "municipality"}
COUNTRY_DEF = {"country", "republic", "kingdom", "nation", "state", "empire"}
THING_DEF = {"planet", "element", "compound", "antibiotic", "drug", "river", "mountain", "lake", "animal", "plant",
             "species", "language", "game", "film", "book", "song", "company", "metal", "star", "moon", "group", "substance"}
BAD_Y_END = {"the", "a", "an", "of", "in", "to", "by", "and", "or", "for", "with", "as", "at", "on", "from", "that", "which", "who",
             "small", "large", "big", "different", "same", "many", "more", "most", "other", "new", "old", "several", "very", "two", "three", "few",
             "than", "but", "if", "while", "when", "where", "because", "so", "about", "into", "over", "under", "between", "through", "during", "is", "are", "was", "were"}
BAD_Y = re.compile(r"^(?:one of|also|not|very|still|now|then|so|too|there|here|that|this|it|its|which|what|who|when|where|because|used|known|called)$")


def entity_type(facts):
    defs = " ".join(Y.lower() for _, rel, Y in facts if rel == "be")
    words = set(re.findall(r"[a-z]+", defs))
    if words & THING_DEF: return "it", "thing"                     # «largest planet» → it, даже если рядом «Zeus»
    if words & PERSON_DEF: return ("she" if words & FEMALE else "he"), "person"
    if words & CITY_DEF: return "it", "city"
    if words & COUNTRY_DEF: return "it", "country"
    return "it", "thing"


def clean_y(Y):
    Y = re.sub(r"\s+", " ", Y).strip(" ,;:")
    Y = re.sub(r"\s+(?:and|or|but|which|that|who|where|when|because|so|as|if|while|although|such)$", "", Y)
    w = Y.lower().split()
    if len(w) < 2 or BAD_Y.match(Y.lower()) or w[-1] in BAD_Y_END: return None
    if w[0] in ("a", "an", "the") and len(w) == 2 and w[1] in ("soft", "hard", "small", "large", "big"): return None   # «a soft,» обрывок
    return Y


def verb(rel, tense):
    if "_" in rel or rel.endswith("@year"): return None
    if tense == "past": return PAST_FORM.get(rel, rel + ("d" if rel.endswith("e") else "ed"))
    if rel == "be": return "is"
    if rel == "have": return "has"
    return rel + ("es" if rel.endswith(("s", "sh", "ch")) else "s")


def compose(X, facts, max_sent=5):
    """facts: [(X, rel, Y)] про одну сущность → абзац своими словами."""
    pron, etype = entity_type(facts)
    died = any(rel == "die" or "died" in Y.lower() for _, rel, Y in facts)
    tense_def = "past" if (etype == "person" and died) else "present"
    groups = collections.OrderedDict(); props = []
    for _, rel, Y in facts:
        if rel.endswith("@year"): continue
        y = clean_y(Y)
        if not y: continue
        if "_" in rel: props.append((rel.replace("_", " "), y)); continue
        if y.lower() not in [v.lower() for v in groups.setdefault(rel, [])]: groups[rel].append(y)
    order = sorted(groups, key=lambda r: (0 if r == "be" else 1 if r == "have" else 2))
    sents = []
    for rel in order:
        ys = groups[rel][:3]
        if not ys: continue
        subj = X if not sents else (pron.capitalize() if len(sents) % 2 else X)
        tense = tense_def if rel in ("be", "have") else "past"       # события — в прошедшем
        v = verb(rel, tense)
        if v is None: continue
        obj = ys[0] if len(ys) == 1 else ", ".join(ys[:-1]) + " and " + ys[-1]
        s = f"{subj} {v} {obj}"; sents.append(s[0].upper() + s[1:] + ".")
        if len(sents) >= max_sent: break
    for prop, y in props[:2]:
        if len(sents) >= max_sent + 1: break
        poss = {"he": "His", "she": "Her"}.get(pron, "Its")
        sents.append(f"{poss} {prop} is {y}.")
    return " ".join(sents)


# ---------- exp40: один факт — много форм ----------
STYLES = ("direct", "inverted", "hedge", "source")
SOURCE_INTRO = {"учебник": "I read in the textbook that", "выучено": "You told me that", "user": "You told me that",
                "заметки": "According to my notes,", "wiki": "I read on Wikipedia that"}


def is_plural(X):
    last = X.split()[-1].lower() if X.split() else ""
    return (last.endswith("s") and not last.endswith(("ss", "us", "is")) and (len(X.split()) > 1 or X[:1].islower())) or last in ("people", "children", "men", "women")


def split_prop(Y):
    """«the capital of Uzbekistan» → (capital, Uzbekistan); «a population of 3 million» → (population, 3 million)"""
    m = re.match(r"^(?:the|a|an)\s+((?:[\w\-]+\s){0,2}[\w\-]+?)\s+of\s+(.+)$", Y.strip(" ,."), re.I)
    if not m or m.group(1).lower() in ("one", "part", "lot", "lots", "number", "kind", "type", "sort", "member") or m.group(1).lower() in BAD_Y_END: return None
    return m.group(1), m.group(2).strip(" ,.")


def sentence(X, rel, Y, tense="present"):
    """прямая форма: подлежащее + глагол связи + Y (без финальной точки)"""
    y = clean_y(Y) or Y.strip(" ,.")
    if rel.startswith("be_called"): return f"{X} is called {y}"
    if "_" in rel and not rel.startswith("be_"): return f"The {rel.replace('_', ' ')} of {X} is {y}"
    v = verb(rel, tense) or rel
    if is_plural(X): v = {"is": "are", "has": "have", "was": "were"}.get(v, v)
    return f"{X} {v} {y}"


def inverted(X, rel, Y, tense="present"):
    """инверсия: Y-свойство вперёд. «Tashkent be the capital of Uzbekistan» → «The capital of Uzbekistan is Tashkent»"""
    sp = split_prop(Y)
    was = "was" if tense == "past" else "is"
    if sp and rel == "be": return f"The {sp[0]} of {sp[1]} {was} {X}"
    if sp and rel == "have": return f"The {sp[0]} of {X} {was} {sp[1]}"
    if "_" in rel and not rel.startswith("be_"): return f"{clean_y(Y) or Y} {was} the {rel.replace('_', ' ')} of {X}"
    return None


def _lc(s):
    """строчная первая буква, если это не имя: «The Nile» → «the Nile», «Tashkent» остаётся"""
    w = s.split()[0] if s.split() else ""
    return s[0].lower() + s[1:] if w in ("The", "A", "An", "It", "He", "She", "They", "As", "You", "I'm", "According", "All", "Most", "Some", "Many", "Each", "Every", "These", "This") else s


def _cap(s): return s[0].upper() + s[1:].rstrip(" .!") + "."


def phrase(X, rel, Y, style="direct", tense="present", src=None, prefix=""):
    """факт → фраза заданного стиля. style: direct|inverted|hedge|source|unsure; prefix: «As I said, » и т.п."""
    base = sentence(X, rel, Y, tense)
    if style == "inverted": s = inverted(X, rel, Y, tense) or base
    elif style == "hedge": s = "As far as I know, " + _lc(base)
    elif style == "unsure": s = "I'm not sure, but I think " + _lc(base)
    elif style == "source":
        intro = SOURCE_INTRO.get(src) or (SOURCE_INTRO["wiki"] if src else "I remember that")
        s = intro + " " + _lc(base)
    else: s = base
    if prefix: s = prefix + _lc(s)
    return _cap(s)


# ---------- exp40 п.0: реплики-собеседника (не поиск фактов) ----------
SMALL_TALK = {
    "greet": ["Hi! Ask me about something — a city, a planet, an element.", "Hello! What do you want to know?",
              "Hi there. I remember facts and can learn new ones — ask me anything.", "Привет! Спроси меня о чём-нибудь."],
    "thanks": ["You're welcome!", "Glad I could help.", "Any time.", "Пожалуйста!"],
    "how_are_you": ["I'm fine, thanks — I have {n} facts in memory and I'm ready for a question.",
                    "Good, thank you. What shall we talk about?", "All good. Ask me something or teach me a new fact."],
    "who": ["I'm Mini AI — a memory without gradients: I keep facts, build answers from them and learn from your sentences.",
            "My name is Mini AI. I answer from facts I have read, and I learn what you tell me."],
    "can": ["I can answer questions from the facts I remember, ask you back when I'm not sure, and learn any sentence you tell me.",
            "Ask me a question — I look for a fact and say it in my own words. If I don't know, I'll ask, and you can teach me in one sentence."],
    "bye": ["Bye! Come back with more questions.", "See you! I'll remember what you taught me.", "Пока!"],
}
SMALL_TALK_RE = [
    ("greet", r"^(?:hi|hello|hey|hiya|yo|good (?:morning|afternoon|evening|day)|привет|здравствуй(?:те)?|салам|salom|assalomu alaykum)[\s!.,]*$"),
    ("thanks", r"^(?:thanks|thank you|thx|ty|cheers|спасибо|рахмат|rahmat)[\s!.,]*(?:very much|a lot|so much)?[\s!.,]*$"),
    ("how_are_you", r"^(?:how are you|how are you doing|how's it going|how do you do|как дела|как ты)[\s?!.]*$"),
    ("who", r"^(?:who are you|what are you|what is your name|what's your name|кто ты|как тебя зовут)[\s?!.]*$"),
    ("can", r"^(?:what can you do|what do you do|what do you know|help|что ты умеешь|что ты можешь)[\s?!.]*$"),
    ("bye", r"^(?:bye|goodbye|see you|good night|пока|до свидания)[\s!.,]*$"),
]


def small_talk(q, rng=None, n_facts=0):
    """реплика без поиска фактов → (kind, text) или None. Приветствие в начале вопроса («Hi, what is…») срезается в dialog.py."""
    ql = re.sub(r"\s+", " ", q.strip().lower())
    for kind, rx in SMALL_TALK_RE:
        if re.match(rx, ql, re.I):
            opts = SMALL_TALK[kind]; t = (rng.choice(opts) if rng else opts[0])
            return kind, t.format(n=f"{n_facts:,}".replace(",", " "))
    return None
