"""Речь своими словами (exp39/40): из ФАКТОВ, не из чужих предложений. Без градиента.
Факт (X, rel, Y) + грамматика связи → фраза; несколько фактов → абзац:
  1. тип сущности — по ОПРЕДЕЛЕНИЮ (факт be: «is a city…» → city; «is a scientist…» → person), не по россыпи слов;
  2. первый факт — полное имя; дальше местоимение (he/she/it) через раз;
  3. факты одной связи сливаются: (X, have, A), (X, have, B) → «X has A and B»;
  4. порядок: определение → свойства → глаголы; время — past для умерших людей и событий с годом;
  5. Y чистится: обрывки и повторы выбрасываются.
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
BAD_Y_END = {"the", "a", "an", "of", "in", "to", "by", "and", "or", "for", "with", "as", "at", "on", "from", "that", "which", "who"}
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
