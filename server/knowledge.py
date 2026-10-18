"""
База знаний самообучения (server_data/knowledge.json) — «темы», а не отдельные ответы.

Тема = один смысл: ответ (ru/uz) + 8–15 РАЗНЫХ формулировок вопроса.
  • ~70% формулировок идут в обучение (train),
  • ~30% прячутся (heldout) — на них экзамен: засчитываем знание, только если модель
    отвечает на формулировки, которых НИКОГДА не видела. Это проверка «понял», а не «зазубрил».
Новый вопрос на уже известную тему (например, «погода в Ташкенте» к теме «погода») —
формулировки добавляются в эту тему, а не плодят новую.
"""
import random
import re
import time
import unicodedata
import zlib
from pathlib import Path

from server import store

MIN_PARA = 8
HELDOUT_FRAC = 0.3


def clean_q(q):
    t = str(q).lower().replace("ё", "е")
    for a in "`’‘ʻʼ´":
        t = t.replace(a, "'")
    t = re.sub(r"[!?.,;:()\"«»…—–-]+", " ", t)
    return " ".join(t.split())


def clean_answer(a):
    """Ударения из Википедии (Пари́ж) убираем — их нет в словаре модели."""
    a = unicodedata.normalize("NFD", str(a)).replace("\u0301", "").replace("\u0300", "")
    a = unicodedata.normalize("NFC", a)
    return " ".join(a.split())


def slug(text):
    tr = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюя", "a b v g d e e zh z i y k l m n o p r s t u f h c ch sh sch _ y _ e yu ya".split()))
    s = "".join(tr.get(c, c) for c in clean_q(text))
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return ("k_" + s)[:40]


def load(data):
    return store.read_json(Path(data) / "knowledge.json", [])


def save(data, topics):
    store.write_json(Path(data) / "knowledge.json", topics)


def split_paras(question, paras):
    """Детерминированно делим формулировки: train / heldout (≥2 спрятанных)."""
    uniq = []
    for p in [question] + list(paras):
        c = clean_q(p)
        if c and c not in uniq:
            uniq.append(c)
    q0 = uniq[0]
    rest = uniq[1:]
    random.Random(zlib.crc32(q0.encode())).shuffle(rest)
    n_hold = max(1 if len(rest) < 6 else 2, round(len(rest) * HELDOUT_FRAC))
    return [q0] + rest[n_hold:], rest[:n_hold]


def add_or_merge(data, question, paras, answer_ru=None, answer_uz=None, topic=None, source="mentor"):
    """→ (topic_dict, created: bool). Ошибка, если формулировок мало."""
    paras = [p for p in paras if clean_q(p)]
    need = 3 if topic else MIN_PARA            # дополнить известную тему можно и меньшим числом
    if len({clean_q(p) for p in paras} - {clean_q(question)}) < need:
        raise ValueError(f"нужно минимум {need} разных формулировок вопроса (сейчас {len(paras)})")
    if not (answer_ru or answer_uz):
        raise ValueError("нужен ответ (ru или uz)")
    topics = load(data)
    train, held = split_paras(question, paras)
    t = next((x for x in topics if x["tag"] == topic), None) if topic else None
    created = t is None
    if created:
        tag = topic or slug(question)
        base, i = tag, 2
        while any(x["tag"] == tag for x in topics):
            tag = f"{base}_{i}"; i += 1
        t = {"tag": tag, "description": question, "responses": {}, "train": [], "heldout": [],
             "source": source, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        topics.append(t)
    seen = set(t["train"]) | set(t["heldout"])
    t["train"] += [p for p in train if p not in seen]
    t["heldout"] += [p for p in held if p not in seen and p not in t["train"]]
    for lang, ans in (("ru", answer_ru), ("uz", answer_uz)):
        if ans:
            ans = clean_answer(ans)
            lst = t["responses"].setdefault(lang, [])
            if ans not in lst:
                lst.append(ans)
    t["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save(data, topics)
    return t, created


def as_intents(topics):
    """Темы → формат data/intents.json (heldout НЕ включаем — они только для экзамена)."""
    out = []
    for t in topics:
        ru = t["responses"].get("ru") or t["responses"].get("uz")
        uz = t["responses"].get("uz") or t["responses"].get("ru")
        out.append({"tag": t["tag"], "description": t["description"], "patterns": t["train"],
                    "responses": {"ru": ru, "uz": uz}})
    return out
