"""Проверка качества УЗБЕКСКИХ диалогов для Mini AI (без ИИ, детерминированно). Основа — dialogs_ru_v2/check_dialogs.py.
    python check_dialogs_uz.py part_XX.jsonl [ещё...]     # отчёт; код выхода 1, если файл не прошёл ЖЁСТКИЕ правила
    python check_dialogs_uz.py --merge out.jsonl part_*.jsonl
Формат строки: {"id","lang":"uz","topic","memory":bool,"turns":[{"role":"user|bot","text"}...]}
Диалог отбраковывается, если:
  - не 4..10 реплик, нет чередования user/bot (начинает user, заканчивает bot);
  - реплика бота не 2..240 символов; есть кириллица (узбекский — ТОЛЬКО латиница); эмодзи; «sun'iy intellekt sifatida»;
  - мало узбекского: в репликах бота < 30% слов из списка частых узбекских слов (защита от русского/английского текста);
  - апостроф не ASCII (o‘ g‘ ʻ ’ → нужно o' g');
  - бот повторяет свою реплику или начинает 2 ответа одинаково (первые 3 слова);
  - memory=true, но в последнем ответе бота нет основы слова (5 букв) из ранних реплик пользователя,
    которой нет в последней реплике пользователя.
По набору: ответ бота, встречающийся > 2 раз (> 6, если ≤ 12 символов) → шаблон, диалог отбракован; дубли диалогов.
ЖЁСТКИЕ правила файла (иначе выход 1): годных ≥ 55 из 60, уникальных ответов бота ≥ 90%, с памятью ≥ 35%,
каждой длины (4, 6, 8, 10 реплик) ≥ 15% годных.
"""
import json
import re
import sys
from collections import Counter

EMOJI = re.compile("[\U0001F300-\U0001FAFF\u2600-\u27BF]")
CYR = re.compile("[\u0400-\u04FF]")
BADAPOS = re.compile("[\u2018\u2019\u02BB\u02BC`]")
BANNED = re.compile(r"(sun'iy intellekt sifatida|til modeli sifatida|as an ai|как ии)", re.I)
WORD = re.compile(r"[a-z']{4,}", re.I)
UZ = set("""men sen u biz siz ular bu shu u mening sening uning bizning sizning nima nega qanday qachon qayer qayerda
qaerda kim qancha necha ha yo'q yoq va ham lekin ammo yoki bilan uchun keyin oldin hozir bugun ertaga kecha juda
yaxshi yomon katta kichik bor emas edi ekan bo'ladi bo'lsa bo'lib qil qiling qilaman qildim qildi qilish deb dedi
aytdi ayt ayting kerak mumkin rahmat salom xayr qalay qalaysan qalaysiz men-chi o'zim o'z sizga senga menga unga
bizga ularga sizni seni meni uni nimani kimni qilyapman qilyapsan qilyapsiz bilaman bilmayman bilasanmi bilasizmi
ismim isming ismingiz yosh yoshda yoshdaman maktab sinf dars uy oila ona ota aka uka opa singil do'st do'stim
o'qiyman o'qish yozish gapir gaplash yordam albatta balki hammasi hamma har bir ikki uch to'rt besh olti yetti
sakkiz to'qqiz o'n yigirma yuz ming ko'p oz endi yana faqat hali allaqachon chunki shuning agar unda bunda
sevaman yoqadi yoqtiraman xohlayman istayman ketdim keldim boraman kelaman ko'raman ko'rdim eshitdim tushundim
tushunmadim to'g'ri noto'g'ri javob savol misol hisob kun yil oy hafta soat vaqt""".split())


def norm(t):
    return re.sub(r"[^\w']+", " ", t.lower()).strip()


SUF = ("man", "san", "miz", "siz", "di", "mi", "lar", "ni", "ga", "da", "dan", "ning", "im", "ing", "gan", "yap",
       "moq", "chi", "dek", "roq", "lik", "sin", "ay", "ib", "sa")
EN = set("the a an is are was were you your i my me it to of and in on for with that this what how do does did not be "
         "have has can will hello hi thanks yes no ok okay great good fine".split())


def uz_share(text):
    """Доля «узбекских» слов: из списка частых, ИЛИ с q/x/o'/g'/sh/ch, ИЛИ с узбекским окончанием.
    Английские служебные слова считаются против."""
    ws = re.findall(r"[a-z']+", text.lower())
    if not ws:
        return 0.0
    en = sum(1 for w in ws if w in EN) / len(ws)
    uz = sum(1 for w in ws if w in UZ or re.search(r"[qx]|o'|g'|sh|ch", w) or w.endswith(SUF)) / len(ws)
    return 0.0 if en > 0.25 else uz


def check_one(d):
    t = d.get("turns") or []
    if not 4 <= len(t) <= 10 or len(t) % 2:
        return "len"
    for i, x in enumerate(t):
        if x.get("role") != ("user" if i % 2 == 0 else "bot"):
            return "order"
        s = str(x.get("text") or "").strip()
        if not s:
            return "empty"
        if EMOJI.search(s):
            return "emoji"
        if CYR.search(s):
            return "cyrillic"
        if BADAPOS.search(s):
            return "apostrophe"
        if x["role"] == "bot":
            if not 2 <= len(s) <= 240:
                return "bot_len"
            if BANNED.search(s):
                return "banned"
    bots_raw = " ".join(x["text"] for x in t if x["role"] == "bot")
    if uz_share(bots_raw) < 0.30:
        return "not_uz"
    bots = [norm(x["text"]) for x in t if x["role"] == "bot"]
    if len(set(bots)) < len(bots):
        return "repeat_in_dialog"
    starts = [" ".join(b.split()[:3]) for b in bots if len(b.split()) >= 4]
    if len(set(starts)) < len(starts):
        return "same_start"
    if d.get("memory"):
        users = [norm(x["text"]) for x in t if x["role"] == "user"]
        if len(users) < 3:
            return "memory_short"
        stems = lambda ws: {w[:5] for w in ws}  # noqa: E731
        early = stems(WORD.findall(" ".join(users[:-1])))
        last_u = stems(WORD.findall(users[-1]))
        last_b = stems(WORD.findall(bots[-1]))
        if not (last_b & (early - last_u)):
            return "memory_not_used"
    return None


def load(paths):
    out = []
    for p in paths:
        for n, line in enumerate(open(p, encoding="utf-8"), 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append((p, json.loads(line)))
            except json.JSONDecodeError:
                out.append((p, {"_bad": n}))
    return out


def evaluate(items):
    reasons, good = Counter(), []
    for p, d in items:
        r = "json" if "_bad" in d else check_one(d)
        if r:
            reasons[r] += 1
        else:
            good.append((p, d))
    cnt = Counter(norm(x["text"]) for _, d in good for x in d["turns"] if x["role"] == "bot")
    final, seen = [], set()
    for p, d in good:
        key = norm(" | ".join(x["text"] for x in d["turns"]))
        if key in seen:
            reasons["dup_dialog"] += 1
            continue
        if [b for b in (norm(x["text"]) for x in d["turns"] if x["role"] == "bot") if cnt[b] > (6 if len(b) <= 12 else 2)]:
            reasons["template"] += 1
            continue
        seen.add(key)
        final.append((p, d))
    bots = [norm(x["text"]) for _, d in final for x in d["turns"] if x["role"] == "bot"]
    return final, reasons, len(set(bots)) / max(1, len(bots))


def main():
    a = sys.argv[1:]
    merge = None
    if a and a[0] == "--merge":
        merge, a = a[1], a[2:]
    items = load(a)
    final, reasons, uniq = evaluate(items)
    mem = sum(1 for _, d in final if d.get("memory"))
    lens = Counter(len(d["turns"]) for _, d in final)
    print(f"всего {len(items)} → годных {len(final)}; уникальных ответов бота {uniq:.0%}; "
          f"с памятью {mem} ({mem / max(1, len(final)):.0%}); длины {dict(sorted(lens.items()))}")
    print("отбраковано:", dict(reasons))
    fail = []
    if not merge:
        n = max(1, len(final))
        if len(items) >= 60 and len(final) < 55:
            fail.append(f"годных {len(final)} < 55")
        if uniq < 0.90:
            fail.append(f"уникальных {uniq:.0%} < 90%")
        if mem / n < 0.35:
            fail.append(f"с памятью {mem / n:.0%} < 35%")
        for L in (4, 6, 8, 10):
            if lens.get(L, 0) / n < 0.15:
                fail.append(f"длины {L} только {lens.get(L, 0)} (<15%)")
        print("ЖЁСТКИЕ правила:", "OK" if not fail else "НЕ ПРОШЁЛ — " + "; ".join(fail))
    if merge:
        with open(merge, "w", encoding="utf-8") as f:
            for i, (_, d) in enumerate(final, 1):
                f.write(json.dumps({**d, "id": f"uz-{i:04d}", "lang": "uz"}, ensure_ascii=False) + "\n")
        print("записано:", merge)
    sys.exit(1 if fail else 0)


if __name__ == "__main__":
    main()
