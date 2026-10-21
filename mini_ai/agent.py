"""Mini AI — АГЕНТ (21.10.2026). Способности как у Claude Code, но без нейросети и без градиента.

Устройство:
  Память знаний   — memory_v2.Memory (сито + рабочая + долговременная, не забывает).
  Память приёмов  — список процедур: вектор запроса-примера → шаги [(инструмент, шаблон аргумента)].
                    Новый запрос → ближайшая процедура по косинусу (сито) → подстановка слотов → выполнение.
  Инструменты     — shell, read, write, wiki (Википедия → память), recall (ответ из памяти), compose.
  Саморазвитие    — если процедура не найдена или проверка провалена: ПОИСК по генофонду шаблонов
                    (одиночные и пары шагов), каждый кандидат выполняется и проверяется; прошедший
                    становится новой процедурой (teach). Без градиента: учится от проверки, как ребёнок
                    от результата. Всё сохраняется на диск → следующий запуск умеет больше.
"""
import os, re, json, subprocess, time
import numpy as np
from mini_ai.memory_v2 import Memory, toks, content_words, C, unit, D


# ---------- слоты: что в запросе похоже на путь, текст, число, выражение, тему ----------
def slots(request):
    s = {}
    paths = re.findall(r"(?:~|/)?[\w.\-]+(?:/[\w.\-]+)+|[\w\-]+\.(?:txt|py|md|json|csv|log)", request)
    if paths: s["path"] = os.path.expanduser(paths[0])
    if len(paths) > 1: s["path2"] = os.path.expanduser(paths[1])
    q = re.findall(r"[«\"']([^«»\"']+)[»\"']", request)
    if q: s["text"] = q[0]
    expr = re.findall(r"\d+(?:\s*[-+*/%]\s*\d+)+", request)
    if expr: s["expr"] = expr[0].replace(" ", "")
    nums = re.findall(r"\b\d+\b", request)
    if nums: s["n"] = nums[0]
    m = re.search(r"(?:wikipedia|википеди\w*)\s+(?:about\s+|про\s+|о\s+)?([A-ZА-Я][\w\-]+(?:\s[A-ZА-Я][\w\-]+)*)", request)
    if m: s["topic"] = m.group(1)
    caps = re.findall(r"\b[A-Z][a-z]+\b", request)
    if caps and "topic" not in s: s["topic"] = caps[-1]
    s["request"] = request
    return s


def fill(template, s):
    try: return template.format(**s), True
    except KeyError: return template, False


# ---------- инструменты ----------
class Tools:
    def __init__(self, agent): self.a = agent

    def shell(self, cmd):
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr).strip()

    def read(self, path):
        with open(path) as f: return f.read()

    def write(self, arg):                       # arg = "path|||text"
        path, text = arg.split("|||", 1)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f: f.write(text)
        return f"записано {len(text)} симв. в {path}"

    def wiki(self, topic):
        from tools.exp24_wiki_crawler import fetch_article, clean_extract
        title, text, _ = fetch_article(topic)
        n = self.a.mem.read_text(clean_extract(text), source=title)
        return f"прочитано {n} предложений из «{title}»"

    def recall(self, question):
        top = self.a.mem.ask(question, topk=3)
        return top[0][0] if top else ""

    def learn_text(self, text):                 # текст → память (из файла или вывода)
        return f"запомнено {self.a.mem.read_text(text, source='user')} предл."


# генофонд: из чего агент собирает новые приёмы (ОДИН раз описан человеком, дальше — только поиск)
# третье поле: True = шаг МЕНЯЕТ мир (файлы). Такие шаги разрешены только если запрос про изменение.
GENE_POOL = [
    ("write", "{path}|||{text}", True),
    ("read", "{path}", False),
    ("shell", "wc -l < {path}", False),
    ("shell", "wc -w < {path}", False),
    ("shell", "ls {path}", False),
    ("shell", "python3 -c \"print({expr})\"", False),
    ("shell", "grep -c '{text}' {path}", False),
    ("shell", "cp {path} {path2}", True),
    ("shell", "head -n {n} {path}", False),
    ("shell", "echo '{text}' >> {path}", True),
    ("wiki", "{topic}", False),
    ("recall", "{request}", False),
    ("learn_text", "{text}", False),
]
MUTATE_WORDS = {"create", "make", "write", "save", "copy", "append", "add", "put", "создай", "запиши", "скопируй", "добавь"}


class Agent:
    def __init__(self, path):
        self.path = path; os.makedirs(path, exist_ok=True)
        self.mem = Memory(path=os.path.join(path, "knowledge"))
        self.tools = Tools(self); self.procs = []; self.log = []
        self.genes = list(GENE_POOL)                       # генофонд растёт (learn_genes_from_text)
        f = os.path.join(path, "procs.json")
        if os.path.exists(f):
            j = json.load(open(f))
            for p in j.get("procs", []): self.procs.append(dict(p, vec=self.req_vec(p["request"])))
            for g in j.get("genes", []):
                g = tuple(g)
                if g not in self.genes: self.genes.append(g)

    def save(self):
        self.mem.save()
        json.dump(dict(procs=[{k: v for k, v in p.items() if k != "vec"} for p in self.procs],
                       genes=[list(g) for g in self.genes if g not in GENE_POOL]),
                  open(os.path.join(self.path, "procs.json"), "w"), ensure_ascii=False, indent=1)

    # ---- вектор запроса: мешок слов, где слоты заменены на их ТИП (путь→pathslot, число→nslot...) ----
    # так «how many lines in /a/b.txt» и «number of lines in /c/d.txt» становятся близки, а пути не мешают
    def req_vec(self, request):
        s = slots(request); r = request
        for k in ("path2", "path", "text", "expr", "topic"):
            if k in s and s[k]:
                for form in (s[k], s[k].replace(os.path.expanduser("~"), "~")):
                    r = r.replace(form, f" {k}slot ")
        ws = [("nslot" if w.isdigit() else w) for w in toks(r)]
        cw = [w for w in ws if w not in ("the", "a", "an", "of", "to", "in", "is", "and", "for", "me", "please")]
        return unit(sum(C(w) for w in cw)) if cw else np.zeros(D)

    def match(self, request, k=3):
        if not self.procs: return []
        v = self.req_vec(request); M = np.stack([p["vec"] for p in self.procs]); s = M @ v
        order = np.argsort(-s)[:k]
        return [(self.procs[i], float(s[i])) for i in order]

    def execute(self, steps, s):
        out = ""
        for tool, tpl in steps:
            arg, ok = fill(tpl, s)
            if not ok: return None, f"нет слота для {tpl}"
            try: out = getattr(self.tools, tool)(arg)
            except Exception as e: return None, f"{tool}({arg}): {e}"
        return out, "ok"

    def teach(self, request, steps):
        self.procs.append(dict(request=request, steps=steps, vec=self.req_vec(request), wins=0, fails=0))

    # ---- САМОРАЗВИТИЕ 2-го ПОРЯДКА: новые ГЕНЫ из прочитанного текста (документация, man, примеры) ----
    # Из текста вынимаются строки-команды (начинаются с известной утилиты); конкретные пути/числа/слова
    # в них заменяются слотами {path}/{n}/{text} → получается шаблон-ген. Ни одного правила про смысл
    # команды: что она делает — выяснит проверка, когда ген попадёт в перебор.
    KNOWN_CMDS = ("wc", "ls", "grep", "head", "tail", "cat", "sort", "uniq", "cut", "tr", "du", "find",
                  "awk", "sed", "python3", "cp", "mv", "echo", "date", "stat", "md5sum", "sha256sum", "rev", "tac", "nl")

    def learn_genes_from_text(self, text):
        new = 0
        for line in text.splitlines():
            line = line.strip().lstrip("$ ").split("#")[0].strip()      # без комментария
            if not line or line.split()[0] not in self.KNOWN_CMDS or len(line) > 80: continue
            tpl = re.sub(r"(?:~|/)?[\w.\-]+(?:/[\w.\-]+)+|\b[\w\-]+\.(?:txt|py|md|json|csv|log)\b", "{path}", line)
            tpl = re.sub(r"'[^']*'", "'{text}'", tpl)
            tpl = re.sub(r"(?<![\w{])\d+(?![\w}])", "{n}", tpl)
            gene = ("shell", tpl, bool({"cp", "mv", ">", ">>", "rm"} & set(tpl.split())))
            if gene not in self.genes: self.genes.append(gene); new += 1
        return new

    # ---- главный вход: запрос → ответ. check(out)->bool — проверка (тест/пользователь), если есть ----
    def run(self, request, check=None, thr=0.55, depth=2, learn=True):
        s = slots(request); out = None
        for proc, score in self.match(request):
            if score < thr: break
            out, st = self.execute(proc["steps"], s)
            if out is not None and (check is None or check(out)):
                proc["wins"] += 1; self.log.append(("proc", request, score)); return out, "память приёмов"
            proc["fails"] += 1
            if check is None: return out, "приём без проверки"
        if check is None or not learn: return out, "нет приёма"
        # САМОРАЗВИТИЕ: поиск по генофонду — одиночные шаги, затем пары.
        # Шаги, меняющие мир, разрешены только если запрос сам про изменение (иначе перебор ломает данные —
        # урок exp25: «echo >> file; grep -c» прошёл проверку, испортив файл).
        mutate_ok = bool(MUTATE_WORDS & set(toks(request)))
        pool = [(t, tpl) for t, tpl, m in self.genes if mutate_ok or not m]
        # сеть (wiki) — только если запрос о ней: иначе слепой перебор пар уходит в интернет на минуты
        if not ({"wikipedia", "wiki", "википедия"} & set(toks(request))): pool = [g for g in pool if g[0] != "wiki"]
        cands = [[g] for g in pool]
        if depth >= 2: cands += [[g1, g2] for g1 in pool for g2 in pool if g1 != g2]
        tried = 0
        for steps in cands:
            if not all(fill(t, s)[1] for _, t in steps): continue
            tried += 1
            out, st = self.execute(steps, s)
            if out is not None and check(out):
                self.teach(request, steps); self.log.append(("learned", request, steps, tried))
                return out, f"НАУЧИЛСЯ (перебрал {tried}): {steps}"
        return None, f"не нашёл (перебрал {tried})"
