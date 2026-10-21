"""Создание кода (exp43): без нейросети. Три механизма, все уже проверены на других слоях:
  1. ПАМЯТЬ ОБРАЗЦОВ: функции из прочитанного кода (def … → тело). Ключ = слова docstring/имени + сигнатура.
     Как learn_genes_from_text для shell (exp28), но единица — функция.
  2. СБОРКА: задача → ближайшие образцы (сито, косинус) → подстановка имён/констант из задачи (слоты) →
     композиция (f(g(x))) если один образец не покрывает слова задачи.
  3. ПРОВЕРКА ЗАПУСКОМ: тесты (вход → ожидаемый выход) выполняются в подпроцессе; прошедший вариант
     запоминается как новый образец с ключом задачи (exp25 teach). Не прошёл → следующий кандидат.
  4. ИСПРАВЛЕНИЕ ПО ОШИБКЕ (минимальное рассуждение): traceback → тип ошибки → правка из таблицы
     (NameError: имя → импорт/параметр; TypeError str+int → str(); IndexError → проверка длины).
     Таблица выучивается из пар (ошибка, правка), которые прошли проверку.
"""
import re, ast, os, json, subprocess, tempfile, textwrap, itertools
import numpy as np
from mini_ai.memory_v2 import toks, content_words, C, unit, D


def func_vec(name, doc, body):
    ws = content_words(re.findall(r"[a-z]+", (name + " " + doc).lower().replace("_", " ")))
    ws += [w for w in re.findall(r"[a-z_]+", body) if w in ("sorted", "reversed", "sum", "len", "max", "min", "split", "join", "replace", "count", "upper", "lower", "set", "dict", "range")]
    return unit(sum(C(w) for w in ws)) if ws else np.zeros(D)


class CodeMemory:
    """Образцы функций. Учится из любого Python-текста."""
    def __init__(self, path=None):
        self.items = []; self.path = path
        if path and os.path.exists(path):
            for it in json.load(open(path)): self.items.append(dict(it, vec=func_vec(it["name"], it["doc"], it["src"])))

    def save(self):
        if self.path: json.dump([{k: v for k, v in it.items() if k != "vec"} for it in self.items], open(self.path, "w"), ensure_ascii=False, indent=1)

    def learn_from_code(self, code, source="read"):
        n = 0
        try: tree = ast.parse(code)
        except SyntaxError: return 0
        lines = code.splitlines()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                src = "\n".join(lines[node.lineno - 1:node.end_lineno]); doc = ast.get_docstring(node) or ""
                args = [a.arg for a in node.args.args]
                if any(it["src"] == src for it in self.items): continue
                self.items.append(dict(name=node.name, doc=doc, args=args, src=textwrap.dedent(src), source=source,
                                       vec=func_vec(node.name, doc, src), wins=0)); n += 1
        return n

    def nearest(self, task, k=5):
        if not self.items: return []
        v = unit(sum(C(w) for w in content_words(toks(task))) if content_words(toks(task)) else np.zeros(D))
        M = np.stack([it["vec"] for it in self.items]); s = M @ v
        return [(self.items[i], float(s[i])) for i in np.argsort(-s)[:k]]


def run_tests(code, fname, tests, timeout=5):
    """tests: [(args_tuple, expected)] → (passed, total, last_error)"""
    harness = code + "\n\nimport json, sys\n_T = json.loads(sys.argv[1])\n_out = []\nfor a, e in _T:\n    try: r = " + fname + "(*a); _out.append([r == e, repr(r)])\n    except Exception as ex: _out.append([False, type(ex).__name__ + ': ' + str(ex)])\nprint(json.dumps(_out))\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f: f.write(harness); p = f.name
    try:
        r = subprocess.run(["python3", p, json.dumps(tests)], capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0: return 0, len(tests), r.stderr.strip().splitlines()[-1] if r.stderr.strip() else "error"
        res = json.loads(r.stdout.strip().splitlines()[-1])
        return sum(ok for ok, _ in res), len(tests), next((m for ok, m in res if not ok), "")
    except subprocess.TimeoutExpired: return 0, len(tests), "Timeout"
    finally: os.unlink(p)


# таблица исправлений по ошибке: (regex по сообщению) → функция-правка кода. Начальная — 4 правила; растёт.
FIXES = [
    (r"NameError: name '(\w+)' is not defined", lambda m, c: f"import {m.group(1)}\n" + c if m.group(1) in ("re", "math", "json", "itertools", "collections", "string") else c),
    (r"TypeError: .*str.*int|can only concatenate str", lambda m, c: re.sub(r"\+\s*(\w+)\s*$", r"+ str(\1)", c, flags=re.M)),
    (r"TypeError: '(\w+)' object is not subscriptable", lambda m, c: c.replace("[0]", "")),
    (r"ZeroDivisionError", lambda m, c: re.sub(r"/\s*len\((\w+)\)", r"/ max(1, len(\1))", c)),
]


class Coder:
    def __init__(self, memory: CodeMemory): self.mem = memory; self.log = []

    def rename(self, src, old, new):
        return re.sub(rf"\b{re.escape(old)}\b", new, src)

    def candidates(self, task, fname, nargs):
        """варианты кода: 1) образец с переименованием; 2) композиция двух образцов f(g(x))"""
        near = self.mem.nearest(task, k=6); out = []
        for it, s in near:
            if len(it["args"]) == nargs: out.append((s, self.rename(it["src"], it["name"], fname), [it["name"]]))
        for (a, sa), (b, sb) in itertools.permutations(near, 2):
            if len(a["args"]) == 1 and len(b["args"]) == 1 and nargs == 1:
                code = a["src"] + "\n\n" + b["src"] + f"\n\ndef {fname}(x):\n    return {b['name']}({a['name']}(x))\n"
                out.append(((sa + sb) / 2 - 0.1, code, [a["name"], b["name"]]))
        out.sort(key=lambda t: -t[0]); return out

    def solve(self, task, fname, tests, max_tries=12, max_fix=2):
        nargs = len(tests[0][0]); tried = 0
        for s, code, parts in self.candidates(task, fname, nargs)[:max_tries]:
            tried += 1; cur = code
            for fix_round in range(max_fix + 1):
                ok, n, err = run_tests(cur, fname, tests)
                if ok == n:
                    self.mem.items.append(dict(name=fname, doc=task, args=[f"a{i}" for i in range(nargs)], src=cur, source="solved",
                                               vec=func_vec(fname, task, cur), wins=1))
                    return dict(code=cur, parts=parts, tried=tried, fixes=fix_round, status="ok")
                fixed = cur
                for rx, fn in FIXES:
                    m = re.search(rx, err)
                    if m: fixed = fn(m, cur); break
                if fixed == cur: break
                cur = fixed
        return dict(code=None, tried=tried, status="fail")
