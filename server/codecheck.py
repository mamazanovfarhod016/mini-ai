"""
Проверка кода, который написала модель: запуск скрытых тестов + оценка качества кода.

    check(task, code) -> {
        "runs": bool,            код выполнился без ошибок (compile+exec прошёл)
        "tests_passed": int,
        "tests_total": int,
        "quality": 0..100,
        "issues": [...],        список текстовых причин снижения оценки
        "passed": bool,         все скрытые тесты + quality >= PASS_QUALITY
    }

Безопасность — лучшее усилие, не полноценная песочница:
  • AST-проверка ДО запуска: запрещаем import os/sys/subprocess/socket/shutil и их детей,
    а также eval/exec/__import__/open/compile и `while True` (потенциально бесконечный цикл).
  • Код модели запускается в отдельном процессе python с timeout (без сети прямого контроля,
    но большинство путей к сети закрыты запретом на socket/urllib/requests).
"""
from __future__ import annotations

import ast
import json
import subprocess
import sys
import tempfile
from pathlib import Path

TIMEOUT_S = 2
PASS_QUALITY = 80

FORBIDDEN_MODULES = {
    "os", "sys", "subprocess", "socket", "shutil", "pathlib", "requests",
    "urllib", "ctypes", "multiprocessing", "threading", "importlib", "shelve",
}
FORBIDDEN_CALLS = {"eval", "exec", "__import__", "open", "compile", "input"}
SHORT_OK = {"i", "j", "k", "n", "x"}


# --------------------------------------------------------------------------------------
# 1. Безопасность: статическая проверка AST
# --------------------------------------------------------------------------------------
def safety_issues(tree: ast.AST) -> list[str]:
    """Список запрещённых конструкций в коде (import os, eval(), while True, ...)."""
    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_MODULES:
                    issues.append(f"запрещённый импорт: {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in FORBIDDEN_MODULES:
                issues.append(f"запрещённый импорт: {node.module}")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                issues.append(f"запрещённый вызов: {node.func.id}")
        elif isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value is True:
            issues.append("while True — бесконечный цикл")
    return issues


# --------------------------------------------------------------------------------------
# 2. Качество: собственные правила стиля (docs/PYTHON_STYLE.md)
# --------------------------------------------------------------------------------------
def _func_defs(tree: ast.AST):
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _complexity(fn: ast.AST) -> int:
    """Цикломатическая сложность = 1 + число точек принятия решений."""
    score = 1
    for node in ast.walk(fn):
        if isinstance(node, (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.With)):
            score += 1
        elif isinstance(node, ast.BoolOp):
            score += len(node.values) - 1
    return score


def style_issues(tree: ast.AST) -> list[str]:
    issues = []
    funcs = _func_defs(tree)
    if not funcs:
        return ["нет определения функции"]
    for fn in funcs:
        name = fn.name
        if len(name) < 2:
            issues.append(f"слишком короткое имя функции: {name}")
        body = fn.body
        has_doc = bool(body) and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str)
        if not has_doc:
            issues.append(f"{name}: нет однострочной докстроки")
        args = fn.args.args
        if any(a.annotation is None for a in args) or fn.returns is None:
            issues.append(f"{name}: не все аргументы/результат аннотированы типами")
        n_lines = (fn.end_lineno or fn.lineno) - fn.lineno
        if n_lines > 12:
            issues.append(f"{name}: функция длиннее 12 строк ({n_lines})")
        if _complexity(fn) > 5:
            issues.append(f"{name}: цикломатическая сложность > 5")
        for node in ast.walk(fn):
            if isinstance(node, ast.Compare):
                for op, comparator in zip(node.ops, node.comparators):
                    if isinstance(op, ast.Eq) and isinstance(comparator, ast.Constant) \
                            and isinstance(comparator.value, bool):
                        issues.append(f"{name}: `== {comparator.value}` вместо прямого условия")
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append(f"{name}: голый except")
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range":
                a0 = node.args[0] if node.args else None
                if isinstance(a0, ast.Call) and isinstance(a0.func, ast.Name) and a0.func.id == "len":
                    issues.append(f"{name}: `range(len(...))` вместо `for item in ...`")
            if isinstance(node, (ast.Assign, ast.comprehension)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for t in targets:
                    if isinstance(t, ast.Name) and len(t.id) < 2 and t.id not in SHORT_OK:
                        issues.append(f"{name}: слишком короткое имя переменной `{t.id}`")
    return issues


def ruff_issues(code: str) -> list[str]:
    """ruff check --select E,F,W,N,SIM,B,UP,RET,PL --output-format json -"""
    ruff = Path(__file__).resolve().parent.parent / ".venv" / "bin" / "ruff"
    cmd = [str(ruff) if ruff.exists() else "ruff", "check",
           "--select", "E,F,W,N,SIM,B,UP,RET,PL", "--output-format", "json", "-"]
    try:
        res = subprocess.run(cmd, input=code, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return []
    try:
        violations = json.loads(res.stdout or "[]")
    except json.JSONDecodeError:
        return []
    return [f"ruff {v['code']}: {v['message']}" for v in violations]


def quality_score(code: str, tree: ast.AST) -> tuple[int, list[str]]:
    issues = style_issues(tree) + ruff_issues(code)
    score = 100
    score -= 15 * sum("докстрок" in i for i in issues)
    score -= 15 * sum("аннотированы" in i for i in issues)
    score -= 20 * sum("range(len" in i for i in issues)
    score -= 15 * sum("== True" in i or "== False" in i for i in issues)
    score -= 15 * sum("голый except" in i for i in issues)
    score -= 10 * sum("длиннее 12" in i for i in issues)
    score -= 10 * sum("сложность > 5" in i for i in issues)
    score -= 5 * sum("короткое имя" in i for i in issues)
    score -= 5 * min(sum(i.startswith("ruff") for i in issues), 8)
    return max(0, score), issues


# --------------------------------------------------------------------------------------
# 3. Исполнение: прогон скрытых тестов в отдельном процессе
# --------------------------------------------------------------------------------------
_RUNNER_TPL = """
import json
ns = {{}}
try:
    exec(compile(CODE, "<task>", "exec"), ns)
    fn = ns[{func!r}]
except Exception as exc:
    print(json.dumps({{"error": str(exc)}}))
else:
    results = []
    for t in {tests!r}:
        try:
            got = fn(*t["args"])
            results.append(got == t["expected"])
        except Exception:
            results.append(False)
    print(json.dumps({{"results": results}}))
"""


def run_hidden_tests(code: str, func_name: str, tests: list[dict]) -> tuple[bool, int, int]:
    """Возвращает (runs_ok, пройдено, всего) — запуск в отдельном процессе с таймаутом."""
    script = "CODE = " + repr(code) + "\n" + _RUNNER_TPL.format(func=func_name, tests=tests)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as fh:
        fh.write(script)
        path = fh.name
    try:
        res = subprocess.run([sys.executable, "-I", path], capture_output=True, text=True, timeout=TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return False, 0, len(tests)
    finally:
        Path(path).unlink(missing_ok=True)
    try:
        out = json.loads(res.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return False, 0, len(tests)
    if "error" in out:
        return False, 0, len(tests)
    passed = sum(out["results"])
    return True, passed, len(tests)


# --------------------------------------------------------------------------------------
# 4. Главная функция
# --------------------------------------------------------------------------------------
def check(task: dict, code: str) -> dict:
    """task: {func_name, hidden_tests, tests?}. code: текст функции от модели."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return {"runs": False, "tests_passed": 0, "tests_total": len(task.get("hidden_tests", [])),
                "quality": 0, "issues": [f"SyntaxError: {exc}"], "passed": False}

    danger = safety_issues(tree)
    if danger:
        total = len(task.get("hidden_tests", []))
        return {"runs": False, "tests_passed": 0, "tests_total": total,
                "quality": 0, "issues": danger, "passed": False}

    tests = task.get("hidden_tests") or task.get("tests") or []
    runs, passed, total = run_hidden_tests(code, task["func_name"], tests)
    quality, issues = quality_score(code, tree)
    all_pass = runs and total > 0 and passed == total
    return {
        "runs": runs,
        "tests_passed": passed,
        "tests_total": total,
        "quality": quality,
        "issues": issues,
        "passed": bool(all_pass and quality >= PASS_QUALITY),
    }
