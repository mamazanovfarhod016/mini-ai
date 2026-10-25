"""TinyGSM (код-решение на python) → ответ в формате GSM8K «<<a*b=c>> … #### N», чтобы reader_gsm8k.plan_of извлёк план как обычно.
Берём только «простые» решения: присваивания, + - * /, числа, имена; всё остальное (вызовы, **, if, циклы) → None."""
import ast


def tinygsm_to_answer(code):
    try: tree = ast.parse(code)
    except (SyntaxError, ValueError): return None
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)), None)
    if fn is None: return None
    env = {}; steps = []
    OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}

    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool): return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            v = ev(n.operand); return None if v is None else -v
        if isinstance(n, ast.Name): return env.get(n.id)
        if isinstance(n, ast.BinOp) and type(n.op) in OPS:
            l, r = ev(n.left), ev(n.right)
            if l is None or r is None: return None
            op = OPS[type(n.op)]
            if op == "/" and r == 0: return None
            v = {"+": l + r, "-": l - r, "*": l * r, "/": l / r if r else 0}[op]
            if abs(v) > 1e12: return None
            steps.append(f"<<{l:.10g}{op}{r:.10g}={v:.10g}>>"); return v
        return None

    for st in fn.body:
        if isinstance(st, ast.Expr) and isinstance(getattr(st, "value", None), ast.Constant): continue      # докстринг
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            v = ev(st.value)
            if v is None: return None
            env[st.targets[0].id] = v; continue
        if isinstance(st, ast.AugAssign) and isinstance(st.target, ast.Name) and type(st.op) in OPS:
            v = ev(ast.BinOp(left=ast.Name(id=st.target.id), op=st.op, right=st.value))
            if v is None: return None
            env[st.target.id] = v; continue
        if isinstance(st, ast.Return):
            v = ev(st.value) if st.value is not None else None
            if v is None or not steps: return None
            return " ".join(steps) + f"\n#### {v:.10g}"
        return None
    return None
