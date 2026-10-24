"""Проверка пачки задач GSM8K-формата: python3 tools/check_gsm_part.py <file.jsonl>
Печатает плохие строки (JSON, нет шагов, арифметика, финал != последний шаг, дубль вопроса) и итог total/bad."""
import json, re, sys
path = sys.argv[1]; bad = 0; n = 0; seen = set()
for line in open(path):
    line = line.strip()
    if not line: continue
    n += 1
    try: d = json.loads(line)
    except Exception: print("JSON", line[:60]); bad += 1; continue
    q, a = d.get("question", ""), d.get("answer", "")
    if q in seen: print("DUP", q[:60]); bad += 1
    seen.add(q)
    steps = re.findall(r"<<([\d.]+)([-+*/])([\d.]+)=([\d.]+)>>", a.replace(",", ""))
    if not steps or "####" not in a: print("NOSTEPS", q[:60]); bad += 1; continue
    for x, op, y, z in steps:
        x, y, z = float(x), float(y), float(z)
        v = {"+": x + y, "-": x - y, "*": x * y, "/": x / y if y else None}[op]
        if v is None or abs(v - z) > 1e-6: print("MATH", x, op, y, "=", z, "|", q[:50]); bad += 1
    try: fin = float(a.split("####")[-1].strip().replace(",", ""))
    except Exception: print("FINAL", a[-30:]); bad += 1; continue
    if abs(fin - float(steps[-1][3])) > 1e-6: print("FINAL!=LAST", fin, steps[-1][3], "|", q[:50]); bad += 1
print("total", n, "bad", bad)
