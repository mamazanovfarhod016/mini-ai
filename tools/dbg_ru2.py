import sys, shutil; sys.path.insert(0, ".")
from mini_ai.experimental import ExperimentalMind
from mini_ai.question import parse_question
shutil.rmtree("/tmp/ru_smoke2", ignore_errors=True)
m = ExperimentalMind(school=False, wiki=False, state_dir="/tmp/ru_smoke2")
m.kb.read_text("Paris is the capital of France. The Seine is a river that flows through Paris. Water boils at 100 degrees Celsius. "
               "Photosynthesis is the process by which plants make food from sunlight. The heart pumps blood through the body. "
               "Mercury is the planet closest to the Sun. Animals need oxygen to breathe. Oxygen is a gas.", "учебник")
q = "at what temperature boils water?"; fr = parse_question(q); m._cur_q = q
D = m.discourse
qw = {w for w in (set(fr.get("X") or []) | set(fr.get("Y") or [])) if w not in m.META}
print("fr:", fr); print("qw:", qw)
for bi, cov in D.bridges(qw, k=4): print(f"  bridge #{bi} cov={cov:.2f}: {m.kb.sents[bi][0]}")
r = D.reason(q, fr); print("RESULT:", {k: v for k, v in (r or {}).items() if k in ("answer", "confidence", "why", "link", "cov", "bridge")})
o = m.ask(q, internet=False); print("ASK:", o.get("how"), o.get("answer"), o.get("arbiter"))
