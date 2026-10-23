"""smoke test: Russian questions through the translation layer (dictionary only, before learned pairs)"""
import sys, shutil; sys.path.insert(0, ".")
from mini_ai.experimental import ExperimentalMind
shutil.rmtree("/tmp/ru_smoke", ignore_errors=True)
m = ExperimentalMind(school=False, wiki=False, state_dir="/tmp/ru_smoke")
m.kb.read_text("Paris is the capital of France. The Seine is a river that flows through Paris. Water boils at 100 degrees Celsius. "
               "Photosynthesis is the process by which plants make food from sunlight. The heart pumps blood through the body. "
               "Mercury is the planet closest to the Sun. Animals need oxygen to breathe. Oxygen is a gas.", "учебник")
print(m.russian.report())
for q in ("Какая столица Франции?", "Что такое фотосинтез?", "Какая планета ближе всего к Солнцу?", "Какой газ нужен животным для дыхания?", "При какой температуре кипит вода?", "Что качает кровь по телу?", "по-русски сердце значит heart", "Что качает кровь по телу?"):
    o = m.ask(q, internet=False)
    print(f"\nQ: {q}\n   → {o.get('note')}\n   how={o.get('how')} answer={o.get('answer')}\n   {o.get('text')}")
