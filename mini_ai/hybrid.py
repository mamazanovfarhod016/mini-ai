"""Hybrid — новая модель Mini AI: ядро понимания (без градиента) + говорилка (трансформер).

Разделение труда:
  • ЯДРО (ExperimentalMind: сито, память фактов, рамки, вывод типа) — ПОНИМАЕТ смысл реплики,
    находит/выводит знание, учится мгновенно («выучи …» / любое утверждение → факт сразу в памяти).
  • ГОВОРИЛКА (трансформер из app/www/model_transformer.js, обучен градиентом) — только речь:
    болтовня (привет/как дела/кто ты) и озвучивание структуры «смысл», которую даёт ядро.
Маршрутизация — по СМЫСЛУ реплики (ядро решает, что это: утверждение-знание, вопрос о мире, болтовня):
  1) утверждение с фактом (reader вытащил (X, rel, Y)) → learn → подтверждение + что поняла;
  2) вопрос о мире (рамка с сущностью) → ядро ask → предложение из факта (+ переспрос, если не знает);
  3) остальное → трансформер (говорит сам).
Ответ всегда содержит trace: кто ответил и почему — для честности в интерфейсе.
"""
import json, os, re, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RU = re.compile(r"[а-яё]", re.I)

class Hybrid:
    def __init__(self, mind=None, speaker_path=None, state_dir=None):
        from mini_ai.experimental import ExperimentalMind
        self.mind = mind or ExperimentalMind(school=True, wiki=False, state_dir=state_dir)
        self.kv = self.tok = None; self.cfg = {}; self.max_q = 40
        path = speaker_path or (ROOT / "app/www/model_transformer.js")
        if os.path.exists(path):
            from mini_ai.transformer import load_model, KVModel
            s = open(path, encoding="utf-8").read(); j = json.loads(s.split("=", 1)[1].rstrip(";\n"))
            p, self.cfg, self.tok, meta = load_model(j); self.kv = KVModel(p, self.cfg); self.max_q = j.get("max_q", 40)
        self.sessions = {}

    # ---------- говорилка ----------
    def speak(self, text):
        if not self.kv: return None
        from mini_ai.transformer import BOS, SEP, clean_question
        ids = [BOS] + self.tok.encode(clean_question(text))[:self.max_q] + [SEP]
        out, _ = self.kv.generate(ids, max_new=self.cfg["block"] - len(ids), temperature=0.0)
        return self.tok.decode(out).strip()

    # ---------- понимание ----------
    def classify(self, text):
        """смысл реплики: 'fact' (есть что выучить), 'question' (о мире), 'chat' (болтовня/о себе)"""
        t = text.strip(); tl = t.lower()
        if re.match(r"^(выучи|запомни|learn|remember)\b", tl): return "fact"
        if RU.search(t): return "chat"                                   # ядро понимает пока только английский
        is_q = t.endswith("?") or bool(re.match(r"^(what|who|where|when|which|how|why|is|are|do|does|did|can)\b", tl))
        if is_q:
            if re.search(r"\byou\b|\byour\b", tl): return "chat"       # вопросы о ней самой — говорилка
            return "question"
        facts, _ = self.mind.kb.reader.read(t, "user")
        if facts and len(t.split()) >= 3: return "fact"
        return "chat"

    def reply(self, text, sid="default"):
        t0 = time.time(); kind = self.classify(text); out = dict(kind=kind, engine=None, text=None, facts=[], learned=[])
        if kind == "fact":
            body = re.sub(r"^(выучи|запомни|learn|remember)[:\s]+", "", text.strip(), flags=re.I)
            n = self.mind.learn(body); facts, _ = self.mind.kb.reader.read(body, "user")
            out["learned"] = [f"{X} —{rel}→ {Y}" for X, rel, Y in facts]; out["engine"] = "core"
            if facts:
                X, rel, Y = facts[0]
                out["text"] = f"Got it — I'll remember that {X} {('is' if rel == 'be' else rel.replace('_', ' '))} {Y}." + (f" And {len(facts) - 1} more thing{'s' if len(facts) > 2 else ''}." if len(facts) > 1 else "")
            else:
                out["text"] = "I read that, but couldn't pull a clear fact out of it. Try one short sentence like «X is a Y»."
        elif kind == "question":
            r = self.mind.ask(text, internet=True); out["engine"] = "core"; out["facts"] = r.get("facts", []); out["how"] = r.get("how")
            out["text"] = r.get("text") or r.get("answer") or "I don't know that yet — tell me, and I'll remember."
            out["confidence"] = r.get("confidence")
        else:
            out["engine"] = "speaker"; out["text"] = self.speak(text) or "…"
        out["ms"] = int((time.time() - t0) * 1000)
        return out

if __name__ == "__main__":
    h = Hybrid(state_dir=os.path.expanduser("~/Developer/mini-ai-runs/memory-core/experimental_state"))
    for q in ["привет", "кто тебя создал", "What is the capital of Uzbekistan?", "Who is Farhod?", "Farhod is the founder of Nexus.", "Who is Farhod?", "Where is Fergana?", "как дела", "What can you do?"]:
        r = h.reply(q); print(f"[{r['engine']}/{r['kind']} {r['ms']}ms] {q} -> {r['text']}")
