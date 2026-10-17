"""
Инференс (использование уже обученной сети) по файлу model.json.
Ровно то же самое делает app/www/brain.js в телефоне.
"""
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np

from .features import fnv1a, is_latin, tokens, vectorize
from .mlp import relu, softmax

ROOT = Path(__file__).resolve().parent.parent
UNKNOWN_RU = "Я пока маленькая нейронка и этого не знаю 🙈 Попробуй что-то попроще: «привет», «как дела», «кто ты»."
UNKNOWN_UZ = "Men hali kichkina neyron tarmoqman va buni bilmayman 🙈 Oddiyroq so'rang: «salom», «qalaysan»."


class Brain:
    def __init__(self, path=ROOT / "model.json"):
        m = json.loads(Path(path).read_text(encoding="utf-8"))
        self.m = m
        D, H, K, S = m["dim"], m["hidden"], len(m["intents"]), m["scale"]
        self.dim = D
        # веса в файле — целые числа (w*scale), возвращаем к дробям
        self.W1 = np.array(m["W1"], dtype=np.float64).reshape(D, H) / S
        self.b1 = np.array(m["b1"], dtype=np.float64) / S
        self.W2 = np.array(m["W2"], dtype=np.float64).reshape(H, K) / S
        self.b2 = np.array(m["b2"], dtype=np.float64) / S
        self.known = set(m["known"])
        self.threshold = m["threshold"]

    def think(self, text):
        """Возвращает всё, что «подумала» сеть: намерение, уверенность, топ-3."""
        feats = tokens(text)
        known = sum(1 for f in feats if fnv1a(f) in self.known)
        x = np.array([vectorize(text, self.dim)])
        p = softmax(relu(x @ self.W1 + self.b1) @ self.W2 + self.b2)[0]
        order = np.argsort(-p, kind="stable")
        top = [(self.m["intents"][i], float(p[i])) for i in order[:3]]
        intent, conf = top[0]
        # Честность: если знакомых кусочков меньше трети или сеть не уверена —
        # отвечаем «не знаю», а не выдумываем.
        familiar = known / len(feats) if feats else 0.0
        if familiar < 0.34 or conf < self.threshold:
            intent = None
        return {"intent": intent, "confidence": conf, "top3": top, "familiar": familiar}

    def reply(self, text, rng=random, now=None):
        t = self.think(text)
        lang = "uz" if is_latin(text) else "ru"
        if t["intent"] is None:
            return (UNKNOWN_UZ if lang == "uz" else UNKNOWN_RU), t
        answer = rng.choice(self.m["responses"][t["intent"]][lang])
        now = now or datetime.now()
        answer = answer.replace("{time}", now.strftime("%H:%M")).replace("{date}", now.strftime("%d.%m.%Y"))
        return answer, t
