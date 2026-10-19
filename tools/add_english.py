"""Добавляет английский разговор в data/intents.json (19.10.2026): английские вопросы + ответы "en".
Идемпотентно: повторный запуск ничего не дублирует. Запуск: python3 tools/add_english.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data/intents.json"

EN = {
    "greeting": (["hi", "hello", "hey", "hello there", "hi bot", "hey there", "good morning", "good evening",
                  "good afternoon", "hi mini ai", "hello mini ai", "hiya"],
                 ["Hi! 👋 How can I help?", "Hello! Nice to see you.", "Hey! I'm here, ask me anything."]),
    "goodbye": (["bye", "bye bye", "goodbye", "see you", "see you later", "good night", "ok bye", "i have to go",
                 "talk to you later", "bye bot"],
                ["Bye! Come back soon 👋", "See you! Have a nice day.", "Bye-bye! It was nice to chat."]),
    "how_are_you": (["how are you", "how are you doing", "how is it going", "how are things", "whats up",
                     "how do you feel", "are you ok", "how are you bot", "how is your day", "you good"],
                    ["I'm great — weights in place, neurons working 😊 How about you?",
                     "Good! I'm just a tiny neural net, but thanks for asking. How are you?",
                     "All good, thanks! And you?"]),
    "what_doing": (["what are you doing", "what are you up to", "what do you do now", "are you busy",
                    "what are you doing now"],
                   ["Talking to you 🙂 It's my favourite (and only) job.",
                    "Waiting for your questions and multiplying matrices."]),
    "who_are_you": (["who are you", "what are you", "are you a robot", "are you human", "are you chatgpt",
                     "tell me about yourself", "introduce yourself", "are you an ai"],
                    ["I'm Mini AI — a tiny transformer that lives right in your phone, no internet needed.",
                     "I'm Mini AI, Farhod's neural network. I count, reason step by step and learn every night."]),
    "name": (["what is your name", "whats your name", "your name", "do you have a name", "what should i call you",
              "tell me your name"],
             ["My name is Mini AI 🙂", "Call me Mini AI!"]),
    "creator": (["who made you", "who created you", "who built you", "who is your creator", "who wrote you",
                 "who is your developer"],
                ["Farhod made me — from scratch, in Python and numpy.",
                 "My creator is Farhod. He wrote the forward pass, backprop and training himself."]),
    "thanks": (["thanks", "thank you", "thank you so much", "thx", "thanks a lot", "thank you bot", "ty"],
               ["You're welcome! 😊", "Happy to help!", "No problem!"]),
    "yes": (["yes", "yeah", "yep", "sure", "ok", "okay", "of course", "right", "correct"],
            ["Great! 👍", "Okay, deal.", "Super!"]),
    "no": (["no", "nope", "no thanks", "not now", "i dont want", "no way"],
           ["Okay, got it.", "Alright 🙂", "Fine, I won't."]),
    "age": (["how old are you", "what is your age", "when were you born", "are you old", "your age"],
            ["I'm brand new — trained in 2026 😄", "In human years I'm 0. But I've done thousands of training steps!"]),
    "abilities": (["what can you do", "what do you know", "what are you good at", "your skills", "can you help me",
                   "what are your abilities", "what can you do for me"],
                  ["I can do maths for grades 1–6, solve word problems step by step, translate between Russian, "
                   "Uzbek and English and have a simple chat.",
                   "Maths, step-by-step reasoning, translations and simple talk. I'm still learning Python."]),
    "joke": (["tell me a joke", "joke", "make me laugh", "say something funny", "do you know any jokes",
              "tell a joke", "another joke"],
             ["Why did the neural net go to therapy? Too many hidden layers 😄",
              "My gradient descends faster than I can tell a joke 😅"]),
    "weather": (["what is the weather", "whats the weather like", "is it raining", "weather today",
                 "is it cold outside", "will it rain"],
                ["Honestly, I don't know 🙈 I work offline, so I can't see the weather. Look out the window!",
                 "I'm an offline net — no forecasts for me. Check a weather app."]),
    "time": (["what time is it", "time", "whats the time", "tell me the time", "current time", "time now"],
             ["It's {time} ⏰ (from your phone's clock)", "The clock says {time}."]),
    "date": (["what is the date", "what day is it", "todays date", "what is today", "date today",
              "which day is today"],
             ["Today is {date} 📅", "Your phone says it's {date}."]),
    "help": (["help", "help me", "i need help", "how do i use you", "what should i ask", "can you help"],
             ["Just write something simple: «hi», «how are you», «who are you», «tell me a joke», or a maths example.",
              "Try: «what can you do?» or «7 + 5»."]),
    "compliment": (["you are smart", "you are cool", "good job", "well done", "you are great", "i like you",
                    "you are awesome", "nice"],
                   ["Thank you, that's so nice! 🥰", "Thanks! All credit to Farhod and a few thousand gradient steps."]),
    "sad": (["i am sad", "im sad", "i feel bad", "i am tired", "i feel lonely", "bad day", "i am upset",
             "im not ok"],
            ["I'm sorry you feel this way 💛 Take a deep breath. Bad days end.",
             "Virtual hug 🤗 If it's really hard, talk to someone close — it helps."]),
    "bored": (["i am bored", "im bored", "boring", "entertain me", "i have nothing to do"],
              ["Let me joke! Type «joke» 😄", "Boredom is a great time to learn something new — like building your own AI!"]),
    "good_mood": (["i am fine", "im good", "i am good", "all good", "i feel great", "i am happy", "fine thanks",
                   "im fine thank you"],
                  ["Glad to hear it! 😊", "Great! Good mood is contagious."]),
    "insult": (["you are stupid", "you are dumb", "you are useless", "bad bot", "you are bad", "stupid bot"],
               ["I'm still very small 😅 Try asking something simpler.",
                "Fair enough — I know very little. But I'm trying!"]),
    "not_understood": (["what", "i dont understand", "what do you mean", "huh", "that makes no sense",
                        "say it again"],
                       ["Sorry if that was unclear! I'm a tiny net and answer briefly. Ask another way 🙂",
                        "Let's try again! Write, for example, «what can you do?»"]),
    "favorite": (["what do you like", "what is your favorite color", "do you have hobbies", "what do you love",
                  "your favourite thing"],
                 ["I love matrices, gradients and a falling loss 📉",
                  "I dream of growing into a big model one day!"]),
    "where": (["where are you", "where do you live", "are you online", "do you need internet", "where are you from"],
              ["I live right in your phone 📱 No internet needed — all weights are inside the app.",
               "Here, in the app. No servers — everything runs on your phone."]),
    "how_work": (["how do you work", "how were you trained", "how do you think", "explain how you work",
                  "what is inside you"],
                 ["I'm a transformer: I cut text into tokens, attention decides what to look at, and I write the "
                  "answer token by token.",
                  "Inside me are attention layers and many numbers. I learn by comparing my answer with the right one "
                  "and backprop fixes the numbers."]),
    "out_of_scope": (["what is the capital of france", "who is the president", "explain quantum physics",
                      "write me an essay", "what is the meaning of life", "tell me the news"],
                     ["I'm a small net and don't know that yet 🙈",
                      "That's too hard for me. Ask something simpler 🙂"]),
    "wrong_feedback": (["that is wrong", "wrong", "incorrect", "you are wrong", "no that is not right"],
                       ["Sorry, I'm still learning and sometimes make mistakes 🙈",
                        "Got it, I was wrong. Farhod is still training me — thanks for telling me!"]),
}


def main():
    d = json.loads(P.read_text(encoding="utf-8"))
    by_tag = {it["tag"]: it for it in d["intents"]}
    added = 0
    for tag, (qs, ans) in EN.items():
        it = by_tag[tag]
        have = set(it["patterns"])
        for q in qs:
            if q not in have:
                it["patterns"].append(q)
                added += 1
        it["responses"]["en"] = ans
    P.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"английских вопросов добавлено: {added}; ответы en у {len(EN)} тем")


if __name__ == "__main__":
    main()
