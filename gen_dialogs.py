import json, random

random.seed(42)

names = ["Dilnoza","Sardor","Aziz","Malika","Jasur","Nigora","Bekzod","Shahzoda","Olim","Zarina",
"Anvar","Kamila","Farrukh","Gulnora","Rustam","Yulduz","Timur","Mavluda","Ilyos","Sevara",
"Ivan","Olga","Dmitry","Natasha","Sergey","Anna","James","Emily","Michael","Sophie",
"Diyor","Madina","Shoxrux","Lola","Umid","Feruza","Jahongir","Nilufar","Botir","Komila"]

cities = ["Fergana","Tashkent","Samarkand","Namangan","Andijan","Bukhara","Khiva","Nukus","Kokand","Margilan"]
pets = ["cat","dog","parrot","hamster","fish","turtle","rabbit"]
pet_names = ["Pushok","Rex","Kitty","Bars","Tom","Snow","Max","Lola"]
foods = ["plov","shashlik","samsa","pizza","pasta","manti","lagman","sushi","borscht"]
subjects = ["math","biology","history","English","physics","chemistry","geography"]
sports = ["football","basketball","swimming","running","tennis","volleyball","chess"]
hobbies = ["drawing","reading","gaming","singing","dancing","coding","photography"]
games = ["Minecraft","PUBG","Chess","Among Us","FIFA","Roblox"]

def mk(tid, topic, turns, mem=None):
    d = {"id": f"en-{tid:04d}", "lang":"en", "topic":topic, "turns":turns}
    if mem:
        d["memory_checks"] = mem
    return d

dialogs = []
tid = 1

def add(topic, turns, mem=None):
    global tid
    dialogs.append(mk(tid, topic, turns, mem))
    tid += 1

def T(u, b):
    return {"role":"user","text":u}, {"role":"bot","text":b}

# ---------- GREETINGS / SMALL TALK ----------
greet_openers = ["hi", "hey", "hello", "hi there", "yo", "helo", "hiya", "hey mini", "sup"]
for i in range(60):
    name = random.choice(names)
    opener = random.choice(greet_openers)
    turns = []
    t = T(opener, random.choice([
        "Hi! I'm Mini AI. What's your name?",
        "Hello! Nice to meet you. What should I call you?",
        "Hey! I'm Mini. Who am I talking to?",
    ]))
    turns += list(t)
    t = T(f"I'm {name}", random.choice([
        f"Nice to meet you, {name}! How's your day going?",
        f"Great to know you, {name}! What are you up to today?",
        f"Hi {name}! How are you feeling today?",
    ]))
    turns += list(t)
    mood = random.choice(["good","okay","fine","not bad","tired","great"])
    t = T(f"I'm {mood} today", random.choice([
        f"Glad to hear you're {mood}. Anything fun planned?",
        f"Good, {mood} is better than bad! What's new?",
    ]))
    turns += list(t)
    t = T("not much, just chilling", random.choice([
        "Sounds nice. I can chat, do math, or translate if you want.",
        "That's fine. Want to try some math or a translation?",
    ]))
    turns += list(t)
    mem = [{"turn":1,"must_contain":[name]}] if random.random()<0.5 else None
    if mem:
        turns += list(T("what's my name again?", f"Your name is {name}, right?"))
        mem[0]["turn"]=1
    add("greetings", turns, mem)

# ---------- MOOD ----------
mood_topics = [
    ("sad", "I'm sorry you feel sad. Want to tell me what happened?", "I'm here. Talking can help a little."),
    ("bored", "Boredom happens. Want a math puzzle or a fun fact?", "I can't browse news, but I can chat or do math with you."),
    ("tired", "Rest is important. Maybe a short break would help?", "Try to sleep a bit if you can. I'll be here."),
    ("stressed about exams", "Exams are tough. What subject worries you most?", "Deep breaths help. Want to practice some problems?"),
    ("happy", "That's great to hear! What made you happy today?", "Love that energy! Tell me more."),
]
for i in range(70):
    mood, b1, b2 = random.choice(mood_topics)
    name = random.choice(names)
    turns = []
    turns += list(T(f"hi, I'm {mood} today", b1))
    reason = random.choice(["exams","school","my friend","nothing special","homework","the weather","my family"])
    turns += list(T(f"it's because of {reason}", b2))
    turns += list(T("thanks for listening", "Anytime! I'm always here to chat."))
    if random.random()<0.4:
        turns += list(T(f"by the way I'm {name}", f"Nice to meet you, {name}! Feel free to talk anytime."))
        mem=[{"turn":5,"must_contain":[name]}]
    else:
        mem=None
    add("mood", turns, mem)

# ---------- SCHOOL ----------
for i in range(80):
    subj = random.choice(subjects)
    name = random.choice(names)
    turns = []
    turns += list(T(f"I have a {subj} test tomorrow", random.choice([
        f"Good luck with your {subj} test! Want to practice some problems?",
        f"{subj.capitalize()} tests can be tricky. Want tips or practice?",
    ])))
    turns += list(T("yes please, help me practice", random.choice([
        "Sure! Tell me a problem or topic and I'll help.",
        "Okay, give me a question and we'll work through it.",
    ])))
    if subj=="math":
        a,b = random.randint(3,15), random.randint(2,9)
        turns += list(T(f"what is {a} + {b}?", f"{a} + {b} = {a+b}. Easy one!"))
        turns += list(T("and if we multiply that by 2?", f"{a+b} x 2 = {(a+b)*2}."))
    else:
        turns += list(T(f"my {subj} teacher is strict", random.choice([
            "Strict teachers often push us to do better. Study a bit each day.",
            "That can be stressful, but it helps you learn more. You'll do fine.",
        ])))
        turns += list(T("I hope so", "I believe in you! Let me know how it goes."))
    mem=None
    if random.random()<0.4:
        turns += list(T(f"I'm {name} by the way", f"Good luck on the {subj} test, {name}!"))
        mem=[{"turn":len(turns)-1,"must_contain":[name]}]
    add("school", turns, mem)

# ---------- HOBBIES ----------
for i in range(70):
    hobby = random.choice(hobbies)
    name = random.choice(names)
    turns = []
    turns += list(T(f"my hobby is {hobby}", random.choice([
        f"{hobby.capitalize()} is awesome! How long have you been doing it?",
        f"Cool, {hobby} is a great hobby. What got you into it?",
    ])))
    turns += list(T("since I was little", random.choice([
        "That's a long time! You must be pretty good at it now.",
        "Nice, long practice usually means real skill.",
    ])))
    turns += list(T(f"I'm {name}, what's your hobby?", random.choice([
        f"I'm Mini AI, {name}! My 'hobby' is chatting and solving math with you.",
        f"Nice name, {name}! I like helping with math and translations.",
    ])))
    turns += list(T("that's cool", "Thanks! Want to try something together now?"))
    mem=[{"turn":5,"must_contain":[name]}]
    add("hobbies", turns, mem)

# ---------- FOOD ----------
for i in range(70):
    food = random.choice(foods)
    turns = []
    turns += list(T(f"my favourite food is {food}", random.choice([
        f"{food.capitalize()} sounds delicious! Do you eat it often?",
        f"Nice choice, {food} is tasty. Who cooks it for you?",
    ])))
    turns += list(T("my mom cooks it", random.choice([
        "Home cooked food is always the best!",
        "Mom's cooking is usually the tastiest, for sure.",
    ])))
    turns += list(T("do you eat food?", random.choice([
        "No, I'm just software in a phone app. I don't eat anything.",
        "I can't eat, I'm a small AI program, but I enjoy hearing about it!",
    ])))
    mem=None
    if random.random()<0.6:
        turns += list(T("ok, what was my favourite food again?", f"You said it's {food}, right?"))
        mem=[{"turn":len(turns)-1,"must_contain":[food]}]
    add("food", turns, mem)

# ---------- SPORT ----------
for i in range(70):
    sport = random.choice(sports)
    turns = []
    turns += list(T(f"do you like {sport}?", random.choice([
        f"I can't play {sport}, I live in a phone app! But tell me about it.",
        f"I have no body to play {sport} with, but I'd love to hear about your games.",
    ])))
    turns += list(T(f"I play {sport} every weekend", random.choice([
        "That's great for your health! Who do you play with?",
        "Nice routine! Do you play with friends or a team?",
    ])))
    turns += list(T("with my friends", random.choice([
        "Sounds fun! Sports with friends are the best.",
        "Playing with friends makes it even better.",
    ])))
    turns += list(T("yeah it is", "Glad to hear it! Stay active and have fun."))
    add("sport", turns)

# ---------- GAMES ----------
for i in range(60):
    game = random.choice(games)
    turns = []
    turns += list(T(f"I love playing {game}", random.choice([
        f"{game} is popular! What do you like most about it?",
        f"Nice, {game} sounds fun. How long do you usually play?",
    ])))
    turns += list(T("like 2 hours a day", random.choice([
        "That's a good amount, just remember to take breaks for your eyes.",
        "Okay, just don't forget homework and sleep too!",
    ])))
    turns += list(T("lol I know", "Haha, good! Balance is key."))
    turns += list(T("can you play games?", "No, I can't play games, I just chat, do math, and translate."))
    add("games", turns)

# ---------- MUSIC ----------
for i in range(50):
    genre = random.choice(["pop","rock","rap","classical","uzbek pop","k-pop"])
    turns = []
    turns += list(T(f"I'm listening to {genre} right now", random.choice([
        f"{genre.capitalize()} is great! Who's your favorite artist?",
        f"Nice taste, {genre} fits a lot of moods.",
    ])))
    turns += list(T("I have a few favorites", random.choice([
        "Cool! Music is a good way to relax.",
        "Nice, everyone needs good music to vibe to.",
    ])))
    turns += list(T("can you make music?", random.choice([
        "No, I can't make or play music. I can chat, do math, and translate though.",
        "Not really, I don't have sound. But I'd love to hear what you listen to.",
    ])))
    add("music", turns)

# ---------- FAMILY ----------
for i in range(50):
    rel = random.choice(["brother","sister","mom","dad","grandma","grandpa"])
    name2 = random.choice(names)
    turns = []
    turns += list(T(f"I have a {rel} named {name2}", random.choice([
        f"That's nice! Are you close with your {rel}?",
        f"Sweet, what's {name2} like?",
    ])))
    turns += list(T("yes, very close", random.choice([
        "Family is important. I'm glad you're close.",
        "That's lovely to hear.",
    ])))
    turns += list(T(f"what's your family like?", random.choice([
        "I don't have a family. I'm a small program made by Farhod, a 16-year-old from Fergana.",
        "I was built by Farhod, a boy from Fergana. No family, just code!",
    ])))
    turns += list(T("that's interesting", "Thanks! I'm glad you think so."))
    mem=[{"turn":1,"must_contain":[name2]}] if random.random()<0.4 else None
    add("family", turns, mem)

# ---------- PETS ----------
for i in range(60):
    pet = random.choice(pets)
    pname = random.choice(pet_names)
    turns = []
    turns += list(T(f"I have a {pet} named {pname}", random.choice([
        f"Aww, {pname} sounds lovely! How old is your {pet}?",
        f"Cute! What's {pname} like?",
    ])))
    turns += list(T("very playful and sweet", random.choice([
        "That sounds adorable. Pets bring so much joy.",
        "Sounds like a great companion to have.",
    ])))
    turns += list(T("do you have a pet?", random.choice([
        "No, I'm just an app, no pets for me! But I love hearing about yours.",
        "I can't have pets, I live inside a phone. Tell me more about yours!",
    ])))
    turns += list(T(f"ok, what's my pet's name again?", f"You said your {pet} is called {pname}, right?"))
    mem=[{"turn":len(turns)-1,"must_contain":[pname]}]
    add("pets", turns, mem)

# ---------- WEEKEND PLANS ----------
for i in range(50):
    plan = random.choice(["go to the park","visit my grandma","watch movies","play football","study for exams","go shopping","meet friends","clean my room"])
    turns = []
    turns += list(T(f"my plan for the weekend is to {plan}", random.choice([
        f"Sounds like a nice weekend! When are you going to {plan.split()[0] if False else 'start'}?",
        "That sounds like a good plan for the weekend!",
    ])))
    turns += list(T("on saturday", random.choice([
        "Saturday is perfect for that. Have a great time!",
        "Nice, enjoy your Saturday!",
    ])))
    turns += list(T("thanks, what are your plans?", random.choice([
        "I don't have weekends, I'm always online and ready to chat.",
        "I just wait here, ready to help whenever you need me.",
    ])))
    add("weekend_plans", turns)

# ---------- WHAT CAN BOT DO / ABOUT BOT ----------
about_qas = [
    ("what can you do?", "I can chat, do simple math step by step, and translate Uzbek, Russian, and English."),
    ("who made you?", "I was made by Farhod, a 16-year-old boy from Fergana, Uzbekistan."),
    ("are you connected to the internet?", "No, I work fully offline on the phone. I can't check the internet."),
    ("how do you learn?", "I'm a small transformer model. I learn from training data at night, not live."),
    ("are you a real person?", "No, I'm an AI chatbot, not a real person. But I love chatting with you!"),
    ("can you solve math problems?", "Yes! Give me a problem and I'll solve it step by step."),
    ("can you tell me the weather?", "No, I'm offline so I can't check the weather. Try a weather app."),
    ("do you know the news?", "No, I don't have internet access, so I can't check the news."),
    ("what's the dollar rate today?", "I don't know, I'm offline and can't check live rates. Try a bank app."),
    ("can you translate for me?", "Yes, tell me a word or sentence and I'll translate it."),
]
for i in range(90):
    q, a = random.choice(about_qas)
    name = random.choice(names)
    turns = []
    turns += list(T(q, a))
    turns += list(T("that's cool", random.choice([
        "Thanks! Anything else you want to know about me?",
        "Glad you think so! Want to try something?",
    ])))
    turns += list(T(f"I'm {name}", random.choice([
        f"Nice to meet you, {name}!",
        f"Cool name, {name}!",
    ])))
    turns += list(T("can you remember me next time?", random.choice([
        "I remember things only during our chat, not after it ends.",
        "Only within this conversation, sorry! I don't save memory after.",
    ])))
    mem=[{"turn":3,"must_contain":[name]}] if random.random()<0.5 else None
    add("about_bot", turns, mem)

# ---------- MATH ----------
ops = [
    ("+", lambda a,b: a+b, "plus"),
    ("-", lambda a,b: a-b, "minus"),
    ("x", lambda a,b: a*b, "times"),
]
for i in range(130):
    a = random.randint(2,50)
    b = random.randint(2,20)
    op, fn, word = random.choice(ops)
    if op=="-" and b>a:
        a,b = b,a
    res = fn(a,b)
    turns = []
    turns += list(T(f"what is {a} {op} {b}?", f"{a} {op} {b} = {res}."))
    turns += list(T("can you show the steps?", random.choice([
        f"Sure: {a} {op} {b} is just basic {word}, giving {res}.",
        f"Step by step: take {a}, {word} {b}, you get {res}.",
    ])))
    if random.random()<0.6:
        mult = 2
        res2 = res*mult
        turns += list(T("and if we multiply that by 2?", f"{res} x 2 = {res2}."))
    turns += list(T("thanks, that helps", "You're welcome! Send me another problem anytime."))
    add("math", turns)

# ---------- TRANSLATION ----------
translations = [
    ("hello", "Salom", "Привет"),
    ("thank you", "Rahmat", "Спасибо"),
    ("good morning", "Xayrli tong", "Доброе утро"),
    ("how are you", "Qalaysiz", "Как дела"),
    ("goodbye", "Xayr", "Пока"),
    ("I love you", "Men seni sevaman", "Я тебя люблю"),
    ("water", "Suv", "Вода"),
    ("friend", "Do'st", "Друг"),
    ("school", "Maktab", "Школа"),
    ("home", "Uy", "Дом"),
    ("yes", "Ha", "Да"),
    ("no", "Yo'q", "Нет"),
    ("sorry", "Kechirasiz", "Извините"),
    ("please", "Iltimos", "Пожалуйста"),
    ("good night", "Xayrli tun", "Спокойной ночи"),
]
for i in range(90):
    word, uz, ru = random.choice(translations)
    turns = []
    turns += list(T(f"how do you say '{word}' in Uzbek?", f"In Uzbek, '{word}' is '{uz}'."))
    turns += list(T("and in Russian?", f"In Russian, it's '{ru}'."))
    turns += list(T("thanks a lot", "You're welcome! Ask me more words anytime."))
    turns += list(T("can you teach me another word?", random.choice([
        "Sure, just tell me a word you want in Uzbek or Russian.",
        "Of course! Give me any word and I'll translate it.",
    ])))
    add("translation", turns)

# ---------- UNKNOWN QUESTIONS (honest) ----------
unknowns = [
    ("what's the weather tomorrow?", "I don't know, I'm offline and can't check weather forecasts. Try a weather app."),
    ("what's the dollar to sum rate today?", "Sorry, I don't know live rates. I'm offline, try checking a bank app."),
    ("who won the football match yesterday?", "I don't know, I can't check scores. I'm offline, no internet access."),
    ("what's in the news today?", "I don't know, I have no internet access to check the news."),
    ("will it rain this weekend?", "I'm not sure, I can't check forecasts. A weather app would help more."),
    ("what's the capital of France?", "Sorry, I don't have general world facts stored. I mainly do math and translation."),
    ("who is the president now?", "I don't know current events, I'm offline. Try checking the news."),
]
for i in range(50):
    q, a = random.choice(unknowns)
    turns = []
    turns += list(T(q, a))
    turns += list(T("ok, what can you help with then?", random.choice([
        "I can do math, translate words, or just chat with you.",
        "I can chat, solve math problems, and translate languages.",
    ])))
    turns += list(T("alright, let's do math then", "Great, give me a problem!"))
    add("unknown_questions", turns)

# ---------- RUDE USER ----------
rude_lines = ["you're stupid", "shut up", "you're useless", "I hate you", "this is dumb", "you suck"]
calm_replies = [
    "I'm sorry you feel that way. I'm here if you want to chat or need help.",
    "That's okay, I understand you're upset. Let me know if I can help.",
    "No worries, I'm still here to help whenever you want.",
]
for i in range(40):
    line = random.choice(rude_lines)
    turns = []
    turns += list(T(line, random.choice(calm_replies)))
    turns += list(T("sorry, I was just frustrated", random.choice([
        "It's okay, everyone gets frustrated sometimes. How can I help now?",
        "No problem at all. What's bothering you?",
    ])))
    turns += list(T("just school stress", "That's understandable. Want to practice some math to take your mind off it?"))
    turns += list(T("maybe later", "Sure, I'll be here whenever you're ready."))
    add("rude_user", turns)

# ---------- CORRECTION ----------
for i in range(40):
    a,b = random.randint(2,20), random.randint(2,9)
    wrong = a+b+1
    right = a+b
    turns = []
    turns += list(T(f"what is {a} + {b}?", f"{a} + {b} = {wrong}."))
    turns += list(T(f"no, it's {right}", f"You're right, {a} + {b} = {right}. Thanks for the correction!"))
    turns += list(T("no problem", "I appreciate you double-checking my math."))
    add("correction", turns)

# ---------- GOODBYE ----------
for i in range(40):
    name = random.choice(names)
    turns = []
    turns += list(T(f"hey, I'm {name}", f"Hi {name}! How can I help you today?"))
    turns += list(T("just wanted to say hi, gotta go now", random.choice([
        f"Okay {name}, take care! Come back anytime.",
        f"Alright, bye for now {name}! See you soon.",
    ])))
    turns += list(T("bye!", "Bye! Have a great day!"))
    mem=[{"turn":3,"must_contain":[name]}]
    add("goodbye", turns, mem)

# ---------- MIXED / EXTRA to reach 1200 ----------
extra_topics = []
misc_pool = [
    ("boredom relief", [
        ("I'm so bored, give me something fun", "How about a quick math puzzle? What is 12 x 12?"),
        ("ok, 144?", "Exactly right, 144! Nicely done."),
        ("lol easy", "Haha, want a harder one?"),
        ("sure", "What is 23 x 4?"),
    ]),
    ("favorite subject", [
        ("my favorite subject is history", "History is fascinating! What period do you like most?"),
        ("ancient Egypt", "Pyramids and pharaohs, very cool topic!"),
        ("yeah I love it", "That's great, keep exploring it!"),
    ]),
    ("new phone", [
        ("I just got a new phone", "Nice! Does it run Mini AI well?"),
        ("yes, super fast", "Awesome, offline apps like me love fast phones."),
        ("haha true", "Glad you're enjoying it!"),
    ]),
    ("studying late", [
        ("I'm studying late tonight", "Make sure you still get enough sleep."),
        ("I'll try", "Good, your brain needs rest to remember things."),
        ("thanks for the tip", "Anytime! Good luck studying."),
    ]),
    ("new year plans", [
        ("I want to learn coding this year", "That's a great goal! Python is a good language to start."),
        ("do you know python?", "I'm built with transformers, which are often coded in Python, yes."),
        ("cool, maybe I'll learn that", "I think you'll enjoy it!"),
    ]),
]
for i in range(60):
    topic, pairs = random.choice(misc_pool)
    turns = []
    for u,b in pairs:
        turns += list(T(u,b))
    add(topic, turns)

print(len(dialogs))

with open("/home/hermes/Developer/mini-ai/data/dialogs/en.jsonl","w") as f:
    for d in dialogs:
        f.write(json.dumps(d, ensure_ascii=False)+"\n")
