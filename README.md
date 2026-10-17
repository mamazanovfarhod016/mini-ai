# Mini AI — своя нейросеть с нуля 🧠📱

**Mini AI** — крошечная нейросеть-собеседник, написанная **с нуля на Python + numpy** (без PyTorch и TensorFlow),
и **офлайн Android-приложение** с тёплым «мобильным» интерфейсом чата. Нейросеть понимает простые фразы на
**русском и узбекском** («привет», «как дела», «кто тебя создал», «salom», «qalaysan», «rahmat»…), терпит опечатки
(«превет», «как дила») и честно говорит «не знаю», если вопрос ей не по силам.

> Автор: **Фарход Мамазанов**, учебный проект «нейронка с нуля». Это **не ChatGPT** и не генеративная модель —
> это маленький классификатор намерений: он угадывает *тип* фразы и отвечает одной из заготовок.

<p>
<img src="docs/screenshots/1_empty_chat.png" width="200">
<img src="docs/screenshots/2_dialog.png" width="200">
<img src="docs/screenshots/3_how_it_works.png" width="200">
<img src="docs/screenshots/2_dialog_dark.png" width="200">
</p>

## Цифры

| | |
|---|---|
| Архитектура | MLP: 2048 → 32 (ReLU) → 27 (softmax) |
| Параметров | **66 459** |
| Размер модели (`model.json`) | **≈ 301 КБ** |
| Датасет | 27 намерений, **1 574 фразы** (ru + uz), + аугментация опечатками |
| Точность на тесте (20% фраз, сеть их не видела) | **88.1%** в среднем по 3 сплитам (86.2 / 91.0 / 87.2) |
| …только разговорные намерения (без «сложных вопросов») | **90.4%** (88.4 / 93.4 / 89.4) |
| Обучение | ~20 секунд на CPU |
| APK | ≈ 182 КБ, **без разрешения INTERNET** |

Полный лог обучения и список ошибок — [`docs/train_log.txt`](docs/train_log.txt), матрица ошибок — [`docs/confusion_matrix.csv`](docs/confusion_matrix.csv).

## Как устроена сеть

```
 «как дила?»
     │  normalize: нижний регистр, без знаков, ё→е, без апострофов
     ▼
 признаки: w:как  w:дила  p:дила  c:␣ка  c:как  c:ак␣  c:␣ди  c:дил  c:ила  c:ла␣
     │  hashing trick: FNV-1a(признак) % 2048  →  мешок признаков, L2-нормализация
     ▼
 x ∈ R^2048 ──[ W1: 2048×32, b1 ]──► ReLU ──► h ∈ R^32
                                              │
                         [ W2: 32×27, b2 ] ◄──┘
                                              ▼
                          softmax ──► p ∈ R^27  (вероятности намерений)
                                              ▼
            argmax → «how_are_you» 99.7% → случайный ответ из 2–4 вариантов
            (если уверенность < 45% или фраза почти незнакомая → «я пока маленькая и не знаю»)
```

* **Символьные триграммы** — главный трюк против опечаток: у «как дила» и «как дела» большая часть кусочков общая.
* **Hashing trick** — словарь не хранится, вход всегда фиксированной длины → модель маленькая.
* **Всё вручную** ([`mini_ai/mlp.py`](mini_ai/mlp.py)): прямой проход, softmax, cross-entropy, обратное
  распространение ошибки (`dL/dlogits = p − onehot(y)` → цепное правило назад), L2-регуляризация, **Adam** (и SGD для сравнения).
* **Аугментация** ([`train.py`](train.py)): к каждой фразе добавляются 3 версии с опечатками (пропуск/повтор/замена/перестановка буквы),
  версия без одного слова и версия со «словом-паразитом» («ну», «бот», «do'stim»).
* **Квантизация весов**: в `model.json` веса хранятся как `round(w·1000)` — целые числа, файл в 3 раза меньше.
* **Честность**: отдельный класс `out_of_scope` («реши уравнение», «столица Франции») + порог уверенности +
  проверка «знакомых кусочков» → ответ «я пока маленькая нейронка и этого не знаю».
* **Время и дата** берутся с часов телефона, погода — честно «не знаю, я офлайн».

## Структура

```
data/intents.json        датасет: 27 намерений, фразы-примеры, ответы ru/uz
mini_ai/features.py      текст → признаки → вектор (hashing trick)
mini_ai/mlp.py           MLP + backprop + Adam/SGD (numpy, с комментариями)
mini_ai/brain.py         инференс в Python по model.json
train.py                 обучение, точность на тесте, матрица ошибок → model.json
app/www/brain.js         ТОТ ЖЕ прямой проход на JavaScript (в телефоне)
app/www/{index.html,style.css,app.js}   интерфейс чата
android/                 сборка APK без Gradle (aapt2 + javac + d8 + apksigner)
tests/                   gradient check, контрольные фразы, сверка Python == JS
tools/                   скриншоты и иконки (headless Chrome)
```

## Как обучить и проверить

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python train.py          # → model.json + app/www/model.js, печатает точность и ошибки
.venv/bin/python -m pytest -q      # 31 тест: gradient check, 25 контрольных фраз, Python==JS (нужен node)
```

Тесты:
* **gradient check** — градиенты из backprop сравниваются с численными `(L(w+ε) − L(w−ε)) / 2ε`, относительная ошибка < 1e-6;
* **контрольные фразы** — «превет», «как дила», «qalaysan», «yo'q», «кто тебя создал» и др. распознаются правильно;
* **сверка Python ↔ JavaScript** — на ~180 фразах намерение, топ-3 и уверенность совпадают до 1e-9.

## Как собрать APK

Нужны JDK 17 и Android SDK (build-tools 34, platform android-34). Gradle не нужен.

```bash
# один раз: свой ключ подписи (хранится ВНЕ репозитория)
mkdir -p ~/.config/mini-ai && openssl rand -base64 18 > ~/.config/mini-ai/keystore.txt
keytool -genkeypair -keystore ~/.config/mini-ai/release.jks -alias miniai -keyalg RSA -keysize 2048 \
  -validity 10000 -storepass "$(cat ~/.config/mini-ai/keystore.txt)" -keypass "$(cat ~/.config/mini-ai/keystore.txt)" \
  -dname "CN=Your Name"

./android/build.sh                 # → android/build/MiniAI.apk (подписан, проверен, без INTERNET)
```

Готовый APK — в [Releases](../../releases). Интерфейс можно открыть и просто в браузере: `app/www/index.html`.
Название приложения меняется одной константой `APP_NAME` в `app/www/app.js` (и `app_name` в `android/res/values/styles.xml`).

## Интерфейс

Тёплый кремовый фон, название модели по центру шапки, меню слева и «новый чат» справа, крупное приветствие с
serif-акцентом, сообщения пользователя — в облачках справа, ответы — текстом слева со звёздочкой, круглая
терракотовая кнопка отправки, чипы-подсказки, «печатает…» и посимвольное появление ответа, история чатов
(localStorage), тёмная тема по системной настройке. Под каждым ответом — чип «🧠 намерение · %», который открывает
шторку **«Как я работаю»**: топ-3 намерения с вероятностями, активность 32 нейронов скрытого слоя, признаки фразы и схема сети.
Иконка и логотип — собственная 8-лучевая звёздочка. Все шрифты системные, никаких CDN — работает без интернета.

## Ограничения (честно)

* Это классификатор, а не языковая модель: он не генерирует новый текст, а выбирает заготовку.
* Понимает только ~27 типов фраз. Длинные и сложные вопросы → «не знаю».
* Точность ~88–90% на незнакомых фразах: иногда путает близкие темы (например, «у меня всё хорошо» и «как дела»).

---

## English

**Mini AI** is a tiny chatbot neural network written **from scratch in numpy** (no PyTorch/TensorFlow) plus an
**offline Android app** with a warm, mobile chat UI. It's an intent classifier: text → bag of words + character
trigrams (typo-tolerant) → feature hashing (2048) → hidden layer 32 ReLU → softmax over 27 intents (Russian + Uzbek).
Forward pass, backprop, cross-entropy, L2 and the Adam optimizer are implemented by hand and verified with a numerical
gradient check. 66,459 parameters, ~301 KB model, **~88% test accuracy** on held-out phrases (90% on conversational
intents). The same forward pass is re-implemented in JavaScript and runs on the phone with **no INTERNET permission**;
a test checks Python and JS outputs match to 1e-9. The app includes a "How I work" sheet that shows the top-3 intents,
confidence and hidden-layer activations for every reply. Built by **Farhod Mamazanov** as a learning project.

```bash
.venv/bin/python train.py && .venv/bin/python -m pytest -q && ./android/build.sh
```
