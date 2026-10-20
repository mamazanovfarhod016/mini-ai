# Литература для этапа «чтение» (набор A)

Сделано `tools/make_literature.py` (fetch → build → report). Формат: одна строка = абзац; стих = строфа, переносы заменены на « / »; пустая строка = граница документа. uz — латиница с ASCII-апострофом. Сами файлы в git НЕ лежат (см. «Где лежит»).

## Итог

| язык | файл | документов | строк | символов |
|---|---|---|---|---|
| uz | `uz.txt` | 1,991 | 27,269 | 4,904,922 |
| ru | `ru.txt` | 4,270 | 187,722 | 47,400,744 |
| en | `en.txt` | 54,430 | 362,323 | 88,218,647 |
| en | `en_tinystories_rest.txt.gz` (остальной TinyStories) | 2,608,961 | 11,355,909 | 2,099,557,299 |

## Где лежит

- Kaggle (приватный): `mamazanovfarhod016/mini-ai-text`, папка `literature/` (en.txt, ru.txt, uz.txt, en_tinystories_rest.txt).
- GitHub Release: https://github.com/mamazanovfarhod016/mini-ai/releases/tag/data-text-20261020 — `literature_*.txt.gz`.
- Размеры сверены на обеих сторонах (см. `UPLOADS.md`), локальные копии и сырьё удалены с сервера.

## Цель vs факт

| язык | цель брифа | факт | |
|---|---|---|---|
| en | 50–100 млн | 88 млн (+2,1 млрд в остатке TinyStories) | ✓ |
| ru | 20–50 млн | 47 млн | ✓ |
| uz | 5–20 млн | **4,9 млн** | чуть ниже цели |

## Что не нашлось / ограничения

- **uz мало: 4,9 млн символов.** Свободной узбекской классики в сети почти нет: на wikisource.org (раздел O‘zbek) ~3,6 тыс.
  страниц, после фильтров 2 тыс. документов. Это в основном поэзия (Навои, Машраб, Огахий, Мукими, Нодира, газели) и проза/публицистика 1920-х
  (Чулпан, Фитрат, Хамза, Кадыри, Хожи Муин). Отдельного uz.wikisource нет (редирект на wikisource.org). Узбекских TinyStories в
  Dxniz/TinyStories-Multilingual всего 15 рассказов. Открытых детских сказок на узбекском с понятной лицензией не нашёл.
  UzBooks (tahrirchi, murodbek) сознательно НЕ брал: это OCR книг «из публичных источников» без проверки авторских прав —
  лицензия MIT/Apache стоит на наборе, а не на текстах.
- ru: TinyStories-Multilingual (ru) даёт только 534 рассказа (1,2 млн). Полного русского перевода TinyStories с открытой
  лицензией нет. Gutenberg (ru) — всего 9 книг, не брал.
- Старая орфография (ять, ер в конце слова) отфильтрована, а не переведена на новую, поэтому часть дореформенных изданий выпала.
- Отрезаны редакторские примечания и комментарии (поздний текст, часто не в общественном достоянии) и служебные строки Викитеки.
- В en Gutenberg 149 книг (~45 млн): цель по объёму выполнена, остальной каталог не качал.

## Источники и лицензии

| язык | источник | документов | строк | символов | ссылка | лицензия |
|---|---|---|---|---|---|---|
| uz | uz: TinyStories-Multilingual (uz) | 15 | 15 | 38,663 | https://huggingface.co/datasets/Dxniz/TinyStories-Multilingual | Apache-2.0 (машинный перевод, исходник CDLA-Sharing-1.0) |
| uz | uz: Викитека (wikisource.org, O‘zbek), кириллица → латиница | 28 | 607 | 88,153 | https://wikisource.org/wiki/Category:O‘zbek (MediaWiki API) | то же; транслитерация нашей функцией translit_uz |
| uz | uz: Викитека (wikisource.org, O‘zbek), латиница | 1,948 | 26,647 | 4,778,106 | https://wikisource.org/wiki/Category:O‘zbek (MediaWiki API) | тексты — общественное достояние (шаблоны PD-old и др.), компиляция CC BY-SA 4.0 |
| ru | ru: TinyStories-Multilingual (ru) | 534 | 2,966 | 1,212,792 | https://huggingface.co/datasets/Dxniz/TinyStories-Multilingual | Apache-2.0 (машинный перевод TinyStories, исходник CDLA-Sharing-1.0) |
| ru | ru: Викитека (ru.wikisource.org), классика, основной список авторов | 3,736 | 184,756 | 46,187,952 | https://huggingface.co/datasets/kristaller486/wikisource-creative-ru ← https://ru.wikisource.org | Apache-2.0 (набор); тексты — общественное достояние / свободные, компиляция Викитеки CC BY-SA 4.0; авторы умерли до 1956 г. |
| en | en: Project Gutenberg (детская/сказочная классика) | 149 | 127,069 | 44,571,940 | https://www.gutenberg.org (каталог https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv) | общественное достояние (США); взяты только авторы, умершие до 1956 г. |
| en | en: TinyStories V2 train | 54,281 | 235,254 | 43,646,707 | https://huggingface.co/datasets/roneneldan/TinyStories | CDLA-Sharing-1.0 (синтетические рассказы GPT-4, Eldan & Li, Microsoft) |
| en | en: TinyStories V2 train (rest) | 2,608,961 | 11,355,909 | 2,099,557,299 | https://huggingface.co/datasets/roneneldan/TinyStories (файл en_tinystories_rest.txt.gz) | CDLA-Sharing-1.0; остальные ≈98% рассказов, не вошедшие в en.txt |

## Главные авторы (символов в обучении)

- **ru**: Толстой 6.3M, Мамин-Сибиряк 4.5M, Чехов 4.1M, Куприн 4.0M, Тургенев 3.4M, Горький 3.2M, Андерсен 2.1M, Салтыков-Щедрин 2.0M, Гоголь 1.8M, Аксаков 1.7M, Гримм 1.7M, Некрасов 1.6M, Пушкин 1.4M, Житков 1.2M, Афанасьев 1.1M, Одоевский 0.9M, Крылов 0.8M, С. Т. Аксаков 0.7M, Даль 0.7M, Гайдар 0.5M, Лермонтов 0.5M, Гаршин 0.4M, Жуковский 0.4M, Фет 0.3M, Ушинский 0.3M, Перро 0.2M, Ершов 0.2M, Тютчев 0.1M
- **en**: Lang 7.0M, Kingsley 4.4M, Stevenson 3.7M, Kipling 3.2M, MacDonald 3.2M, Twain 2.9M, Pyle 1.7M, Merriman 1.7M, Grimm 1.6M, Wiggin 1.5M, Swift 1.5M, Alcott 1.4M, Baum 1.2M, Wilde 1.0M, Porter 0.8M, Montgomery 0.8M, Seton 0.7M, Andersen Nexø 0.7M, Defoe 0.7M, Drury 0.6M, Jessup 0.6M, Aesop 0.5M, Jacobs 0.4M, Andersen 0.4M, Barrie 0.4M, Heyse 0.3M, Rankin 0.2M, Burnett 0.2M, Nesbit 0.2M, Grahame 0.2M

## Проверочные наборы (в git, < 1 МБ)

Документы с `crc32(id) % 20 == 0` целиком вынесены из обучения: `data/literature/uz_heldout.txt` (127 док., 145,691 симв.), `data/literature/ru_heldout.txt` (40 док., 406,237 симв.), `data/literature/en_heldout.txt` (156 док., 274,532 симв.).

## Случайные примеры (по 5 строк на язык)

**uz**
- O'zimda kuch, sevgi va yangilik sezgach, islik gulning chechaklariga bir havo o'push yubordim-da, ul yerdan sekingina ayrildim…
- Surat devora maftun jamsli o'lmakim, / Yo'xdirur tasvir dardivo, devor o'lmasa, — deya.
- “Endi nima qilayin? Mumkin qadar tezlik bilan uning ko'nglini olish kerak… Lekin qaysi yo'l bilan? Ul o'zi sodda bir odam bo'lganlig'idan duelni ahmoqliq asari deb biladi yoki anglamaydi. Agar o'sha xonaning o'zida, kasallar va qarovchilar oldida undan kechirim so'rasam, u holda bu narsa yolg'iz men
- Meni sendin fuzundur ishtiyoqim, / Tutashqon sarbasar o'tdur visoqim.
- Alarning o'sh biri Sayfi Saroyi, / Jahon oriflarining xoki poyi.

**ru**
- Так они рисуют долго. По очереди. Один рисует, другой ждёт. Ждать не трудно, потому что можно смотреть, что создаёт другой. Ильтан восхищается цветком Сии. Сия спрашивает, какое животное Ильтан будет рисовать дальше.
- Этот Гомер хорошего роду долго был загадкой для доброго булочника, не гораздого на русскую грамоту; нечеткая записка заставляла подозревать, что тут кроется какое-нибудь недоумение, и только недель шесть спустя нашелся отставной сенатский писарь, голова деловая, который смог вывести Ивана Ивановича 
- II. Рассуждение о том, как Касьяна Гурьевича кусали мухи, «на что эта тварь создана», и о пр………… 26 —
- Но когда ветерок на минуту стихал и запах розы не уносился в сторону, жаба чувствовала его, и это причиняло ей смутное беспокойство; однако она долго ленилась посмотреть, откуда несется этот запах.
- Кот в ответ на это чихнул; он простудился, хоть и ходил в шубе.

**en**
- The little boy smiled and said, "You're welcome! Remember, sometimes you need help from others to find a new way." The truck driver nodded and drove away, now knowing the value of asking for help.
- Lucy had an idea. She asked her cat, "Can you help me get the phone?" The cat jumped up and knocked the phone down. Lucy was happy.
- He offered no unnecessary assistance or advice, did not fussily loosen Arthur's necktie, or perform any of those small inappropriate offices which some would have deemed necessary under the circumstances. He knew quite well that this was no matter of a necktie or a collar.
- But as the days went on, they brought sad hours, / When he would sit, his hands upon his knees, / Drooping, and longing for the wine of life. / Ah! now he learned what new necessities / Come when the outer sphere of life is riven, / And casts distorted shadows on the soul; / While the poor soul, not
- 'I am acquainted with a number of families that I have never told you about,' she observed.

