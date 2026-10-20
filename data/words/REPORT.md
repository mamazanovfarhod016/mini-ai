# Значения слов (стадия «слова»)

Собрано скриптом `tools/make_word_meanings.py` из Викисловаря (CC BY-SA). Задания q→a на языке слова,
по 3 формулировки: **meaning** (что значит), **synonym**, **antonym**, **odd** (лишнее слово — другая часть речи,
части речи по Викисловарю, только слова с одной частью речи), **fill** (вставь слово, 3 варианта ответа).

**Экзамен:** слово спрятано целиком, если `crc32(ключ) % 10 == 0` (ключ: нижний регистр, uz-апостроф `'`, ru ё→е).
Учебное задание выбрасывается, если экзаменационная лемма есть где угодно в вопросе или ответе (включая
определение, пример, варианты) — проверка по леммам всех слов строки. Для учёбы берётся то значение/пример, где
спрятанных слов нет; слова из шаблонов вопросов в экзамен не прячутся. Тест: `tests/test_word_meanings.py`.

| язык | частых лемм (headwords) | из них экзамен | заданий учёба | символов учёба | заданий экзамен (полный / в git) | определение / только перевод / синонимы / антонимы / пример |
|---|---|---|---|---|---|---|
| uz | 3,529 | 366 | 14,865 | 1,675,842 | 2,472 / 783 | 3,159 / 370 / 249 / 77 / 2,894 |
| ru | 9,993 | 1,034 | 73,749 | 8,001,220 | 11,064 / 3,342 | 9,993 / 0 / 5,486 / 3,624 / 5,171 |
| en | 6,422 | 676 | 26,845 | 3,021,767 | 5,226 / 1,507 | 6,422 / 0 / 1,922 / 291 / 3,537 |

## Задания по типам (учёба)

| язык | meaning | synonym | antonym | odd | fill | выброшено (утечка экзамена) |
|---|---|---|---|---|---|---|
| uz | 2,892 | 282 | 48 | 9,036 | 2,607 | 6,696 |
| ru | 23,367 | 12,561 | 5,610 | 23,895 | 8,316 | 19,653 |
| en | 9,036 | 1,750 | 0 | 6,990 | 9,069 | 16,466 |

## Частоты (корпус «чтение»)

- uz: 1,415,187 слов, 150,320 разных; файлы: `uz_heldout.txt`, `uz.txt`, `uz_test.txt`, `dxniz.jsonl`, `uz_wikisource.jsonl`; без леммы в словаре: 48% словоупотреблений; частота самого редкого взятого слова: 3
- ru: 99,048,638 слов, 609,835 разных; файлы: `ru_heldout.txt`, `ru.txt`, `ru_test.txt`, `ruws_0.parquet`, `ruws_1.parquet`, `ruws_2.parquet`, `ruws_3.parquet`, `ruws_4.parquet`, `dxniz.jsonl`; без леммы в словаре: 19% словоупотреблений; частота самого редкого взятого слова: 774
- en: 40,475,345 слов, 13,623 разных; файлы: `en_heldout.txt`, `en.txt`, `en_test.txt`, `TinyStoriesV2-GPT4-train.txt`; без леммы в словаре: 4% словоупотреблений; частота самого редкого взятого слова: 10

## Источники и лицензии

- uz: uz.wiktionary.org — дамп pages-articles (определения и примеры на узбекском) — https://dumps.wikimedia.org/uzwiktionary/latest/uzwiktionary-latest-pages-articles.xml.bz2 — CC BY-SA 4.0 / GFDL
- uz: kaikki.org — English Wiktionary, Uzbek (переводы, формы, синонимы) — https://kaikki.org/dictionary/Uzbek/kaikki.org-dictionary-Uzbek.jsonl — CC BY-SA 4.0 / GFDL
- ru: kaikki.org — Russian Wiktionary (ru.wiktionary), Русский (wiktextract) — https://kaikki.org/ruwiktionary/Русский/kaikki.org-dictionary-Русский.jsonl — CC BY-SA 4.0 / GFDL
- en: kaikki.org — English Wiktionary, English (wiktextract) — https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl — CC BY-SA 4.0 / GFDL

Лицензия набора: CC BY-SA 4.0 (производное от Викисловаря; авторы — участники Wiktionary, извлечение — wiktextract, Tatu Ylonen, kaikki.org).

## Чего нет / ограничения

- uz: в узбекском Викисловаре синонимов и антонимов мало (249 и 77 слов из 3 529); 370 слов — только английский перевод (kaikki), узбекского определения нет. 48% словоупотреблений корпуса без леммы (узбекская морфология; отсечение аффиксов по таблице).
- uz: частых слов меньше 5 тыс. — словарь и корпус малы, взяты все с частотой ≥ 3.
- en: антонимов в учёбе 0 — почти все пары уходят в экзамен вместе со спрятанным словом или неупотребительны; антонимы в экзамене есть (60).
- en: 6 422 слова (< 10 тыс.): TinyStories простой, в нём ~13,6 тыс. разных слов; взяты слова с частотой ≥ 10.

## Где лежит

- В git: `data/words/exam_{uz,ru,en}.jsonl` (экзамен, одна формулировка на слово×тип, < 1 МБ), `data/words/exam_words_{lang}.txt` (список спрятанных слов), этот отчёт.
- Полные файлы (`words_{lang}_train.jsonl`, `words_{lang}_exam.jsonl`, `lexicon_{lang}.jsonl`): см. раздел ниже.

## Загрузки (проверено размером)

- Kaggle (приватный): https://www.kaggle.com/datasets/mamazanovfarhod016/mini-ai-text — папка `words/` — загружено ✓
- GitHub Release: https://github.com/mamazanovfarhod016/mini-ai/releases/tag/data-text-20261020

| файл | байт | Release (.gz) |
|---|---|---|
| lexicon_en.jsonl | 3,375,406 | [скачать](https://github.com/mamazanovfarhod016/mini-ai/releases/download/data-text-20261020/words_lexicon_en.jsonl.gz) |
| lexicon_ru.jsonl | 6,669,544 | [скачать](https://github.com/mamazanovfarhod016/mini-ai/releases/download/data-text-20261020/words_lexicon_ru.jsonl.gz) |
| lexicon_uz.jsonl | 2,028,361 | [скачать](https://github.com/mamazanovfarhod016/mini-ai/releases/download/data-text-20261020/words_lexicon_uz.jsonl.gz) |
| words_en_exam.jsonl | 990,919 | [скачать](https://github.com/mamazanovfarhod016/mini-ai/releases/download/data-text-20261020/words_words_en_exam.jsonl.gz) |
| words_en_train.jsonl | 4,916,461 | [скачать](https://github.com/mamazanovfarhod016/mini-ai/releases/download/data-text-20261020/words_words_en_train.jsonl.gz) |
| words_ru_exam.jsonl | 3,315,809 | [скачать](https://github.com/mamazanovfarhod016/mini-ai/releases/download/data-text-20261020/words_words_ru_exam.jsonl.gz) |
| words_ru_train.jsonl | 20,540,006 | [скачать](https://github.com/mamazanovfarhod016/mini-ai/releases/download/data-text-20261020/words_words_ru_train.jsonl.gz) |
| words_uz_exam.jsonl | 449,100 | [скачать](https://github.com/mamazanovfarhod016/mini-ai/releases/download/data-text-20261020/words_words_uz_exam.jsonl.gz) |
| words_uz_train.jsonl | 2,732,861 | [скачать](https://github.com/mamazanovfarhod016/mini-ai/releases/download/data-text-20261020/words_words_uz_train.jsonl.gz) |


## Примеры (5 случайных учебных на язык)

**uz**
- `meaning` — Q: ishlatmoq so'zi nimani anglatadi? → A: ishlatmoq — biror manfaatga yo'naltirmoq.
- `fill` — Q: so'zni qo'yib gapni yoz (ilmiy, changimoq, hafta): Unga o'g'lidan har ___ xat keladi. → A: Unga o'g'lidan har hafta xat keladi.
- `meaning` — Q: yoyilmoq so'zi nimani anglatadi? → A: yoyilmoq — to'shalmoq, yozilmoq.
- `odd` — Q: ortiqcha so'zni top: badan, daraja, sovurmoq, la'nat → A: Ortiqcha so'z — sovurmoq: u fe'l, qolganlari — ot.
- `fill` — Q: gapga mos so'zni qo'y (ko'pirtirmoq, olam, yalang'och): Umrini shaharda, shovqinli sexlarda o'tkazgan bu yigit uchun qishloq mutlaqo yangi ___ edi. → A: Umrini shaharda, shovqinli sexlarda o'tkazgan bu yigit uchun qishloq mutlaqo yangi olam edi.

**ru**
- `meaning` — Q: что значит слово «нива»? → A: «нива» — поле, возделанное для посева или засеянное хлебом.
- `antonym` — Q: какое слово обратно по смыслу «поднимать»? → A: Антоним к слову «поднимать» — «опускать», «спускать».
- `odd` — Q: какое слово лишнее: бережный, приёмный, кадр, временной? → A: Лишнее слово — «кадр»: это существительное, а остальные — прилагательные.
- `odd` — Q: найди лишнее слово: отечественный, составляться, крупный, необходимо → A: Лишнее слово — «составляться»: это глагол, а остальные — прилагательные.
- `meaning` — Q: объясни слово «исполниться» → A: «исполниться» — осуществиться, претвориться в жизнь.

**en**
- `fill` — Q: complete the sentence (neighborhood, sprint, determined): We have just moved to a pleasant ___. → A: We have just moved to a pleasant neighborhood.
- `odd` — Q: which word does not belong: nanna, nug, respectful, leman? → A: The odd word is "respectful": it is an adjective, the others are nouns.
- `odd` — Q: which word does not belong: avoiding, completely, seemingly, surely? → A: The odd word is "avoiding": it is a noun, the others are adverbs.
- `odd` — Q: odd one out: discover, coaming, sprint, imitate → A: The odd word is "coaming": it is a noun, the others are verbs.
- `synonym` — Q: what word means the same as "shared"? → A: A synonym for "shared" is "common".
