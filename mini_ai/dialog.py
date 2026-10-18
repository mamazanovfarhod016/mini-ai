"""
Диалоговый (multi-turn) формат для Mini AI — история чата прямо в промпте.

НЕ добавляем новых спец-токенов (словарь не меняется): используем обычные
текстовые маркеры "U: " (пользователь) / "A: " (бот) + "\n" между репликами —
они кодируются lossless-токенизатором (raw=True, mini_ai/bpe.py) как обычный
текст, поэтому любой язык (uz/ru/en) и даже сам текст "U:"/"A:" внутри реплики
не ломают формат (разделение только по структуре последовательности, не по
специальным id).

Последовательность одного тренировочного примера (один БОТ-ответ = один пример):
    <bos> + история(предыдущие реплики, каждая строка "U: ..."/"A: ...") +
    "\nU: " + последний вопрос пользователя + <sep> + ответ бота + <eos>
loss считается только на токенах ответа бота (как в qa/context).

История режется СЛЕВА целыми репликами (а не по токенам), чтобы не обрывать
реплику на середине — старые реплики выкидываются первыми, последняя реплика
пользователя остаётся ВСЕГДА.

Используется и в обучении (mini_ai/curriculum.py: source type "dialog"),
и в экзаменах (mini_ai/exams.py: exam_dialog), и зеркально в приложении
(app/www/transformer.js: promptIdsDialog — тот же алгоритм на JS).
"""
import glob as _glob
import json
import zlib
from pathlib import Path

from mini_ai.transformer import BOS, EOS, SEP

U_PREFIX = "U: "
A_PREFIX = "A: "


def turn_line(turn):
    """Одна реплика → строка "U: текст" / "A: текст" (role: "user"/"bot")."""
    prefix = U_PREFIX if turn.get("role") == "user" else A_PREFIX
    return prefix + str(turn.get("text", ""))


def build_prompt_text(history_turns, last_user_text):
    """history_turns — реплики СТРОГО ДО последнего вопроса пользователя (может быть пустым).
    last_user_text — текст последнего вопроса (его добавляем с "\nU: ", но без финального
    пробела — raw-токенизатор это сохранит буква в букву)."""
    lines = [turn_line(t) for t in history_turns]
    hist = "\n".join(lines)
    tail = "U: " + str(last_user_text)
    return (hist + "\n" + tail) if hist else tail


def is_dialog_held_out(dialog_id):
    """10% диалогов (по crc32(id) % 10 == 0) — ВСЕГДА held-out: никогда не используются
    при обучении, только в экзаменах (exam_dialog)."""
    return (zlib.crc32(str(dialog_id).encode("utf-8")) % 10) == 0


def _prior_and_last_user(turns, bot_idx):
    """turns[bot_idx] — реплика бота (role=="bot"). Возвращает (history_turns, last_user_text)
    где history_turns — все реплики ДО последнего вопроса пользователя, last_user_text — текст
    того последнего вопроса (turns[bot_idx-1], должен быть role=="user")."""
    prior = turns[:bot_idx]
    if not prior or prior[-1].get("role") != "user":
        raise ValueError("перед репликой бота должна быть реплика пользователя")
    return prior[:-1], prior[-1]["text"]


def encode_dialog_prompt_ids(tok, history_turns, last_user_text, encode_fn):
    """encode_fn(text) -> list[int] (raw=True lossless-кодирование). Возвращает ids ПРОМПТА
    (без <bos>/<sep>) — именно то, что приложение добавляет в <bos> ... <sep> перед генерацией."""
    text = build_prompt_text(history_turns, last_user_text)
    return encode_fn(text)


def dialog_example_to_seq(tok, turns, bot_idx, max_len, raw=True):
    """Один тренировочный пример: реплика бота turns[bot_idx] с историей до неё.
    Возвращает (seq, tgt, mask) как qa/context_example_to_seq.
    Труncate истории СЛЕВА целыми репликами, чтобы влезть в max_len; последний вопрос
    пользователя никогда не обрезается."""
    from mini_ai.curriculum import encode_answer, encode_prompt

    def enc(text):
        return encode_prompt(tok, text, raw=True)

    history_turns, last_user_text = _prior_and_last_user(turns, bot_idx)
    bot_ids = encode_answer(tok, turns[bot_idx]["text"])

    # дропаем историю с НАЧАЛА (старейшие реплики) целыми репликами, пока не влезет
    hist = list(history_turns)
    while True:
        prompt_ids = encode_dialog_prompt_ids(tok, hist, last_user_text, enc)
        total = 1 + len(prompt_ids) + 1 + len(bot_ids) + 1   # bos + prompt + sep + bot + eos
        if total <= max_len or not hist:
            break
        hist = hist[1:]

    seq = [BOS] + prompt_ids + [SEP] + bot_ids + [EOS]
    ans_start = 1 + len(prompt_ids) + 1
    if len(seq) > max_len:
        # история пуста, но даже один вопрос+ответ не влезает — режем С НАЧАЛА промпта
        # (как context_example_to_seq): конец (sep+ответ) — самое важное, его не трогаем
        overflow = len(seq) - max_len
        prompt_ids = prompt_ids[overflow:]
        seq = [BOS] + prompt_ids + [SEP] + bot_ids + [EOS]
        ans_start = max(1, ans_start - overflow)
    full_seq = seq[:-1]
    tgt = seq[1:]
    mask = [False] * len(full_seq)
    for t in range(ans_start - 1, len(full_seq)):
        mask[t] = True
    return full_seq, tgt, mask


def load_dialogs(files_or_glob):
    """files_or_glob: список путей ИЛИ строка-glob ("data/dialogs/*.jsonl"). Каждая строка
    файла — один JSON-диалог (см. формат в docstring curriculum.py / спеке задачи)."""
    if isinstance(files_or_glob, str):
        paths = sorted(_glob.glob(files_or_glob))
    else:
        paths = []
        for f in files_or_glob:
            paths.extend(sorted(_glob.glob(str(f))) if any(c in str(f) for c in "*?[") else [str(f)])
    dialogs = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                dialogs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return dialogs


def split_train_test_dialogs(dialogs):
    """10% held-out (crc32(id)%10==0) НИКОГДА не используется в обучении — только в exam_dialog."""
    train, test = [], []
    for d in dialogs:
        (test if is_dialog_held_out(d.get("id", "")) else train).append(d)
    return train, test


def dialog_examples_from_dialogs(tok, dialogs, max_len, raw=True):
    """Каждая реплика бота в диалоге → один тренировочный пример (растущая история)."""
    examples = []
    for d in dialogs:
        turns = d.get("turns", [])
        for i, t in enumerate(turns):
            if t.get("role") == "bot" and i > 0 and turns[i - 1].get("role") == "user":
                examples.append(dialog_example_to_seq(tok, turns, i, max_len, raw=raw))
    return examples


# ======================================================================================
# Простая эвристика языка/скрипта реплики — для exam_dialog (language consistency)
# ======================================================================================
_CYRILLIC = set("абвгдежзийклмнопрстуфхцчшщъыьэюяёАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯЁ")
_UZ_MARKERS = set("ʻʼ'ʼʻ'g'o'")  # апострофы uz (oʻ/gʻ) и характерные буквосочетания


def detect_script(text):
    """Грубая эвристика: cyrillic → "ru"; latin с uz-апострофами (oʻ/gʻ/o'/g') → "uz";
    latin без них → "en". Для коротких/пустых ответов — None (не считаем ошибкой)."""
    text = str(text)
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return None
    n_cyr = sum(1 for c in letters if c in _CYRILLIC)
    if n_cyr / len(letters) > 0.3:
        return "ru"
    if any(m in text for m in ("ʻ", "ʼ", "o'", "g'", "O'", "G'")):
        return "uz"
    return "en"
