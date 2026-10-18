"""
Mini AI v0.3 — свой BPE-токенизатор (byte-pair encoding) С НУЛЯ на Python.

Идея (как в SentencePiece/GPT-2, но вручную, без библиотек):
    1. Берём корпус текста, каждое СЛОВО разбиваем по пробелам.
    2. Каждое слово представляем как «▁слово» (▁ = префикс начала слова —
       нужен, чтобы при decode() понимать, где был пробел, ведь мы НЕ
       храним пробелы как отдельный символ внутри слова).
    3. Начальный словарь = все отдельные символы, что встретились.
    4. Повторяем: ищем самую частую пару СОСЕДНИХ токенов внутри слов →
       сливаем её в один новый токен → запоминаем это слияние (merge) по
       порядку (ранг = порядковый номер появления). Останавливаемся, когда
       словарь достиг нужного размера VOCAB.
    5. encode(): текст → слова → "▁"+слово → по символам → затем ПРИМЕНЯЕМ
       все merges в порядке рангов (жадно — всегда сливаем пару с
       наименьшим рангом, пока такие пары есть).

Спецтокены SPECIALS всегда первые (id 0..4), ①/② (TIME_TOK/DATE_TOK) —
отдельные неделимые токены (никогда не участвуют в слияниях).

Выбор насчёт неизвестных символов: используем токен <unk> (5-й спецтокен,
id=4) — если символ не входит в словарь, он кодируется как <unk>, а не
пропускается молча (так потеря информации заметна в decode, а не тиха).

============================================================================
v0.4 — lossless-режим «raw» для кода и длинных текстов (ветка feat/bpe-code)
============================================================================
Старый режим encode()/decode() (raw=False, по умолчанию) НЕ ТРОГАЕМ: он
разбивает по одиночному пробелу, теряет множественные пробелы/отступы/
переносы строк — но зато даёт РОВНО ТЕ ЖЕ id, что и раньше (старые модели
model_transformer_math.json / model_transformer_school.json продолжают
работать без переобучения).

Для кода/длинных текстов добавлен ВТОРОЙ, lossless-режим raw=True:
    • каждый '\\n' — отдельный неделимый токен (NEWLINE_TOK);
    • каждый пробел — отдельный токен WORD_PREFIX ('▁'), то есть N пробелов
      подряд = N токенов '▁' (после BPE они обычно сливаются в один токен
      «отступ», например '▁▁▁▁' для 4 пробелов — но ни один пробел не
      теряется, т.к. это просто слияние одинаковых символов);
    • каждый '\\t' — отдельный токен TAB_TOK;
    • все остальные символы — как есть, БЕЗ префикса (в отличие от старого
      режима, где первый символ слова получал префикс '▁').
Получаем decode(encode(s, raw=True), raw=True) == s для абсолютно любого s
(код, Cyrillic, узбекские апострофы, эмодзи) — включая случаи, когда
встречается символ, которого не было при обучении: он уходит в байтовый
fallback (токены вида '<0xXX>' — UTF-8 байты), а не в <unk>, поэтому текст
никогда тихо не портится.
"""
import json
import re

SPECIALS = ["<pad>", "<bos>", "<sep>", "<eos>", "<unk>"]
PAD, BOS, SEP, EOS, UNK = 0, 1, 2, 3, 4
WORD_PREFIX = "\u2581"      # ▁ — префикс начала слова (как в SentencePiece)
TIME_TOK = "\u2460"          # ①
DATE_TOK = "\u2461"          # ②
NEWLINE_TOK = "\n"           # атомарный токен переноса строки (raw-режим)
TAB_TOK = "\t"               # атомарный токен табуляции (raw-режим)
NO_MERGE = {TIME_TOK, DATE_TOK, NEWLINE_TOK}   # эти символы никогда не сливаются с соседями

_BYTE_TOK_RE = re.compile(r"<0x([0-9A-Fa-f]{2})>")
_BYTE_RUN_RE = re.compile(r"(?:<0x[0-9A-Fa-f]{2}>)+")


def _atomic(sym):
    """Неделимый кусочек: ①/②/'\\n' и отдельные ЦИФРЫ. Каждая цифра — отдельный токен, иначе «23» и «45»
    были бы для модели целыми «словами» и она не увидела бы разряды (нужно для счёта, 18.10).
    ВАЖНО для скорости: атомарные символы НИКОГДА не участвуют в merge (см. _run_bpe/_bpe_merge),
    поэтому составной токен (len > 1) гарантированно не атомарен — проверяем это быстро, без
    посимвольного сканирования длинных строк (критично для скорости encode() на мегабайтах текста)."""
    if len(sym) != 1:
        return False
    return sym in NO_MERGE or ("0" <= sym <= "9")


def _word_to_symbols(word):
    """«привет» → ['▁п','р','и','в','е','т'] — первый символ слова с префиксом ▁.

    ①/② внутри слова выделяем как отдельный неделимый символ (без ▁-склейки
    с соседними буквами), т.к. это спец-сентинелы времени/даты.
    """
    if not word:
        return []
    chars = list(word)
    chars[0] = WORD_PREFIX + chars[0]
    return chars


def _split_words(text):
    """Текст → список слов (разбиение по пробелам, пробелы не хранятся). СТАРЫЙ режим."""
    return text.split(" ") if text else []


def _rich_tokenize_line(line):
    """Одна строка (без '\\n') → базовые символы lossless raw-режима:
    пробел → '▁' (один символ = один пробел, подряд — несколько токенов '▁'),
    таб → '\\t', всё остальное — как есть, по кодовым точкам (эмодзи/суррогатные
    пары Python хранит одним elements в str, так что иерархия корректна)."""
    if not line:
        return []
    out = []
    for c in line:
        if c == " ":
            out.append(WORD_PREFIX)
        elif c == "\t":
            out.append(TAB_TOK)
        else:
            out.append(c)
    return out


def _byte_token(b):
    return f"<0x{b:02X}>"


def _decode_byte_runs(text):
    """Склеивает соседние '<0xXX>' токены в байты и декодирует их как UTF-8
    (один символ может занимать несколько байт — поэтому группируем ПОДРЯД идущие)."""
    if "<0x" not in text:
        return text

    def repl(m):
        bs = bytes(int(h, 16) for h in _BYTE_TOK_RE.findall(m.group(0)))
        return bs.decode("utf-8", errors="replace")

    return _BYTE_RUN_RE.sub(repl, text)


def _run_bpe(corpus, add_symbol, add_merge, target_len, len_fn, min_count=2):
    """Эффективный жадный BPE-цикл с ИНКРЕМЕНТАЛЬНЫМ учётом пар (без полного
    пересчёта всех пар на каждой итерации — иначе extend() с extra=несколько
    тысяч на корпусе из тысяч строк работает минуты/часы).

    corpus: список [list(symbols), count] (изменяется in-place).
    add_symbol(tok): добавить новый токен в словарь (если его ещё нет).
    add_merge(a, b): зафиксировать merge (a, b) -> a+b (ранг = порядок вызовов).
    len_fn(): текущий размер словаря.
    target_len: останавливаемся, когда len_fn() >= target_len.
    min_count: минимальная частота пары, чтобы её взять.
    """
    pair_count = {}
    pair_words = {}

    def _word_pairs(syms):
        for i in range(len(syms) - 1):
            a, b = syms[i], syms[i + 1]
            if not (_atomic(a) or _atomic(b)):
                yield (a, b)

    for idx, (syms, c) in enumerate(corpus):
        for p in _word_pairs(syms):
            pair_count[p] = pair_count.get(p, 0) + c
            pair_words.setdefault(p, set()).add(idx)

    while len_fn() < target_len:
        if not pair_count:
            break
        best = max(pair_count, key=lambda k: (pair_count[k], k))
        if pair_count[best] < min_count:
            break
        a, b = best
        new_tok = a + b
        add_merge(a, b)
        add_symbol(new_tok)
        affected = pair_words.pop(best, set())
        del pair_count[best]
        for idx in affected:
            syms, c = corpus[idx]
            old_pairs = list(_word_pairs(syms))
            for p in old_pairs:
                if p in pair_count:
                    pair_count[p] -= c
                    if pair_count[p] <= 0:
                        del pair_count[p]
                        pair_words.pop(p, None)
                    else:
                        pair_words.get(p, set()).discard(idx)
            i = 0
            while i < len(syms) - 1:
                if syms[i] == a and syms[i + 1] == b:
                    syms[i:i + 2] = [new_tok]
                else:
                    i += 1
            for p in _word_pairs(syms):
                pair_count[p] = pair_count.get(p, 0) + c
                pair_words.setdefault(p, set()).add(idx)
    return corpus


class BPETokenizer:
    """Токенизатор на базе byte-pair encoding, обучаемый на собственном корпусе."""

    def __init__(self):
        self.itos = list(SPECIALS)
        self.stoi = {c: i for i, c in enumerate(self.itos)}
        self.merges = []            # список [a, b] в порядке обучения (ранг = индекс)
        self.merge_rank = {}        # (a, b) -> ранг (чем меньше, тем сливаем раньше)

    def __len__(self):
        return len(self.itos)

    # ------------------------------------------------------------------
    # Обучение
    # ------------------------------------------------------------------
    def train(self, texts, vocab_size=1000, verbose=False):
        """texts: список строк корпуса (уже очищенных/как есть — склеиваем все слова)."""
        # 1) разбиваем весь корпус на слова -> на символы-с-префиксом
        words = []
        for t in texts:
            for w in _split_words(t):
                if w:
                    words.append(_word_to_symbols(w))

        # 2) начальный словарь: спецтокены + все встретившиеся символы (кроме спецтокенов)
        base_symbols = set()
        for w in words:
            base_symbols.update(w)
        base_symbols -= set(self.itos)
        for s in sorted(base_symbols):
            self._add_symbol(s)
        # всегда гарантируем базовые атомарные символы raw-режима (нужны для
        # lossless-кодирования кода/длинных текстов даже если их не было в корпусе)
        self._add_symbol(NEWLINE_TOK)
        self._add_symbol(TAB_TOK)
        self._add_symbol(WORD_PREFIX)

        # 3) считаем частоты слов (word -> count), чтобы не пересчитывать каждый раз одно и то же
        from collections import Counter
        word_counts = Counter(tuple(w) for w in words)
        # представим как список [list(symbols), count] для изменяемости
        corpus = [[list(w), c] for w, c in word_counts.items()]

        def _add_merge(a, b):
            rank = len(self.merges)
            self.merges.append([a, b])
            self.merge_rank[(a, b)] = rank
            if verbose and rank % 50 == 0:
                print(f"  merge #{rank}: {a!r}+{b!r} vocab={len(self.itos)}")

        _run_bpe(corpus, self._add_symbol, _add_merge, vocab_size, lambda: len(self.itos), min_count=2)
        return self

    def extend(self, texts, extra=200, min_count=20, verbose=False):
        """ПОСТОЯННЫЕ ЗНАНИЯ: дополнить словарь, НЕ меняя старые номера токенов.
        Новые буквы (например, английская «w») и новые слияния добавляются В КОНЕЦ —
        старые id значат то же самое, поэтому выученные веса сети остаются верными.

        Учимся СРАЗУ на двух представлениях корпуса, чтобы одни и те же merges
        (например 'def', 'return', '    ' — 4 пробела отступа) были доступны и
        старому word-режиму (encode(raw=False)), и новому lossless raw-режиму
        (encode(raw=True), код/длинные тексты с отступами и переносами строк):
          1. word_lists  — как раньше: текст -> слова по пробелу -> '▁'+буквы;
          2. line_lists  — raw: текст -> строки по '\\n' -> пробелы/табы как
             свои атомарные символы '▁'/'\\t' (без '▁'-склейки с первой буквой).

        min_count: порог частоты пары, ниже которого словарь не раздуваем.
        Возвращает число добавленных токенов."""
        n0 = len(self.itos)
        # гарантируем, что базовые raw-символы всегда есть (у очень старых
        # сохранённых токенизаторов их могло не быть)
        self._add_symbol(NEWLINE_TOK)
        self._add_symbol(TAB_TOK)
        self._add_symbol(WORD_PREFIX)

        word_lists = [self._bpe_word(_word_to_symbols(w)) for t in texts for w in _split_words(t) if w]
        line_lists = []
        for t in texts:
            for line in str(t).split("\n"):
                if line == "":
                    continue
                line_lists.append(self._bpe_merge(_rich_tokenize_line(line)))

        all_lists = word_lists + line_lists
        for w in all_lists:                               # новые буквы — отдельными токенами
            for sym in w:
                if sym not in self.stoi and len(sym) <= 2:
                    self._add_symbol(sym)

        from collections import Counter
        corpus = [[list(w), c] for w, c in Counter(tuple(w) for w in all_lists).items()]
        target = len(self.itos) + extra

        def _add_merge(a, b):
            self.merge_rank[(a, b)] = len(self.merges)
            self.merges.append([a, b])

        _run_bpe(corpus, self._add_symbol, _add_merge, target, lambda: len(self.itos), min_count=min_count)
        if verbose:
            print(f"Словарь дополнен: {n0} → {len(self.itos)} токенов (старые номера не изменились)")
        return len(self.itos) - n0

    def _add_symbol(self, s):
        if s not in self.stoi:
            self.stoi[s] = len(self.itos)
            self.itos.append(s)

    # ------------------------------------------------------------------
    # Кодирование / декодирование
    # ------------------------------------------------------------------
    def _bpe_merge(self, symbols):
        """Жадно применяем merges в порядке рангов к списку символов (слово ИЛИ строка).
        Эффективная версия с мин-кучей по рангу пары вместо полного пересканирования
        списка на каждом шаге (критично для скорости encode() длинных строк кода)."""
        symbols = list(symbols)
        n = len(symbols)
        if n <= 1:
            return symbols
        merge_rank = self.merge_rank
        atomic = _atomic
        # doubly-linked list через индексы + heap кандидатов (rank, pos, a, b)
        import heapq
        nxt = list(range(1, n)) + [None]
        prv = [None] + list(range(0, n - 1))
        alive = [True] * n
        heap = []
        for i in range(n - 1):
            a, b = symbols[i], symbols[i + 1]
            if not (atomic(a) or atomic(b)):
                r = merge_rank.get((a, b))
                if r is not None:
                    heapq.heappush(heap, (r, i))
        while heap:
            r, i = heapq.heappop(heap)
            if not alive[i]:
                continue
            j = nxt[i]
            if j is None or not alive[j]:
                continue
            a, b = symbols[i], symbols[j]
            if merge_rank.get((a, b)) != r:
                continue  # устарело (символы на i/j изменились после другого слияния)
            new_sym = a + b
            symbols[i] = new_sym
            alive[j] = False
            k = nxt[j]
            nxt[i] = k
            if k is not None:
                prv[k] = i
            # пересчитываем пары слева и справа от нового токена
            p = prv[i]
            if p is not None and alive[p]:
                pa, pb = symbols[p], symbols[i]
                if not (atomic(pa) or atomic(pb)):
                    rr = merge_rank.get((pa, pb))
                    if rr is not None:
                        heapq.heappush(heap, (rr, p))
            if k is not None:
                na, nb = symbols[i], symbols[k]
                if not (atomic(na) or atomic(nb)):
                    rr = merge_rank.get((na, nb))
                    if rr is not None:
                        heapq.heappush(heap, (rr, i))
        out = []
        i = 0
        while i is not None:
            if alive[i]:
                out.append(symbols[i])
            i = nxt[i]
        return out

    def _bpe_word(self, symbols):
        """Старое имя — алиас _bpe_merge (используется в тестах и train_transformer.py)."""
        return self._bpe_merge(symbols)

    def _byte_fallback_ids(self, ch, cache=None):
        """Символ, которого нет в словаре целиком и посимвольно -> его UTF-8 байты
        как атомарные токены '<0xXX>' (а не тихий <unk>). Токены добавляются в словарь
        динамически (В КОНЕЦ — не меняет старые id)."""
        ids = []
        for b in ch.encode("utf-8"):
            tok = _byte_token(b)
            self._add_symbol(tok)
            ids.append(self.stoi[tok])
        return ids

    def _word_cache(self):
        if not hasattr(self, "_word_cache_dict"):
            self._word_cache_dict = {}
        return self._word_cache_dict

    def _line_cache(self):
        if not hasattr(self, "_line_cache_dict"):
            self._line_cache_dict = {}
        return self._line_cache_dict

    def encode(self, text, raw=False):
        """Текст -> список id.
        raw=False (по умолчанию, СТАРОЕ поведение): разбиение по одиночному
            пробелу, пробелы/отступы/переносы строк не сохраняются ровно —
            но id идентичны старым моделям (без этого параметра совместимость
            со старыми сохранёнными весами была бы потеряна).
        raw=True (НОВОЕ, lossless): decode(encode(text, raw=True), raw=True) == text
            для произвольного текста — используем для кода и длинных текстов."""
        if raw:
            return self._encode_rich(text)
        return self._encode_simple(text)

    def _encode_simple(self, text):
        ids = []
        cache = self._word_cache()
        for w in _split_words(text):
            if not w:
                continue
            syms = cache.get(w)
            if syms is None:
                syms = self._bpe_word(_word_to_symbols(w))
                if len(cache) < 200000:          # не даём кэшу расти бесконечно
                    cache[w] = syms
            for s in syms:
                ids.append(self.stoi.get(s, UNK))
        return ids

    def _encode_rich(self, text):
        ids = []
        lines = str(text).split("\n")
        cache = self._line_cache()
        for li, line in enumerate(lines):
            if li > 0:
                ids.append(self.stoi.get(NEWLINE_TOK, UNK))
            if line == "":
                continue
            merged = cache.get(line)
            if merged is None:
                merged = self._bpe_merge(_rich_tokenize_line(line))
                if len(cache) < 200000:
                    cache[line] = merged
            for s in merged:
                tid = self.stoi.get(s)
                if tid is not None:
                    ids.append(tid)
                    continue
                # новый мульти-символьный токен целиком неизвестен — разбираем
                # по символам, а неизвестные символы уходят в байтовый fallback
                for ch in s:
                    cid = self.stoi.get(ch)
                    if cid is not None:
                        ids.append(cid)
                    else:
                        ids.extend(self._byte_fallback_ids(ch))
        return ids

    def decode(self, ids, raw=False):
        """id -> текст: склеиваем токены, ▁ -> пробел, спецтокены пропускаем.
        raw=True: НЕ убираем ведущие пробелы (lossless), raw=False — старое поведение."""
        out = []
        for i in ids:
            if i < len(SPECIALS):
                continue
            if i >= len(self.itos):
                continue
            out.append(self.itos[i])
        text = "".join(out)
        text = _decode_byte_runs(text)
        text = text.replace(WORD_PREFIX, " ")
        if raw:
            return text
        return text.lstrip(" ")

    def prompt(self, question, clean_fn, max_q_tok=24, raw=False):
        """<bos> + encode(question)[:max_q_tok] + <sep>.
        raw=False (старое): encode(clean_fn(question)) — как раньше.
        raw=True (новое): encode(question, raw=True) без clean_fn — для кода/длинных
        контекстов, где нижний регистр/пунктуация важны и их МЕНЯТЬ НЕЛЬЗЯ."""
        if raw:
            q = self.encode(question, raw=True)[:max_q_tok]
        else:
            q = self.encode(clean_fn(question))[:max_q_tok]
        return [BOS] + q + [SEP]

    # ------------------------------------------------------------------
    # Сохранение/загрузка
    # ------------------------------------------------------------------
    def to_dict(self):
        return {"type": "bpe", "vocab": self.itos, "merges": self.merges}

    @classmethod
    def from_dict(cls, d):
        tok = cls()
        tok.itos = list(d["vocab"])
        tok.stoi = {c: i for i, c in enumerate(tok.itos)}
        tok.merges = [list(m) for m in d["merges"]]
        tok.merge_rank = {(a, b): r for r, (a, b) in enumerate(tok.merges)}
        return tok


def prompt_ids(tok, text, raw=False, clean_fn=None, max_q_tok=24):
    """Удобная функция-обёртка (требование задачи): формирует <bos>..<sep> промпт.
    raw=False — старое поведение (clean_question + encode), raw=True — lossless
    (используем для Python-кода и длинных текстов, где регистр/пробелы важны)."""
    if clean_fn is None:
        from mini_ai.transformer import clean_question as clean_fn
    return tok.prompt(text, clean_fn, max_q_tok=max_q_tok, raw=raw)
