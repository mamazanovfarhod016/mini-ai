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
"""
import json
import re

SPECIALS = ["<pad>", "<bos>", "<sep>", "<eos>", "<unk>"]
PAD, BOS, SEP, EOS, UNK = 0, 1, 2, 3, 4
WORD_PREFIX = "\u2581"      # ▁ — префикс начала слова (как в SentencePiece)
TIME_TOK = "\u2460"          # ①
DATE_TOK = "\u2461"          # ②
NO_MERGE = {TIME_TOK, DATE_TOK}   # эти символы никогда не сливаются с соседями


def _atomic(sym):
    """Неделимый кусочек: ①/② и любые ЦИФРЫ. Каждая цифра — отдельный токен, иначе «23» и «45»
    были бы для модели целыми «словами» и она не увидела бы разряды (нужно для счёта, 18.10)."""
    return sym in NO_MERGE or any("0" <= c <= "9" for c in sym)


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
    """Текст → список слов (разбиение по пробелам, пробелы не хранятся)."""
    return text.split(" ") if text else []


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

        # 3) считаем частоты слов (word -> count), чтобы не пересчитывать каждый раз одно и то же
        from collections import Counter
        word_counts = Counter(tuple(w) for w in words)
        # представим как список [list(symbols), count] для изменяемости
        corpus = [[list(w), c] for w, c in word_counts.items()]

        def get_pair_counts():
            counts = {}
            for syms, c in corpus:
                for i in range(len(syms) - 1):
                    a, b = syms[i], syms[i + 1]
                    if _atomic(a) or _atomic(b):
                        continue
                    counts[(a, b)] = counts.get((a, b), 0) + c
            return counts

        while len(self.itos) < vocab_size:
            pair_counts = get_pair_counts()
            if not pair_counts:
                break
            best_pair = max(pair_counts, key=lambda k: (pair_counts[k], k))
            if pair_counts[best_pair] < 2:
                break
            a, b = best_pair
            new_tok = a + b
            rank = len(self.merges)
            self.merges.append([a, b])
            self.merge_rank[(a, b)] = rank
            self._add_symbol(new_tok)
            # сливаем во всём корпусе
            for syms, _c in corpus:
                i = 0
                while i < len(syms) - 1:
                    if syms[i] == a and syms[i + 1] == b:
                        syms[i:i + 2] = [new_tok]
                    else:
                        i += 1
            if verbose and rank % 50 == 0:
                print(f"  merge #{rank}: {a!r}+{b!r} -> {new_tok!r} (freq={pair_counts[best_pair]}) vocab={len(self.itos)}")
        return self

    def extend(self, texts, extra=200, verbose=False):
        """ПОСТОЯННЫЕ ЗНАНИЯ: дополнить словарь, НЕ меняя старые номера токенов.
        Новые буквы (например, английская «w») и новые слияния добавляются В КОНЕЦ —
        старые id значат то же самое, поэтому выученные веса сети остаются верными.
        Возвращает число добавленных токенов."""
        n0 = len(self.itos)
        words = [self._bpe_word(_word_to_symbols(w)) for t in texts for w in _split_words(t) if w]
        for w in words:                                   # новые буквы — отдельными токенами
            for sym in w:
                if sym not in self.stoi and len(sym) <= 2:
                    self._add_symbol(sym)
        from collections import Counter
        corpus = [[list(w), c] for w, c in Counter(tuple(w) for w in words).items()]
        target = len(self.itos) + extra
        while len(self.itos) < target:                    # продолжаем BPE с того места, где остановились
            counts = {}
            for syms, c in corpus:
                for i in range(len(syms) - 1):
                    a, b = syms[i], syms[i + 1]
                    if not (_atomic(a) or _atomic(b)):
                        counts[(a, b)] = counts.get((a, b), 0) + c
            if not counts:
                break
            best = max(counts, key=lambda k: (counts[k], k))
            if counts[best] < 20:                          # редкие пары не берём — словарь не раздуваем
                break
            a, b = best
            self.merge_rank[(a, b)] = len(self.merges)
            self.merges.append([a, b])
            self._add_symbol(a + b)
            for syms, _c in corpus:
                i = 0
                while i < len(syms) - 1:
                    if syms[i] == a and syms[i + 1] == b:
                        syms[i:i + 2] = [a + b]
                    else:
                        i += 1
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
    def _bpe_word(self, symbols):
        """Жадно применяем merges в порядке рангов к списку символов одного слова."""
        symbols = list(symbols)
        while len(symbols) > 1:
            best_rank, best_i = None, None
            for i in range(len(symbols) - 1):
                a, b = symbols[i], symbols[i + 1]
                if _atomic(a) or _atomic(b):
                    continue
                r = self.merge_rank.get((a, b))
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i is None:
                break
            a, b = symbols[best_i], symbols[best_i + 1]
            symbols[best_i:best_i + 2] = [a + b]
        return symbols

    def encode(self, text):
        """Текст (уже очищенный) -> список id."""
        ids = []
        for w in _split_words(text):
            if not w:
                continue
            syms = self._bpe_word(_word_to_symbols(w))
            for s in syms:
                ids.append(self.stoi.get(s, UNK))
        return ids

    def decode(self, ids):
        """id -> текст: склеиваем токены, ▁ -> пробел, спецтокены пропускаем."""
        out = []
        for i in ids:
            if i < len(SPECIALS):
                continue
            if i >= len(self.itos):
                continue
            out.append(self.itos[i])
        text = "".join(out)
        text = text.replace(WORD_PREFIX, " ")
        return text.lstrip(" ")

    def prompt(self, question, clean_fn, max_q_tok=24):
        """<bos> + encode(clean_fn(question))[:max_q_tok] + <sep>."""
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
