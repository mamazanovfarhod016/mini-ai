/*
 * Mini AI v0.3 — BPE-токенизатор на JS, ТОЧНАЯ копия mini_ai/bpe.py.
 * encode/decode должны давать идентичные id, что и Python (см. tests/test_bpe.py,
 * tests/js_bpe_encode.js).
 *
 * v0.4: добавлен lossless raw-режим (encode(text, true) / decode(ids, true)) —
 * используется для кода и длинных текстов. Старое поведение (без raw) НЕ менялось,
 * id идентичны старым моделям.
 */
(function (root) {
  "use strict";

  var SPECIALS = ["<pad>", "<bos>", "<sep>", "<eos>", "<unk>"];
  var PAD = 0, BOS = 1, SEP = 2, EOS = 3, UNK = 4;
  var WORD_PREFIX = "\u2581";
  var TIME_TOK = "\u2460";
  var DATE_TOK = "\u2461";
  var NEWLINE_TOK = "\n";
  var TAB_TOK = "\t";
  var NO_MERGE = {};
  NO_MERGE[TIME_TOK] = true;
  NO_MERGE[DATE_TOK] = true;
  NO_MERGE[NEWLINE_TOK] = true;

  // как _atomic в bpe.py: ①/②/'\n' и любые цифры не сливаются (каждая цифра — свой токен)
  function atomic(sym) { return !!NO_MERGE[sym] || /[0-9]/.test(sym); }

  function wordToSymbols(word) {
    if (!word) return [];
    var chars = Array.from(word);
    chars[0] = WORD_PREFIX + chars[0];
    return chars;
  }

  function splitWords(text) {
    return text ? text.split(" ") : [];
  }

  function richTokenizeLine(line) {
    // пробел -> '▁' (один символ на пробел), таб -> '\t', остальное — как есть
    if (!line) return [];
    var chars = Array.from(line);
    var out = [];
    for (var i = 0; i < chars.length; i++) {
      var c = chars[i];
      if (c === " ") out.push(WORD_PREFIX);
      else if (c === "\t") out.push(TAB_TOK);
      else out.push(c);
    }
    return out;
  }

  function byteToken(b) {
    var hex = b.toString(16).toUpperCase();
    if (hex.length < 2) hex = "0" + hex;
    return "<0x" + hex + ">";
  }

  var BYTE_RUN_RE = /(?:<0x[0-9A-Fa-f]{2}>)+/g;
  var BYTE_TOK_RE = /<0x([0-9A-Fa-f]{2})>/g;

  function decodeByteRuns(text) {
    if (text.indexOf("<0x") === -1) return text;
    return text.replace(BYTE_RUN_RE, function (run) {
      var bytes = [];
      var m;
      BYTE_TOK_RE.lastIndex = 0;
      while ((m = BYTE_TOK_RE.exec(run)) !== null) {
        bytes.push(parseInt(m[1], 16));
      }
      return Buffer.from(bytes).toString("utf8");
    });
  }

  function BPETokenizer(dict) {
    this.itos = dict.vocab.slice();
    this.stoi = {};
    for (var i = 0; i < this.itos.length; i++) this.stoi[this.itos[i]] = i;
    this.merges = dict.merges.map(function (m) { return [m[0], m[1]]; });
    this.mergeRank = {};
    for (var r = 0; r < this.merges.length; r++) {
      this.mergeRank[this.merges[r][0] + "\u0001" + this.merges[r][1]] = r;
    }
    this._wordCache = {};
    this._lineCache = {};
  }

  BPETokenizer.prototype._addSymbol = function (s) {
    if (!this.stoi.hasOwnProperty(s)) {
      this.stoi[s] = this.itos.length;
      this.itos.push(s);
    }
  };

  BPETokenizer.prototype._bpeMerge = function (symbols) {
    symbols = symbols.slice();
    while (symbols.length > 1) {
      var bestRank = null, bestI = -1;
      for (var i = 0; i < symbols.length - 1; i++) {
        var a = symbols[i], b = symbols[i + 1];
        if (atomic(a) || atomic(b)) continue;
        var key = a + "\u0001" + b;
        var r = this.mergeRank.hasOwnProperty(key) ? this.mergeRank[key] : null;
        if (r !== null && (bestRank === null || r < bestRank)) {
          bestRank = r; bestI = i;
        }
      }
      if (bestI === -1) break;
      var merged = symbols[bestI] + symbols[bestI + 1];
      symbols.splice(bestI, 2, merged);
    }
    return symbols;
  };

  // старое имя (оставлено для обратной совместимости вызовов)
  BPETokenizer.prototype._bpeWord = function (symbols) { return this._bpeMerge(symbols); };

  BPETokenizer.prototype._byteFallbackIds = function (ch) {
    var bytes = Buffer.from(ch, "utf8");
    var ids = [];
    for (var i = 0; i < bytes.length; i++) {
      var tok = byteToken(bytes[i]);
      this._addSymbol(tok);
      ids.push(this.stoi[tok]);
    }
    return ids;
  };

  BPETokenizer.prototype.encode = function (text, raw) {
    return raw ? this._encodeRich(text) : this._encodeSimple(text);
  };

  BPETokenizer.prototype._encodeSimple = function (text) {
    var ids = [];
    var words = splitWords(text);
    for (var w = 0; w < words.length; w++) {
      if (!words[w]) continue;
      var syms = this._wordCache.hasOwnProperty(words[w]) ? this._wordCache[words[w]] : null;
      if (syms === null) {
        syms = this._bpeMerge(wordToSymbols(words[w]));
        this._wordCache[words[w]] = syms;
      }
      for (var i = 0; i < syms.length; i++) {
        var id = this.stoi.hasOwnProperty(syms[i]) ? this.stoi[syms[i]] : UNK;
        ids.push(id);
      }
    }
    return ids;
  };

  BPETokenizer.prototype._encodeRich = function (text) {
    var ids = [];
    var lines = String(text).split("\n");
    for (var li = 0; li < lines.length; li++) {
      if (li > 0) {
        ids.push(this.stoi.hasOwnProperty(NEWLINE_TOK) ? this.stoi[NEWLINE_TOK] : UNK);
      }
      var line = lines[li];
      if (line === "") continue;
      var merged = this._lineCache.hasOwnProperty(line) ? this._lineCache[line] : null;
      if (merged === null) {
        merged = this._bpeMerge(richTokenizeLine(line));
        this._lineCache[line] = merged;
      }
      for (var i = 0; i < merged.length; i++) {
        var s = merged[i];
        if (this.stoi.hasOwnProperty(s)) {
          ids.push(this.stoi[s]);
          continue;
        }
        var chars = Array.from(s);
        for (var j = 0; j < chars.length; j++) {
          var ch = chars[j];
          if (this.stoi.hasOwnProperty(ch)) {
            ids.push(this.stoi[ch]);
          } else {
            var fb = this._byteFallbackIds(ch);
            for (var k = 0; k < fb.length; k++) ids.push(fb[k]);
          }
        }
      }
    }
    return ids;
  };

  BPETokenizer.prototype.decode = function (ids, raw) {
    var out = [];
    for (var i = 0; i < ids.length; i++) {
      var id = ids[i];
      if (id < SPECIALS.length) continue;
      if (id >= this.itos.length) continue;
      out.push(this.itos[id]);
    }
    var text = out.join("");
    text = decodeByteRuns(text);
    text = text.split(WORD_PREFIX).join(" ");
    if (raw) return text;
    return text.replace(/^ +/, "");
  };

  BPETokenizer.prototype.promptIds = function (question, cleanFn, maxQTok, raw) {
    maxQTok = maxQTok || 24;
    var q = raw ? this.encode(question, true).slice(0, maxQTok)
                : this.encode(cleanFn(question)).slice(0, maxQTok);
    return [BOS].concat(q, [SEP]);
  };

  var MiniAIBPE = {
    SPECIALS: SPECIALS, PAD: PAD, BOS: BOS, SEP: SEP, EOS: EOS, UNK: UNK,
    WORD_PREFIX: WORD_PREFIX, TIME_TOK: TIME_TOK, DATE_TOK: DATE_TOK,
    NEWLINE_TOK: NEWLINE_TOK, TAB_TOK: TAB_TOK,
    BPETokenizer: BPETokenizer,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = MiniAIBPE;
  } else {
    root.MiniAIBPE = MiniAIBPE;
  }
})(typeof window !== "undefined" ? window : this);
