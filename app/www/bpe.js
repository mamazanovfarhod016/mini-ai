/*
 * Mini AI v0.3 — BPE-токенизатор на JS, ТОЧНАЯ копия mini_ai/bpe.py.
 * encode/decode должны давать идентичные id, что и Python (см. tests/test_bpe.py,
 * tests/js_bpe_encode.js).
 */
(function (root) {
  "use strict";

  var SPECIALS = ["<pad>", "<bos>", "<sep>", "<eos>", "<unk>"];
  var PAD = 0, BOS = 1, SEP = 2, EOS = 3, UNK = 4;
  var WORD_PREFIX = "\u2581";
  var TIME_TOK = "\u2460";
  var DATE_TOK = "\u2461";
  var NO_MERGE = {};
  NO_MERGE[TIME_TOK] = true;
  NO_MERGE[DATE_TOK] = true;

  function wordToSymbols(word) {
    if (!word) return [];
    var chars = Array.from(word);
    chars[0] = WORD_PREFIX + chars[0];
    return chars;
  }

  function splitWords(text) {
    return text ? text.split(" ") : [];
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
  }

  BPETokenizer.prototype._bpeWord = function (symbols) {
    symbols = symbols.slice();
    while (symbols.length > 1) {
      var bestRank = null, bestI = -1;
      for (var i = 0; i < symbols.length - 1; i++) {
        var a = symbols[i], b = symbols[i + 1];
        if (NO_MERGE[a] || NO_MERGE[b]) continue;
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

  BPETokenizer.prototype.encode = function (text) {
    var ids = [];
    var words = splitWords(text);
    for (var w = 0; w < words.length; w++) {
      if (!words[w]) continue;
      var syms = this._bpeWord(wordToSymbols(words[w]));
      for (var i = 0; i < syms.length; i++) {
        var id = this.stoi.hasOwnProperty(syms[i]) ? this.stoi[syms[i]] : UNK;
        ids.push(id);
      }
    }
    return ids;
  };

  BPETokenizer.prototype.decode = function (ids) {
    var out = [];
    for (var i = 0; i < ids.length; i++) {
      var id = ids[i];
      if (id < SPECIALS.length) continue;
      if (id >= this.itos.length) continue;
      out.push(this.itos[id]);
    }
    var text = out.join("");
    text = text.split(WORD_PREFIX).join(" ");
    return text.replace(/^ +/, "");
  };

  BPETokenizer.prototype.promptIds = function (question, cleanFn, maxQTok) {
    maxQTok = maxQTok || 24;
    var q = this.encode(cleanFn(question)).slice(0, maxQTok);
    return [BOS].concat(q, [SEP]);
  };

  var MiniAIBPE = {
    SPECIALS: SPECIALS, PAD: PAD, BOS: BOS, SEP: SEP, EOS: EOS, UNK: UNK,
    WORD_PREFIX: WORD_PREFIX, TIME_TOK: TIME_TOK, DATE_TOK: DATE_TOK,
    BPETokenizer: BPETokenizer,
  };

  if (typeof module !== "undefined" && module.exports) {
    module.exports = MiniAIBPE;
  } else {
    root.MiniAIBPE = MiniAIBPE;
  }
})(typeof window !== "undefined" ? window : this);
