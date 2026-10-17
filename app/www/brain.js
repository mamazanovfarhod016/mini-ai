/*
 * Mini AI — «мозг» на JavaScript. Работает полностью офлайн в телефоне.
 * Это тот же самый прямой проход, что и в mini_ai/brain.py (Python):
 *   текст → признаки (слова + триграммы) → хеш FNV-1a → вектор x (2048)
 *   h = ReLU(x·W1 + b1)  →  p = softmax(h·W2 + b2)
 * Веса берутся из model.js (window.MINI_AI_MODEL), который генерирует train.py.
 * Тест tests/test_parity.py проверяет, что Python и JS отвечают одинаково.
 */
(function (root) {
  "use strict";

  var APOS = /['`’‘ʻʼ´]/g;
  var NON_LETTERS = /[^a-z0-9а-я]+/g;

  function normalize(text) {
    var t = String(text).toLowerCase().replace(/ё/g, "е").replace(APOS, "");
    return t.replace(NON_LETTERS, " ").trim().split(/\s+/).filter(Boolean).join(" ");
  }

  function tokens(text) {
    var words = normalize(text).split(" ").filter(Boolean);
    var feats = [];
    words.forEach(function (w) {
      feats.push("w:" + w);
      if (w.length > 4) feats.push("p:" + w.slice(0, 4));
      var padded = " " + w + " ";
      for (var i = 0; i < padded.length - 2; i++) feats.push("c:" + padded.slice(i, i + 3));
    });
    return feats;
  }

  // 32-битный FNV-1a: Math.imul + >>> 0 дают точно такое же число, как Python & 0xFFFFFFFF
  function fnv1a(s) {
    var h = 2166136261;
    for (var i = 0; i < s.length; i++) {
      h ^= s.charCodeAt(i);
      h = Math.imul(h, 16777619) >>> 0;
    }
    return h >>> 0;
  }

  function isLatin(text) {
    var t = normalize(text), lat = 0, cyr = 0;
    for (var i = 0; i < t.length; i++) {
      var c = t[i];
      if (c >= "a" && c <= "z") lat++;
      else if (c >= "а" && c <= "я") cyr++;
    }
    return lat > cyr;
  }

  function Brain(model) {
    var m = model;
    this.m = m;
    this.dim = m.dim; this.H = m.hidden; this.K = m.intents.length;
    var S = m.scale;
    // веса в файле — целые (w*scale), переводим обратно в дроби
    this.W1 = Float64Array.from(m.W1, function (v) { return v / S; });
    this.b1 = Float64Array.from(m.b1, function (v) { return v / S; });
    this.W2 = Float64Array.from(m.W2, function (v) { return v / S; });
    this.b2 = Float64Array.from(m.b2, function (v) { return v / S; });
    this.known = new Set(m.known);
  }

  Brain.prototype.think = function (text) {
    var D = this.dim, H = this.H, K = this.K;
    var feats = tokens(text);
    var self = this, known = 0;

    // 1) вектор признаков (разреженный: запоминаем только ненулевые ячейки)
    var x = new Map();
    feats.forEach(function (f) {
      var h = fnv1a(f);
      if (self.known.has(h)) known++;
      var j = h % D;
      x.set(j, (x.get(j) || 0) + 1);
    });
    var norm = 0;
    x.forEach(function (v) { norm += v * v; });
    norm = Math.sqrt(norm);

    // 2) скрытый слой: h = ReLU(x·W1 + b1)
    var hid = new Float64Array(H);
    for (var k = 0; k < H; k++) hid[k] = 0;
    var idx = Array.from(x.keys()).sort(function (a, b) { return a - b; });
    idx.forEach(function (j) {
      var xv = x.get(j) / norm, row = j * H;
      for (var k2 = 0; k2 < H; k2++) hid[k2] += xv * self.W1[row + k2];
    });
    for (var a = 0; a < H; a++) { hid[a] += this.b1[a]; if (hid[a] < 0) hid[a] = 0; }

    // 3) выходной слой: logits = h·W2 + b2, потом softmax
    var logits = new Float64Array(K);
    for (var c = 0; c < K; c++) {
      var s = this.b2[c];
      for (var r = 0; r < H; r++) s += hid[r] * this.W2[r * K + c];
      logits[c] = s;
    }
    var mx = -Infinity;
    for (c = 0; c < K; c++) if (logits[c] > mx) mx = logits[c];
    var sum = 0, p = new Float64Array(K);
    for (c = 0; c < K; c++) { p[c] = Math.exp(logits[c] - mx); sum += p[c]; }
    for (c = 0; c < K; c++) p[c] /= sum;

    var order = Array.from({ length: K }, function (_, i) { return i; })
      .sort(function (a2, b2) { return p[b2] - p[a2]; });
    var top = order.slice(0, 3).map(function (i) { return [self.m.intents[i], p[i]]; });
    var familiar = feats.length ? known / feats.length : 0;
    var intent = top[0][0];
    if (familiar < 0.34 || top[0][1] < this.m.threshold) intent = null;
    return { intent: intent, confidence: top[0][1], top3: top, familiar: familiar,
             features: feats.length, hidden: Array.from(hid) };
  };

  var UNKNOWN = {
    ru: "Я пока маленькая нейронка и этого не знаю 🙈 Попробуй что-то попроще: «привет», «как дела», «кто ты».",
    uz: "Men hali kichkina neyron tarmoqman va buni bilmayman 🙈 Oddiyroq so'rang: «salom», «qalaysan»."
  };

  function pad(n) { return (n < 10 ? "0" : "") + n; }

  Brain.prototype.reply = function (text, rnd, now) {
    rnd = rnd || Math.random;
    now = now || new Date();   // время и дата — с часов телефона, не из интернета
    var t = this.think(text);
    var lang = isLatin(text) ? "uz" : "ru";
    if (!t.intent) return { text: UNKNOWN[lang], thought: t, lang: lang };
    var opts = this.m.responses[t.intent][lang];
    var ans = opts[Math.floor(rnd() * opts.length)]
      .replace("{time}", pad(now.getHours()) + ":" + pad(now.getMinutes()))
      .replace("{date}", pad(now.getDate()) + "." + pad(now.getMonth() + 1) + "." + now.getFullYear());
    return { text: ans, thought: t, lang: lang };
  };

  var api = { Brain: Brain, tokens: tokens, normalize: normalize, fnv1a: fnv1a, isLatin: isLatin };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.MiniBrain = api;
})(typeof window !== "undefined" ? window : this);
