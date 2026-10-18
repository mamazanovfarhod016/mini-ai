/*
 * Mini AI v0.2 — трансформер на JavaScript. Тот же самый KV-cache инференс,
 * что и mini_ai/transformer.py (класс KVModel), только без обучения — только
 * «чтение» уже готовых весов из model_transformer.js.
 *
 * Формулы (см. комментарии в transformer.py, здесь — то же самое кодом):
 *   x = wte[символ] + wpe[позиция]                         эмбеддинги
 *   h = LayerNorm(x)
 *   Q,K,V = h·Wqkv + bqkv (по головам)
 *   s = (q·kᵗ) / sqrt(dh)         softmax по всем k ДО текущей позиции (кэш)
 *   y = concat_головы(softmax(s)·V) · Wo + bo
 *   x = x + y                                              residual
 *   x = x + MLP(LayerNorm(x))                               residual
 *   logits = LayerNorm(x) · wteᵗ + bout
 *
 * Тест tests/js_transformer_parity.js сверяет logits и greedy-ответы с Python.
 */
(function (root) {
  "use strict";

  var SPECIALS = ["<pad>", "<bos>", "<sep>", "<eos>"];
  var PAD = 0, BOS = 1, SEP = 2, EOS = 3;
  var APOS = /['`’‘ʻʼ´]/g;

  function cleanQuestion(text) {
    var t = String(text).toLowerCase().replace(/ё/g, "е");
    t = t.replace(APOS, "'");
    // как в Python clean_question: знаки препинания → пробел (в обучающих вопросах их нет)
    t = normalizeMath(t);
    t = t.replace(/[!?.,;:()"«»…—–-]+/g, " ");
    return t.split(/\s+/).filter(Boolean).join(" ");
  }

  // как normalize_math в transformer.py: единая запись примеров «23 + 45», «8 * 3», «10 / 2», «7 − 3»
  function normalizeMath(t) {
    t = t.replace(/(\d)\s*[x×х*]\s*(?=\d)/g, "$1 * ");
    t = t.replace(/(\d)\s*[-−–]\s*(?=\d)/g, "$1 − ");
    t = t.replace(/(\d)\s*[:÷\/]\s*(?=\d)/g, "$1 / ");
    t = t.replace(/\s*([+=])\s*/g, " $1 ");
    return t;
  }

  // ---------- float16 (IEEE 754 half) → float64 ----------
  function f16ToFloat(h) {
    var s = (h & 0x8000) ? -1 : 1;
    var e = (h >> 10) & 0x1f;
    var f = h & 0x3ff;
    if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + f / 1024);
  }

  function base64ToF16Array(b64) {
    var bin = atob(b64);
    var n = bin.length / 2;
    var out = new Float64Array(n);
    for (var i = 0; i < n; i++) {
      var lo = bin.charCodeAt(2 * i), hi = bin.charCodeAt(2 * i + 1);
      out[i] = f16ToFloat(lo | (hi << 8));
    }
    return out;
  }

  function reshape(flat, shape) {
    // Возвращаем «сырой» Float64Array + shape; индексация делается вручную
    // (без вложенных JS-массивов — так быстрее и экономнее по памяти).
    return { data: flat, shape: shape };
  }

  // ---------- базовые операции ----------
  function layernorm(x, d, g, b, eps) {
    eps = eps || 1e-5;
    var mu = 0;
    for (var i = 0; i < d; i++) mu += x[i];
    mu /= d;
    var varr = 0;
    for (i = 0; i < d; i++) { var c = x[i] - mu; varr += c * c; }
    varr /= d;
    var rstd = 1 / Math.sqrt(varr + eps);
    var out = new Float64Array(d);
    for (i = 0; i < d; i++) out[i] = (x[i] - mu) * rstd * g[i] + b[i];
    return out;
  }

  var GELU_C = Math.sqrt(2 / Math.PI);
  function geluVec(x) {
    var out = new Float64Array(x.length);
    for (var i = 0; i < x.length; i++) {
      var v = x[i];
      var u = GELU_C * (v + 0.044715 * v * v * v);
      out[i] = 0.5 * v * (1 + Math.tanh(u));
    }
    return out;
  }

  function softmaxInPlace(a) {
    var mx = -Infinity;
    for (var i = 0; i < a.length; i++) if (a[i] > mx) mx = a[i];
    var sum = 0;
    for (i = 0; i < a.length; i++) { a[i] = Math.exp(a[i] - mx); sum += a[i]; }
    for (i = 0; i < a.length; i++) a[i] /= sum;
    return a;
  }

  // out[d] += vec[d] @ mat[d x d2] (mat хранится по строкам, row-major)
  function matVec(mat, d, d2, vec, bias) {
    var out = bias ? Float64Array.from(bias) : new Float64Array(d2);
    for (var i = 0; i < d; i++) {
      var vi = vec[i], row = i * d2;
      if (vi === 0) continue;
      for (var j = 0; j < d2; j++) out[j] += vi * mat[row + j];
    }
    return out;
  }

  // ======================================================================================
  // Модель
  // ======================================================================================
  function TransformerModel(m) {
    this.m = m;
    this.cfg = m.config;
    this.vocab = m.vocab;
    this.tokenizerType = m.tokenizer || "char";
    this.stoi = {};
    for (var i = 0; i < this.vocab.length; i++) this.stoi[this.vocab[i]] = i;
    this.maxQ = m.max_q;
    this.p = {};
    for (var k = 0; k < m.order.length; k++) {
      var key = m.order[k];
      this.p[key] = reshape(base64ToF16Array(m.weights[key]), m.shapes[key]);
    }
    if (this.tokenizerType === "bpe") {
      // BPETokenizer из bpe.js (должен быть загружен ДО этого файла)
      var MiniAIBPE = (typeof window !== "undefined" && window.MiniAIBPE) ||
                       (typeof require !== "undefined" ? require("./bpe.js") : null);
      this.bpe = new MiniAIBPE.BPETokenizer({ vocab: m.vocab, merges: m.merges || [] });
    }
  }

  TransformerModel.prototype.encode = function (s) {
    if (this.tokenizerType === "bpe") return this.bpe.encode(s);
    var out = [];
    for (var i = 0; i < s.length; i++) {
      var id = this.stoi[s[i]];
      if (id !== undefined) out.push(id);
    }
    return out;
  };

  TransformerModel.prototype.decode = function (ids) {
    if (this.tokenizerType === "bpe") return this.bpe.decode(ids);
    var out = "";
    for (var i = 0; i < ids.length; i++) {
      if (ids[i] >= SPECIALS.length) out += this.vocab[ids[i]];
    }
    return out;
  };

  TransformerModel.prototype.promptIds = function (question) {
    var q = this.encode(cleanQuestion(question)).slice(0, this.maxQ);
    return [BOS].concat(q, [SEP]);
  };

  // Один шаг декодирования: tok, позиция pos, cache (массив {k:[],v:[]} по слоям).
  // wantAttn=true → дополнительно возвращает attn[layer] = Float64Array (H*t) весов внимания.
  TransformerModel.prototype.step = function (tok, pos, cache, wantAttn) {
    var cfg = this.cfg, p = this.p;
    var d = cfg.d, H = cfg.n_head, dh = d / H;
    var wte = p.wte.data, wpe = p.wpe.data;
    var x = new Float64Array(d);
    for (var i = 0; i < d; i++) x[i] = wte[tok * d + i] + wpe[pos * d + i];
    var attnOut = wantAttn ? [] : null;

    for (var l = 0; l < cfg.n_layer; l++) {
      var pre = l + ".";
      var h = layernorm(x, d, p[pre + "ln1.g"].data, p[pre + "ln1.b"].data);
      var qkv = matVec(p[pre + "attn.wqkv"].data, d, 3 * d, h, p[pre + "attn.bqkv"].data);
      var layerCache = cache[l];
      var y = new Float64Array(d);
      var headAttn = wantAttn ? new Float64Array(H * (layerCache.k.length + 1)) : null;
      for (var head = 0; head < H; head++) {
        var qOff = head * dh, kOff = d + head * dh, vOff = 2 * d + head * dh;
        var q = qkv.subarray(qOff, qOff + dh);
        var kNew = qkv.slice(kOff, kOff + dh);
        var vNew = qkv.slice(vOff, vOff + dh);
        layerCache.k[head] = layerCache.k[head] || [];
        layerCache.v[head] = layerCache.v[head] || [];
        layerCache.k[head].push(kNew);
        layerCache.v[head].push(vNew);
        var T = layerCache.k[head].length;
        var scores = new Float64Array(T);
        var scale = 1 / Math.sqrt(dh);
        for (var t = 0; t < T; t++) {
          var kk = layerCache.k[head][t], s = 0;
          for (var c = 0; c < dh; c++) s += q[c] * kk[c];
          scores[t] = s * scale;
        }
        softmaxInPlace(scores);
        if (wantAttn) for (t = 0; t < T; t++) headAttn[head * T + t] = scores[t];
        for (t = 0; t < T; t++) {
          var vv = layerCache.v[head][t], sc = scores[t];
          for (c = 0; c < dh; c++) y[head * dh + c] += sc * vv[c];
        }
      }
      var attnOutProj = matVec(p[pre + "attn.wo"].data, d, d, y, p[pre + "attn.bo"].data);
      for (i = 0; i < d; i++) x[i] += attnOutProj[i];
      if (wantAttn) attnOut.push(headAttn);

      var h2 = layernorm(x, d, p[pre + "ln2.g"].data, p[pre + "ln2.b"].data);
      var dff = cfg.d_ff;
      var z = matVec(p[pre + "mlp.w1"].data, d, dff, h2, p[pre + "mlp.b1"].data);
      var gz = geluVec(z);
      var f = matVec(p[pre + "mlp.w2"].data, dff, d, gz, p[pre + "mlp.b2"].data);
      for (i = 0; i < d; i++) x[i] += f[i];
    }

    var hf = layernorm(x, d, p["lnf.g"].data, p["lnf.b"].data);
    var V = cfg.V;
    var logits = Float64Array.from(p.bout.data);
    for (i = 0; i < d; i++) {
      var hv = hf[i];
      if (hv === 0) continue;
      for (var vv2 = 0; vv2 < V; vv2++) logits[vv2] += hv * wte[vv2 * d + i];
    }
    return { logits: logits, attn: attnOut };
  };

  TransformerModel.prototype.newCache = function () {
    var cache = [];
    for (var l = 0; l < this.cfg.n_layer; l++) cache.push({ k: [], v: [] });
    return cache;
  };

  function argmax(a) {
    var best = 0;
    for (var i = 1; i < a.length; i++) if (a[i] > a[best]) best = i;
    return best;
  }

  function sampleWithTemperature(logits, temperature, rnd) {
    var p = Float64Array.from(logits);
    for (var i = 0; i < p.length; i++) p[i] /= temperature;
    softmaxInPlace(p);
    var r = (rnd || Math.random)(), acc = 0;
    for (i = 0; i < p.length; i++) { acc += p[i]; if (r <= acc) return i; }
    return p.length - 1;
  }

  // Выбор следующего символа (greedy при temperature=0, иначе сэмплинг) —
  // публичный метод, чтобы приложение могло «вручную» гонять шаги step()
  // с задержкой между ними и получать настоящий посимвольный стриминг
  // (а не притворную анимацию уже готовой строки).
  TransformerModel.prototype.pickNext = function (logits, temperature, rnd) {
    return temperature > 0 ? sampleWithTemperature(logits, temperature, rnd) : argmax(logits);
  };

  // Топ-k вероятностей следующего символа (для экрана «Как я работаю»)
  TransformerModel.prototype.topK = function (logits, k) {
    var probs = Float64Array.from(logits);
    softmaxInPlace(probs);
    var idx = Array.from({ length: probs.length }, function (_, i) { return i; });
    idx.sort(function (a, b) { return probs[b] - probs[a]; });
    var self = this;
    return idx.slice(0, k).map(function (i) {
      return { ch: i < SPECIALS.length ? SPECIALS[i] : self.piece(i), p: probs[i] };
    });
  };

  // Как токен выглядит на экране: для BPE «▁при» → « при» (▁ = начало слова), для char — сам символ.
  TransformerModel.prototype.piece = function (id) {
    var s = this.vocab[id] || "";
    return this.tokenizerType === "bpe" ? s.split("\u2581").join(" ") : s;
  };

  /*
   * generate — посимвольная генерация с KV-cache.
   *   promptIds: [<bos>, ...вопрос..., <sep>]
   *   opts.temperature = 0 → greedy (всегда самый вероятный символ)
   *   opts.onToken(charId, info) — вызывается на КАЖДЫЙ новый символ ответа:
   *        info.logits — Float64Array сырых логитов (для топ-5 на экране),
   *        info.attn   — внимание по слоям/головам на этом шаге (для тепловой карты)
   *   Возвращает {text, steps: [{logits, attn}, ...]} (steps — для отладки/тестов).
   */
  TransformerModel.prototype.generate = function (promptIds, opts) {
    opts = opts || {};
    var temperature = opts.temperature || 0;
    var maxNew = opts.maxNew || (this.cfg.block - promptIds.length);
    var wantAttn = !!opts.onToken || !!opts.wantAttn;
    var cache = this.newCache();
    var last = null;
    for (var i = 0; i < promptIds.length; i++) {
      last = this.step(promptIds[i], i, cache, wantAttn && i === promptIds.length - 1);
    }
    var out = [], pos = promptIds.length, steps = [];
    for (var n = 0; n < maxNew; n++) {
      var nxt = temperature > 0 ? sampleWithTemperature(last.logits, temperature, opts.rnd) : argmax(last.logits);
      steps.push({ logits: last.logits, attn: last.attn });
      if (nxt === EOS || pos >= this.cfg.block) break;
      out.push(nxt);
      if (opts.onToken) opts.onToken(nxt, { logits: last.logits, attn: last.attn, charIndex: out.length - 1 });
      last = this.step(nxt, pos, cache, wantAttn);
      pos++;
    }
    return { ids: out, text: this.decode(out), steps: steps };
  };

  var api = { TransformerModel: TransformerModel, cleanQuestion: cleanQuestion, f16ToFloat: f16ToFloat,
              base64ToF16Array: base64ToF16Array, SPECIALS: SPECIALS, BOS: BOS, SEP: SEP, EOS: EOS, PAD: PAD };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.MiniTransformer = api;
})(typeof window !== "undefined" ? window : this);
