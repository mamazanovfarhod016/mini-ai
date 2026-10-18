/*
 * Mini AI — дообучение трансформера НА ТЕЛЕФОНЕ, JS-реализация backprop.
 *
 * Это тот же самый ручной forward/backward, что в mini_ai/transformer.py
 * (см. комментарии там — здесь повторяем формулы в JS), но:
 *   - без батч-измерения в матрицах: каждый пример (вопрос+ответ) обрабатывается
 *     ПО ОДНОМУ (цикл по batch снаружи), поэтому не нужен паддинг/маска длины —
 *     у каждого примера своя настоящая длина T;
 *   - веса — обычные Float64Array (не float16): для обучения нужна точность;
 *   - работает и в Node (для тестов и сверки с Python), и в Web Worker в телефоне.
 *
 * Эта же модель здесь используется параллельно с app/www/transformer.js (тот
 * файл — только инференс, float16, оптимизирован под скорость чтения). Для
 * дообучения мы временно держим «рабочую» float64-копию весов в памяти, а
 * сохраняем в IndexedDB — уже как float16 base64 (тот же формат, что и
 * model_transformer.json), чтобы инференс (transformer.js) мог их читать.
 */
(function (root) {
  "use strict";

  var SPECIALS = ["<pad>", "<bos>", "<sep>", "<eos>"];
  var PAD = 0, BOS = 1, SEP = 2, EOS = 3;
  var APOS = /['`’‘ʻʼ´]/g;

  function cleanQuestion(text) {
    var t = String(text).toLowerCase().replace(/ё/g, "е");
    t = t.replace(APOS, "'");
    t = t.replace(/[!?.,;:()"«»…—–-]+/g, " ");
    return t.split(/\s+/).filter(Boolean).join(" ");
  }

  // ---------- float16 helpers (те же, что в transformer.js) ----------
  function f16ToFloat(h) {
    var s = (h & 0x8000) ? -1 : 1;
    var e = (h >> 10) & 0x1f;
    var f = h & 0x3ff;
    if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + f / 1024);
  }

  function floatToF16(v) {
    if (v === 0) return 0;
    if (Number.isNaN(v)) return 0x7e00;
    var s = v < 0 ? 1 : 0;
    v = Math.abs(v);
    if (v > 65504) v = 65504; // клампим под диапазон float16
    var e = Math.floor(Math.log2(v));
    var m = v / Math.pow(2, e) - 1;
    e += 15;
    if (e <= 0) { // субнормальные
      m = v / Math.pow(2, -14);
      return (s << 15) | Math.round(m * 1024);
    }
    if (e >= 31) return (s << 15) | (31 << 10); // overflow → inf
    return (s << 15) | (e << 10) | Math.round(m * 1024);
  }

  function base64ToF16Array(b64) {
    var bin = typeof atob === "function" ? atob(b64) : Buffer.from(b64, "base64").toString("binary");
    var n = bin.length / 2;
    var out = new Float64Array(n);
    for (var i = 0; i < n; i++) {
      var lo = bin.charCodeAt(2 * i), hi = bin.charCodeAt(2 * i + 1);
      out[i] = f16ToFloat(lo | (hi << 8));
    }
    return out;
  }

  function f64ArrayToBase64F16(arr) {
    var bytes = new Uint8Array(arr.length * 2);
    for (var i = 0; i < arr.length; i++) {
      var h = floatToF16(arr[i]);
      bytes[2 * i] = h & 0xff;
      bytes[2 * i + 1] = (h >> 8) & 0xff;
    }
    var bin = "";
    for (i = 0; i < bytes.length; i++) bin += String.fromCharCode(bytes[i]);
    return typeof btoa === "function" ? btoa(bin) : Buffer.from(bin, "binary").toString("base64");
  }

  // ======================================================================================
  // Параметры: грузим из JSON-модели (как model_transformer.json) в Float64Array
  // ======================================================================================
  function loadParamsF64(modelJson) {
    var p = {};
    for (var i = 0; i < modelJson.order.length; i++) {
      var k = modelJson.order[i];
      p[k] = base64ToF16Array(modelJson.weights[k]);
    }
    return p;
  }

  function cloneParams(p) {
    var out = {};
    for (var k in p) out[k] = Float64Array.from(p[k]);
    return out;
  }

  function zerosLike(p, keys) {
    var out = {};
    (keys || Object.keys(p)).forEach(function (k) { out[k] = new Float64Array(p[k].length); });
    return out;
  }

  // ======================================================================================
  // Базовые слои (по одному примеру: X — Float64Array длины n*d, n строк)
  // ======================================================================================
  function matmul(X, n, d, W, d2, bias) {
    var Y = new Float64Array(n * d2);
    if (bias) for (var i = 0; i < n; i++) Y.set(bias, i * d2);
    for (i = 0; i < n; i++) {
      var xo = i * d, yo = i * d2;
      for (var k = 0; k < d; k++) {
        var xv = X[xo + k];
        if (xv === 0) continue;
        var wo = k * d2;
        for (var j = 0; j < d2; j++) Y[yo + j] += xv * W[wo + j];
      }
    }
    return Y;
  }

  function matmulBackward(dY, X, n, d, W, d2, wantDX) {
    var dW = new Float64Array(d * d2), db = new Float64Array(d2);
    var dX = wantDX ? new Float64Array(n * d) : null;
    for (var i = 0; i < n; i++) {
      var xo = i * d, yo = i * d2;
      for (var j = 0; j < d2; j++) db[j] += dY[yo + j];
      for (var k = 0; k < d; k++) {
        var xv = X[xo + k];
        var wo = k * d2;
        var dxacc = 0;
        for (var j2 = 0; j2 < d2; j2++) {
          var dy = dY[yo + j2];
          dW[wo + j2] += xv * dy;
          if (wantDX) dxacc += dy * W[wo + j2];
        }
        if (wantDX) dX[xo + k] = dxacc;
      }
    }
    return { dX: dX, dW: dW, db: db };
  }

  var LN_EPS = 1e-5;

  function layernormForward(X, n, d, g, b) {
    var Y = new Float64Array(n * d);
    var xhat = new Float64Array(n * d), rstd = new Float64Array(n);
    for (var i = 0; i < n; i++) {
      var o = i * d, mu = 0;
      for (var k = 0; k < d; k++) mu += X[o + k];
      mu /= d;
      var varr = 0;
      for (k = 0; k < d; k++) { var c = X[o + k] - mu; varr += c * c; }
      varr /= d;
      var rs = 1 / Math.sqrt(varr + LN_EPS);
      rstd[i] = rs;
      for (k = 0; k < d; k++) {
        var xh = (X[o + k] - mu) * rs;
        xhat[o + k] = xh;
        Y[o + k] = xh * g[k] + b[k];
      }
    }
    return { Y: Y, cache: { xhat: xhat, rstd: rstd } };
  }

  function layernormBackward(dY, n, d, g, cache) {
    var xhat = cache.xhat, rstd = cache.rstd;
    var dX = new Float64Array(n * d), dg = new Float64Array(d), db = new Float64Array(d);
    for (var i = 0; i < n; i++) {
      var o = i * d;
      var sumDxhat = 0, sumDxhatXhat = 0;
      var dxhat = new Float64Array(d);
      for (var k = 0; k < d; k++) {
        var dy = dY[o + k];
        dg[k] += dy * xhat[o + k];
        db[k] += dy;
        var dxh = dy * g[k];
        dxhat[k] = dxh;
        sumDxhat += dxh;
        sumDxhatXhat += dxh * xhat[o + k];
      }
      sumDxhat /= d; sumDxhatXhat /= d;
      for (k = 0; k < d; k++) {
        dX[o + k] = rstd[i] * (dxhat[k] - sumDxhat - xhat[o + k] * sumDxhatXhat);
      }
    }
    return { dX: dX, dg: dg, db: db };
  }

  var GELU_C = Math.sqrt(2 / Math.PI);

  function geluForward(X) {
    var n = X.length;
    var Y = new Float64Array(n), T = new Float64Array(n);
    for (var i = 0; i < n; i++) {
      var v = X[i];
      var u = GELU_C * (v + 0.044715 * v * v * v);
      var t = Math.tanh(u);
      T[i] = t;
      Y[i] = 0.5 * v * (1 + t);
    }
    return { Y: Y, cache: { X: X, T: T } };
  }

  function geluBackward(dY, cache) {
    var X = cache.X, T = cache.T;
    var dX = new Float64Array(X.length);
    for (var i = 0; i < X.length; i++) {
      var x = X[i], t = T[i];
      dX[i] = dY[i] * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * GELU_C * (1 + 3 * 0.044715 * x * x));
    }
    return dX;
  }

  function softmaxRow(a, lo, hi) {
    // softmax по a[lo..hi) in-place
    var mx = -Infinity;
    for (var i = lo; i < hi; i++) if (a[i] > mx) mx = a[i];
    var sum = 0;
    for (i = lo; i < hi; i++) { a[i] = Math.exp(a[i] - mx); sum += a[i]; }
    for (i = lo; i < hi; i++) a[i] /= sum;
  }

  // ======================================================================================
  // Causal self-attention для ОДНОЙ последовательности (T позиций, d_model, H голов)
  // ======================================================================================
  function attentionForward(h, T, d, Wqkv, bqkv, Wo, bo, H) {
    var dh = d / H;
    var qkv = matmul(h, T, d, Wqkv, 3 * d, bqkv); // T x 3d
    var scale = 1 / Math.sqrt(dh);
    // A[head][t] = Float64Array(t+1) веса внимания на позиции 0..t
    var A = [];
    var Ym = new Float64Array(T * d); // склеенные головы
    for (var head = 0; head < H; head++) {
      A.push([]);
      var qOff = head * dh, kOff = d + head * dh, vOff = 2 * d + head * dh;
      for (var t = 0; t < T; t++) {
        var scores = new Float64Array(t + 1);
        for (var j = 0; j <= t; j++) {
          var s = 0;
          for (var c = 0; c < dh; c++) s += qkv[t * 3 * d + qOff + c] * qkv[j * 3 * d + kOff + c];
          scores[j] = s * scale;
        }
        softmaxRow(scores, 0, t + 1);
        A[head].push(scores);
        for (j = 0; j <= t; j++) {
          var w = scores[j];
          for (c = 0; c < dh; c++) Ym[t * d + head * dh + c] += w * qkv[j * 3 * d + vOff + c];
        }
      }
    }
    var out = matmul(Ym, T, d, Wo, d, bo);
    return { out: out, cache: { h: h, qkv: qkv, A: A, Ym: Ym, Wqkv: Wqkv, Wo: Wo, scale: scale, H: H, dh: dh, T: T, d: d } };
  }

  function attentionBackward(dOut, cache) {
    var h = cache.h, qkv = cache.qkv, A = cache.A, Ym = cache.Ym, Wqkv = cache.Wqkv, Wo = cache.Wo;
    var scale = cache.scale, H = cache.H, dh = cache.dh, T = cache.T, d = cache.d;
    var mm = matmulBackward(dOut, Ym, T, d, Wo, d, true);
    var dWo = mm.dW, dbo = mm.db, dYm = mm.dX; // T x d
    var dqkv = new Float64Array(T * 3 * d);
    for (var head = 0; head < H; head++) {
      var qOff = head * dh, kOff = d + head * dh, vOff = 2 * d + head * dh;
      // dV[j] = sum_t A[t][j] * dYm[t, head]      (j <= t)
      // dA[t][j] = dYm[t,head] . V[j]
      // dS[t][j] (через якобиан softmax) -> dQ[t], dK[j]
      for (var t = 0; t < T; t++) {
        var scores = A[head][t];
        var dA = new Float64Array(t + 1);
        for (var j = 0; j <= t; j++) {
          var s = 0;
          for (var c = 0; c < dh; c++) s += dYm[t * d + head * dh + c] * qkv[j * 3 * d + vOff + c];
          dA[j] = s;
        }
        var dot = 0;
        for (j = 0; j <= t; j++) dot += dA[j] * scores[j];
        var dS = new Float64Array(t + 1);
        for (j = 0; j <= t; j++) dS[j] = scores[j] * (dA[j] - dot) * scale;
        // dQ[t] += sum_j dS[j] * K[j];  dK[j] += dS[j] * Q[t];  dV[j] += scores[j]*dYm[t]
        for (j = 0; j <= t; j++) {
          var dsj = dS[j];
          for (c = 0; c < dh; c++) {
            dqkv[t * 3 * d + qOff + c] += dsj * qkv[j * 3 * d + kOff + c];
            dqkv[j * 3 * d + kOff + c] += dsj * qkv[t * 3 * d + qOff + c];
          }
          var wj = scores[j];
          for (c = 0; c < dh; c++) dqkv[j * 3 * d + vOff + c] += wj * dYm[t * d + head * dh + c];
        }
      }
    }
    var mm2 = matmulBackward(dqkv, h, T, d, Wqkv, 3 * d, true);
    return { dH: mm2.dX, dWqkv: mm2.dW, dbqkv: mm2.db, dWo: dWo, dbo: dbo };
  }

  // ======================================================================================
  // Блок трансформера (pre-LN): x = x + Attn(LN1(x));  x = x + MLP(LN2(x))
  // ======================================================================================
  function blockForward(x, T, d, p, l, H, dff) {
    var pre = l + ".";
    var ln1 = layernormForward(x, T, d, p[pre + "ln1.g"], p[pre + "ln1.b"]);
    var att = attentionForward(ln1.Y, T, d, p[pre + "attn.wqkv"], p[pre + "attn.bqkv"], p[pre + "attn.wo"], p[pre + "attn.bo"], H);
    var x2 = new Float64Array(T * d);
    for (var i = 0; i < T * d; i++) x2[i] = x[i] + att.out[i];
    var ln2 = layernormForward(x2, T, d, p[pre + "ln2.g"], p[pre + "ln2.b"]);
    var z = matmul(ln2.Y, T, d, p[pre + "mlp.w1"], dff, p[pre + "mlp.b1"]);
    var gz = geluForward(z);
    var f = matmul(gz.Y, T, dff, p[pre + "mlp.w2"], d, p[pre + "mlp.b2"]);
    var y = new Float64Array(T * d);
    for (i = 0; i < T * d; i++) y[i] = x2[i] + f[i];
    return { y: y, cache: { ln1: ln1, att: att, x2: x2, ln2: ln2, z: z, gz: gz, T: T, d: d, dff: dff } };
  }

  function blockBackward(dy, cache, p, l, grads) {
    var pre = l + ".";
    var c = cache, T = c.T, d = c.d, dff = c.dff;
    // --- MLP-ветка ---
    var mm2 = matmulBackward(dy, c.gz.Y, T, dff, p[pre + "mlp.w2"], d, true);
    addInto(grads[pre + "mlp.w2"], mm2.dW); addInto(grads[pre + "mlp.b2"], mm2.db);
    var dz = geluBackward(mm2.dX, c.gz.cache);
    var mm1 = matmulBackward(dz, c.ln2.Y, T, d, p[pre + "mlp.w1"], dff, true);
    addInto(grads[pre + "mlp.w1"], mm1.dW); addInto(grads[pre + "mlp.b1"], mm1.db);
    var ln2b = layernormBackward(mm1.dX, T, d, p[pre + "ln2.g"], c.ln2.cache);
    addInto(grads[pre + "ln2.g"], ln2b.dg); addInto(grads[pre + "ln2.b"], ln2b.db);
    var dx2 = new Float64Array(T * d);
    for (var i = 0; i < T * d; i++) dx2[i] = ln2b.dX[i] + dy[i];
    // --- ветка внимания ---
    var attB = attentionBackward(dx2, c.att.cache);
    addInto(grads[pre + "attn.wqkv"], attB.dWqkv); addInto(grads[pre + "attn.bqkv"], attB.dbqkv);
    addInto(grads[pre + "attn.wo"], attB.dWo); addInto(grads[pre + "attn.bo"], attB.dbo);
    var ln1b = layernormBackward(attB.dH, T, d, p[pre + "ln1.g"], c.ln1.cache);
    addInto(grads[pre + "ln1.g"], ln1b.dg); addInto(grads[pre + "ln1.b"], ln1b.db);
    var dx = new Float64Array(T * d);
    for (i = 0; i < T * d; i++) dx[i] = ln1b.dX[i] + dx2[i];
    return dx;
  }

  function addInto(dst, src) { for (var i = 0; i < dst.length; i++) dst[i] += src[i]; }

  // ======================================================================================
  // Вся модель, по одному примеру (seq — массив id символов длины T)
  //   trainFrom: индекс слоя, НАЧИНАЯ с которого считаем backward (0 = все слои, как в Python).
  //   Слои < trainFrom не получают градиент (экономим время при частичном дообучении) —
  //   но forward всё равно считаем по всем слоям (нужно для верного ответа).
  // ======================================================================================
  function forwardSeq(p, cfg, seq) {
    var T = seq.length, d = cfg.d;
    var x = new Float64Array(T * d);
    for (var t = 0; t < T; t++) {
      var tok = seq[t];
      for (var k = 0; k < d; k++) x[t * d + k] = p.wte[tok * d + k] + p.wpe[t * d + k];
    }
    var caches = [];
    for (var l = 0; l < cfg.n_layer; l++) {
      var b = blockForward(x, T, d, p, l, cfg.n_head, cfg.d_ff);
      x = b.y;
      caches.push(b.cache);
    }
    var lnf = layernormForward(x, T, d, p["lnf.g"], p["lnf.b"]);
    var V = cfg.V;
    var logits = matmul(lnf.Y, T, d, transposeWte(p.wte, V, d), V, p.bout);
    return { logits: logits, cache: { seq: seq, T: T, caches: caches, lnf: lnf, hf: lnf.Y } };
  }

  // wte: V x d (строка на символ). Для logits = hf @ wte^T нужна матрица d x V.
  // Кэшируем транспонирование не нужно пересчитывать каждый раз — но для простоты
  // (размеры маленькие: V~151, d~96) считаем прямым произведением без явного transpose.
  function transposeWte(wte, V, d) {
    var out = new Float64Array(d * V);
    for (var v = 0; v < V; v++) for (var k = 0; k < d; k++) out[k * V + v] = wte[v * d + k];
    return out;
  }

  function backwardSeq(p, cfg, dlogits, cache, grads, trainFrom) {
    var T = cache.T, d = cfg.d, V = cfg.V;
    // grads.bout += sum dlogits; grads.wte += dlogits^T @ hf (+ сложить обратно эмбеддинги)
    for (var t = 0; t < T; t++) for (var v = 0; v < V; v++) grads.bout[v] += dlogits[t * V + v];
    var hf = cache.hf;
    // dwte (из logits=hf@wte^T): dwte[v,k] += sum_t dlogits[t,v]*hf[t,k]
    for (t = 0; t < T; t++) {
      for (var v = 0; v < V; v++) {
        var dl = dlogits[t * V + v];
        if (dl === 0) continue;
        var wo = v * d;
        for (var k = 0; k < d; k++) grads.wte[wo + k] += dl * hf[t * d + k];
      }
    }
    // dhf[t,k] = sum_v dlogits[t,v]*wte[v,k]
    var dhf = new Float64Array(T * d);
    for (t = 0; t < T; t++) {
      for (var v2 = 0; v2 < V; v2++) {
        var dl2 = dlogits[t * V + v2];
        if (dl2 === 0) continue;
        var wo2 = v2 * d;
        for (var k2 = 0; k2 < d; k2++) dhf[t * d + k2] += dl2 * p.wte[wo2 + k2];
      }
    }
    var lnfB = layernormBackward(dhf, T, d, p["lnf.g"], cache.lnf.cache);
    addInto(grads["lnf.g"], lnfB.dg); addInto(grads["lnf.b"], lnfB.db);
    var dx = lnfB.dX;
    for (var l = cfg.n_layer - 1; l >= 0; l--) {
      if (l >= trainFrom) {
        dx = blockBackward(dx, cache.caches[l], p, l, grads);
      } else {
        break; // слои ниже trainFrom не трогаем вовсе (ни grads, ни дальнейший dx не нужен)
      }
    }
    if (trainFrom === 0) {
      // эмбеддинги получают градиент только если дообучаем всё (wte используется и
      // как embedding, и как выходная матрица — поэтому его грамотно обучать всегда,
      // здесь выше мы уже добавили вклад из выходного слоя; добавим вклад из входа,
      // только если реально обучаем входной эмбеддинг — т.е. trainFrom===0).
      for (t = 0; t < T; t++) {
        var tok = cache.seq[t];
        for (var k3 = 0; k3 < d; k3++) {
          grads.wpe[t * d + k3] += dx[t * d + k3];
          grads.wte[tok * d + k3] += dx[t * d + k3];
        }
      }
    }
  }

  // ======================================================================================
  // Кодирование примера: <bos> вопрос <sep> ответ <eos>
  // ======================================================================================
  function encodeExample(stoi, q, a, maxQ) {
    var qc = cleanQuestion(q);
    var qIds = encodeStr(stoi, qc).slice(0, maxQ);
    var aIds = encodeStr(stoi, a);
    var seq = [BOS].concat(qIds, [SEP], aIds, [EOS]);
    var ansStart = 1 + qIds.length + 1;
    return { seq: seq, ansStart: ansStart };
  }

  function encodeStr(stoi, s) {
    var out = [];
    for (var i = 0; i < s.length; i++) if (stoi[s[i]] !== undefined) out.push(stoi[s[i]]);
    return out;
  }

  // ======================================================================================
  // Loss + backward для одного примера, накопление в grads. Возвращает {loss, nTok}.
  // ======================================================================================
  function lossAndGradsOne(p, cfg, stoi, example, grads, trainFrom, totalTok) {
    var enc = encodeExample(stoi, example.q, example.a, cfg.max_q || 40);
    var seq = enc.seq.slice(0, cfg.block);
    var T = seq.length, V = cfg.V;
    if (T < 2) return { loss: 0, nTok: 0 };
    var fw = forwardSeq(p, cfg, seq);
    var logits = fw.logits;
    var dlogits = new Float64Array(T * V);
    var loss = 0, nTok = 0;
    for (var t = enc.ansStart - 1; t < T - 1; t++) {
      var target = seq[t + 1];
      var row = logits.subarray(t * V, t * V + V);
      var probs = Float64Array.from(row);
      softmaxRow(probs, 0, V);
      var w = 1 / totalTok;
      for (var v = 0; v < V; v++) dlogits[t * V + v] = probs[v] * w;
      dlogits[t * V + target] -= w;
      loss += -Math.log(Math.max(probs[target], 1e-30));
      nTok++;
    }
    backwardSeq(p, cfg, dlogits, fw.cache, grads, trainFrom);
    return { loss: loss, nTok: nTok };
  }

  // ======================================================================================
  // Adam — только по ключам, которые реально обучаем (trainableKeys)
  // ======================================================================================
  function AdamJS(trainableKeys, p, lr, wd) {
    this.keys = trainableKeys;
    this.lr = lr; this.b1 = 0.9; this.b2 = 0.98; this.eps = 1e-8; this.wd = wd || 0.0;
    this.m = {}; this.v = {};
    var self = this;
    trainableKeys.forEach(function (k) { self.m[k] = new Float64Array(p[k].length); self.v[k] = new Float64Array(p[k].length); });
    this.t = 0;
  }
  AdamJS.prototype.step = function (p, grads, lr, isMatrix) {
    this.t++;
    var c1 = 1 - Math.pow(this.b1, this.t), c2 = 1 - Math.pow(this.b2, this.t);
    for (var ki = 0; ki < this.keys.length; ki++) {
      var k = this.keys[ki];
      var g = grads[k], w = p[k], m = this.m[k], v = this.v[k];
      var wd = (isMatrix && isMatrix(k)) ? this.wd : 0;
      for (var i = 0; i < w.length; i++) {
        m[i] = this.b1 * m[i] + (1 - this.b1) * g[i];
        v[i] = this.b2 * v[i] + (1 - this.b2) * g[i] * g[i];
        var upd = (m[i] / c1) / (Math.sqrt(v[i] / c2) + this.eps) + wd * w[i];
        w[i] -= lr * upd;
      }
    }
  };

  function clipGrads(grads, keys, maxNorm) {
    var sum = 0;
    keys.forEach(function (k) { var g = grads[k]; for (var i = 0; i < g.length; i++) sum += g[i] * g[i]; });
    var norm = Math.sqrt(sum);
    if (norm > maxNorm) {
      var s = maxNorm / (norm + 1e-6);
      keys.forEach(function (k) { var g = grads[k]; for (var i = 0; i < g.length; i++) g[i] *= s; });
    }
    return norm;
  }

  // Какие ключи весов обучаем, если дообучаем только «верх»: последний блок,
  // финальный LayerNorm и выходной слой (bout; wte участвует в выходе тоже).
  function trainableKeysFor(cfg, mode) {
    var keys = [];
    if (mode === "all") {
      for (var l = 0; l < cfg.n_layer; l++) keys = keys.concat(layerKeys(l));
      keys.push("wte", "wpe", "lnf.g", "lnf.b", "bout");
      return { keys: keys, trainFrom: 0 };
    }
    // "top": последний блок + финальный LN + bout (+ wte, он общий со входом и выходом)
    var topLayer = cfg.n_layer - 1;
    keys = layerKeys(topLayer).concat(["lnf.g", "lnf.b", "bout", "wte"]);
    return { keys: keys, trainFrom: topLayer };
  }

  function layerKeys(l) {
    var pre = l + ".";
    return [pre + "ln1.g", pre + "ln1.b", pre + "attn.wqkv", pre + "attn.bqkv", pre + "attn.wo", pre + "attn.bo",
            pre + "ln2.g", pre + "ln2.b", pre + "mlp.w1", pre + "mlp.b1", pre + "mlp.w2", pre + "mlp.b2"];
  }

  function isMatrixKey(k) {
    return !(k === "wpe" || /ln[12f]?\.(g|b)$/.test(k) || k === "bout");
  }

  // ======================================================================================
  // Цикл обучения: примеры = [{q,a}], replay = [{q,a}] (старые пары — против забывания).
  // opts: {mode:'all'|'top', steps, batch, lr, warmup, onProgress(step,total,loss), shouldStop()}
  // ======================================================================================
  function trainTeach(p, cfg, stoi, newExamples, replayExamples, opts) {
    opts = opts || {};
    var mode = opts.mode || "top";
    var tk = trainableKeysFor(cfg, mode);
    var keys = tk.keys, trainFrom = tk.trainFrom;
    var steps = opts.steps || 60;
    var batch = opts.batch || 16;
    var lr0 = opts.lr || 2e-3;
    var warmup = opts.warmup || 8;
    var opt = new AdamJS(keys, p, lr0, 0.01);
    var rng = opts.rng || Math.random;
    var lastLoss = 0;
    for (var step = 0; step < steps; step++) {
      if (opts.shouldStop && opts.shouldStop()) break;
      // Батч: половина новых пар (с повтором, если их мало), половина — replay.
      var batchEx = [];
      for (var i = 0; i < batch; i++) {
        if (i % 2 === 0 || replayExamples.length === 0) {
          batchEx.push(newExamples[Math.floor(rng() * newExamples.length)]);
        } else {
          batchEx.push(replayExamples[Math.floor(rng() * replayExamples.length)]);
        }
      }
      var grads = zerosLike(p, keys);
      // Сначала считаем общее число токенов-ответов в батче (для правильного
      // усреднения loss/градиента — как mask/N в Python).
      var totalTok = 0, encs = [];
      for (i = 0; i < batchEx.length; i++) {
        var enc = encodeExample(stoi, batchEx[i].q, batchEx[i].a, cfg.max_q || 40);
        enc.seq = enc.seq.slice(0, cfg.block);
        encs.push(enc);
        totalTok += Math.max(0, enc.seq.length - enc.ansStart);
      }
      totalTok = Math.max(1, totalTok);
      var lossSum = 0;
      for (i = 0; i < batchEx.length; i++) {
        var r = lossAndGradsOne(p, cfg, stoi, batchEx[i], grads, trainFrom, totalTok);
        lossSum += r.loss;
      }
      var lr = step < warmup ? lr0 * (step + 1) / warmup : lr0 * (0.1 + 0.9 * 0.5 * (1 + Math.cos(Math.PI * Math.min(1, (step - warmup) / Math.max(1, steps - warmup)))));
      clipGrads(grads, keys, 1.0);
      opt.step(p, grads, lr, isMatrixKey);
      lastLoss = lossSum / totalTok;
      if (opts.onProgress) opts.onProgress(step + 1, steps, lastLoss);
    }
    return { loss: lastLoss };
  }

  // ======================================================================================
  // Greedy-генерация ответа по обучаемым (full precision) весам — для проверки «сразу после обучения».
  // Переиспользуем ту же логику, что KVModel в transformer.py / TransformerModel в transformer.js,
  // но здесь без KV-кэша (модель маленькая, для проверки пары вопросов это не важно).
  // ======================================================================================
  function greedyAnswer(p, cfg, stoi, itos, question, maxNew) {
    var qIds = encodeStr(stoi, cleanQuestion(question)).slice(0, cfg.max_q || 40);
    var seq = [BOS].concat(qIds, [SEP]);
    maxNew = maxNew || (cfg.block - seq.length);
    for (var n = 0; n < maxNew; n++) {
      var fw = forwardSeq(p, cfg, seq);
      var T = fw.cache.T, V = cfg.V;
      var lastLogits = fw.logits.subarray((T - 1) * V, T * V);
      var best = 0;
      for (var v = 1; v < V; v++) if (lastLogits[v] > lastLogits[best]) best = v;
      if (best === EOS || seq.length >= cfg.block) break;
      seq.push(best);
    }
    var out = "";
    for (var i = qIds.length + 2; i < seq.length; i++) if (seq[i] >= SPECIALS.length) out += itos[seq[i]];
    return out;
  }

  var api = {
    cleanQuestion: cleanQuestion, f16ToFloat: f16ToFloat, floatToF16: floatToF16,
    base64ToF16Array: base64ToF16Array, f64ArrayToBase64F16: f64ArrayToBase64F16,
    loadParamsF64: loadParamsF64, cloneParams: cloneParams, zerosLike: zerosLike,
    matmul: matmul, matmulBackward: matmulBackward,
    layernormForward: layernormForward, layernormBackward: layernormBackward,
    geluForward: geluForward, geluBackward: geluBackward,
    attentionForward: attentionForward, attentionBackward: attentionBackward,
    blockForward: blockForward, blockBackward: blockBackward,
    forwardSeq: forwardSeq, backwardSeq: backwardSeq,
    encodeExample: encodeExample, encodeStr: encodeStr,
    lossAndGradsOne: lossAndGradsOne,
    AdamJS: AdamJS, clipGrads: clipGrads,
    trainableKeysFor: trainableKeysFor, layerKeys: layerKeys, isMatrixKey: isMatrixKey,
    trainTeach: trainTeach, greedyAnswer: greedyAnswer,
    SPECIALS: SPECIALS, BOS: BOS, SEP: SEP, EOS: EOS, PAD: PAD,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.MiniTeachCore = api;
})(typeof self !== "undefined" ? self : (typeof window !== "undefined" ? window : this));
