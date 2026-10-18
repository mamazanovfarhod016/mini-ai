/*
 * Mini AI v0.3 — самообучение через интернет (app/www/learn.js).
 *
 * Как работает (схема — в README, раздел «Самообучение»):
 *   1. Каждый вопрос и ответ модели уходят на сервер (/api/log) — из них сервер учится.
 *   2. Если модель ответила «не знаю» или пользователь нажал «Неправильно» — ищем ответ
 *      в интернете (/api/research: Википедия + сверка с Wikidata).
 *   3. Нашли → показываем с источником, спрашиваем «Верно?». Верно → запоминаем.
 *   4. Не нашли или «Неправильно» → «Отправить этот вопрос наставнику?» → /api/mentor.
 *   5. Ответ наставника приходит в чат (опрос /api/mentor/answers) и запоминается.
 *   6. Ночью сервер дообучает модель; при запуске приложение скачивает новую версию,
 *      но только если она прошла экзамен на сервере (не хуже прежней).
 * Без интернета всё работает как раньше; вопросы копятся в очереди и уйдут позже.
 */
(function (root) {
  "use strict";

  var DISCOVERY = "https://raw.githubusercontent.com/mamazanovfarhod016/mini-ai/live/server.json";
  var K = {
    server: "miniai.server", outbox: "miniai.outbox", memory: "miniai.memory",
    pending: "miniai.mentor.pending", device: "miniai.device",
    modelV: "miniai.model.version", modelJson: "miniai.model.json"
  };

  function get(k, d) { try { var v = localStorage.getItem(k); return v === null ? d : JSON.parse(v); } catch (e) { return d; } }
  function set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); return true; } catch (e) { return false; } }

  var device = get(K.device, null);
  if (!device) { device = "d" + Math.random().toString(36).slice(2, 10); set(K.device, device); }

  function clean(q) {
    return root.MiniTransformer ? root.MiniTransformer.cleanQuestion(q)
      : String(q).toLowerCase().replace(/[!?.,;:]+/g, " ").trim();
  }

  // ---------- сервер ----------
  var serverUrl = get(K.server, null), discovering = null;

  function withTimeout(p, ms) {
    return Promise.race([p, new Promise(function (_, rej) { setTimeout(function () { rej(new Error("timeout")); }, ms); })]);
  }

  function discover() {
    if (discovering) return discovering;
    discovering = withTimeout(fetch(DISCOVERY + "?t=" + Date.now(), { cache: "no-store" }), 8000)
      .then(function (r) { return r.json(); })
      .then(function (j) { if (j && j.url) { serverUrl = j.url.replace(/\/$/, ""); set(K.server, serverUrl); } return serverUrl; })
      .catch(function () { return serverUrl; })
      .then(function (u) { discovering = null; return u; });
    return discovering;
  }

  function call(path, body, retried) {
    var go = function (base) {
      if (!base) return Promise.reject(new Error("no server"));
      var opt = body ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) } : {};
      return withTimeout(fetch(base + path, opt), body && path === "/api/research" ? 20000 : 10000)
        .then(function (r) { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); });
    };
    return go(serverUrl).catch(function (e) {
      if (retried) throw e;
      return discover().then(function (u) { return go(u); });   // адрес сервера мог смениться
    });
  }

  // ---------- 1. журнал вопросов (с очередью на случай офлайна) ----------
  function flushOutbox() {
    var box = get(K.outbox, []);
    if (!box.length) return Promise.resolve();
    var item = box[0];
    return call("/api/log", item).then(function () {
      var b = get(K.outbox, []); b.shift(); set(K.outbox, b);
      return flushOutbox();
    }).catch(function () {});
  }

  function logQuestion(q, a, ok, lang) {
    var box = get(K.outbox, []);
    box.push({ q: q, a: a, ok: ok, lang: lang, device: device });
    set(K.outbox, box.slice(-300));
    flushOutbox();
  }

  // ---------- память: выученное от наставника и из интернета (сразу, до ночного обучения) ----------
  function recall(q) { var m = get(K.memory, {}); return m[clean(q)] || null; }
  function remember(q, a, source) { var m = get(K.memory, {}); m[clean(q)] = { a: a, source: source }; set(K.memory, m); }

  // ---------- понимает ли модель вопрос ----------
  // 1) Вопрос «про незнакомое»: есть слово, которого не было в обучении (с допуском 1 опечатки),
  //    И это вопрос о факте («что такое…», «кто такой…», «столица…»). Маленькая модель на таких
  //    вопросах всё равно уверенно пишет заготовку — её softmax-уверенности верить нельзя.
  // 2) Ответ модели — «не знаю»/«не поняла» (намерения out_of_scope / not_understood).
  var WEAK = { out_of_scope: 1, not_understood: 1, wrong_feedback: 1 };
  var CUE = /^(что|кто|где|когда|сколько|какой|какая|какое|какие|каков|почему|зачем|расскажи|объясни|столица|nima|kim|qayer|qachon|qancha|necha|nega|qanday|qaysi)(\s|$)|(^|\s)(такое|такой|столица|poytaxti|haqida)(\s|$)/;
  var STOPW = { "что": 1, "кто": 1, "как": 1, "где": 1, "мне": 1, "тебя": 1 };
  var known = null, byLen = null;
  function loadKnown() {
    if (known) return;
    known = {}; byLen = {};
    (root.MINI_AI_KNOWN_WORDS || []).forEach(function (w) { known[w] = 1; (byLen[w.length] = byLen[w.length] || []).push(w); });
    var m = get(K.memory, {});
    Object.keys(m).forEach(function (q) { q.split(" ").forEach(function (w) { known[w] = 1; }); });
  }
  function lev1(a, b) {
    if (Math.abs(a.length - b.length) > 1) return false;
    if (a.length > b.length) { var t = a; a = b; b = t; }
    var i = 0, j = 0, d = 0;
    while (i < a.length && j < b.length) {
      if (a[i] === b[j]) { i++; j++; continue; }
      if (++d > 1) return false;
      if (a.length === b.length) i++;
      j++;
    }
    return d + (b.length - j) <= 1;
  }
  function familiar(w) {
    if (known[w] || STOPW[w]) return true;
    if (w.length < 4) return false;
    for (var L = w.length - 1; L <= w.length + 1; L++) {
      var list = byLen[L] || [];
      for (var i = 0; i < list.length; i++) if (lev1(w, list[i])) return true;
    }
    return false;
  }
  function unknownWords(q) {
    loadKnown();
    return clean(q).split(" ").filter(function (w) { return w.length >= 3 && /^[a-zа-яё']+$/.test(w) && !familiar(w); });
  }
  function needsWeb(q) { return !!root.MINI_AI_KNOWN_WORDS && unknownWords(q).length > 0 && CUE.test(clean(q)); }

  function intentOf(raw, model) {
    var resp = (model && model.responses) || {};
    var t = String(raw || "").trim();
    for (var tag in resp) {
      for (var lang in resp[tag]) {
        var list = resp[tag][lang];
        for (var i = 0; i < list.length; i++) {
          var g = list[i].split("{time}").join("\u2460").split("{date}").join("\u2461").trim();
          if (g === t) return tag;
        }
      }
    }
    return null;
  }
  function isConfident(info) {
    if (info.memory) return true;
    if (needsWeb(info.q)) return false;
    if (info.engine === "mlp") return !!(info.thought && info.thought.intent) && !WEAK[info.thought.intent];
    var tag = intentOf(info.raw, root.MINI_AI_TRANSFORMER);
    return !!tag && !WEAK[tag];
  }

  // ---------- 2–4. карточка «ищу в интернете → верно? → наставнику?» ----------
  var T = {
    ru: { search: "Не уверена в ответе — ищу в интернете…", found: "Нашла в интернете:", src: "Источник", right: "Верно",
          wrong: "Неправильно", ask: "Не нашла правильного ответа. Отправить этот вопрос моему наставнику?",
          offline: "Нет связи с сервером. Отправить вопрос наставнику, когда появится интернет?",
          yes: "Отправить", no: "Не надо", sent: "Отправила наставнику. Его ответ появится здесь, и я его запомню.",
          queued: "Запомнила — отправлю наставнику, когда появится интернет.", saved: "Запомнила.", mentor: "Ответ наставника на «",
          ok: "Понятно, не буду.", wrongQ: "Неправильно?" },
    uz: { search: "Javobga ishonchim yo'q — internetdan qidiryapman…", found: "Internetdan topdim:", src: "Manba", right: "To'g'ri",
          wrong: "Noto'g'ri", ask: "To'g'ri javob topilmadi. Bu savolni ustozimga yuboraymi?",
          offline: "Server bilan aloqa yo'q. Internet bo'lganda ustozga yuboraymi?",
          yes: "Yuborish", no: "Kerak emas", sent: "Ustozga yubordim. Javobi shu yerda chiqadi va eslab qolaman.",
          queued: "Eslab qoldim — internet bo'lganda yuboraman.", saved: "Eslab qoldim.", mentor: "Ustozning javobi: «",
          ok: "Tushundim.", wrongQ: "Noto'g'rimi?" }
  };

  function el(tag, cls, text) { var d = document.createElement(tag); if (cls) d.className = cls; if (text != null) d.textContent = text; return d; }
  function btn(text, primary, fn) { var b = el("button", "learn-btn" + (primary ? " primary" : ""), text); b.type = "button"; b.onclick = fn; return b; }

  function card(host, ui) {
    var c = host.querySelector(".learn");
    if (!c) { c = el("div", "learn"); host.appendChild(c); }
    c.innerHTML = "";
    ui.scroll && ui.scroll();
    return c;
  }

  function askMentor(host, ui, q, a, web, lang) {
    var t = T[lang], c = card(host, ui);
    c.appendChild(el("p", "learn-note", web && web.offline ? t.offline : t.ask));
    var row = el("div", "learn-row");
    row.appendChild(btn(t.yes, true, function () {
      sendMentor(q, a, web, lang).then(function (queued) {
        var c2 = card(host, ui); c2.appendChild(el("p", "learn-note", queued ? t.queued : t.sent));
      });
    }));
    row.appendChild(btn(t.no, false, function () { var c2 = card(host, ui); c2.appendChild(el("p", "learn-note", t.ok)); }));
    c.appendChild(row);
    ui.scroll && ui.scroll();
  }

  function sendMentor(q, a, web, lang) {
    var body = { q: q, a: a, web: web && web.found ? web : null, lang: lang, device: device };
    return call("/api/mentor", body).then(function (r) {
      var p = get(K.pending, []); p.push({ id: r.id, q: q }); set(K.pending, p);
      return false;
    }).catch(function () {
      var p = get(K.pending, []); p.push({ id: null, q: q, body: body }); set(K.pending, p);
      return true;
    });
  }

  function research(host, ui, q, a, lang) {
    var t = T[lang], c = card(host, ui);
    c.appendChild(el("p", "learn-note muted", t.search));
    call("/api/research", { q: q, lang: lang, device: device }).then(function (r) {
      if (!r || !r.found) return askMentor(host, ui, q, a, r, lang);
      var c2 = card(host, ui);
      c2.appendChild(el("p", "learn-note muted", t.found));
      c2.appendChild(el("p", "learn-answer", r.answer));
      var s = el("a", "learn-src", t.src + ": " + (r.title || "Википедия"));
      s.href = r.url; s.target = "_blank"; s.rel = "noopener";
      c2.appendChild(s);
      var row = el("div", "learn-row");
      row.appendChild(btn(t.right, true, function () {
        remember(q, r.answer, r.url);
        // наставник превратит это в ТЕМУ: 8+ формулировок, часть спрячет для экзамена → модель учит смысл
        call("/api/confirm", { q: q, a: r.answer, web: r, lang: lang, device: device }).catch(function () {});
        var c3 = card(host, ui); c3.appendChild(el("p", "learn-note", t.saved));
      }));
      row.appendChild(btn(t.wrong, false, function () { askMentor(host, ui, q, a, r, lang); }));
      c2.appendChild(row);
      ui.scroll && ui.scroll();
    }).catch(function () { askMentor(host, ui, q, a, { found: false, offline: true }, lang); });
  }

  /* Вызывается из app.js после КАЖДОГО ответа.
   * info: {q, answer, raw, lang, engine, thought, memory, host (DOM-узел ответа), scroll()} */
  function afterAnswer(info) {
    var ok = isConfident(info);
    if (!ok && needsWeb(info.q)) {
      // вопрос о незнакомом факте — заготовку модели не показываем, это была бы выдумка
      var txt = info.host.querySelector(".text");
      if (txt) txt.textContent = info.lang === "uz" ? "Buni hali bilmayman." : "Этого я пока не знаю.";
      info.answer = "";
    }
    logQuestion(info.q, info.answer, ok, info.lang);
    if (!ok) return research(info.host, info, info.q, info.answer, info.lang);
    // уверенный ответ — даём тихую ссылку «Неправильно?»
    var meta = info.host.querySelector(".meta");
    if (meta && !meta.querySelector(".learn-wrong")) {
      var w = el("button", "learn-wrong", T[info.lang].wrongQ); w.type = "button";
      w.onclick = function () { w.remove(); research(info.host, info, info.q, info.answer, info.lang); };
      meta.appendChild(w);
    }
  }

  // ---------- 5. ответы наставника ----------
  var onMentorAnswer = null;
  function pollMentor() {
    var p = get(K.pending, []);
    if (!p.length) return;
    // сначала отправляем то, что копилось офлайн
    p.filter(function (x) { return x.id === null; }).forEach(function (x) {
      call("/api/mentor", x.body).then(function (r) {
        var cur = get(K.pending, []);
        cur.forEach(function (y) { if (y.id === null && y.q === x.q) { y.id = r.id; delete y.body; } });
        set(K.pending, cur);
      }).catch(function () {});
    });
    var ids = p.filter(function (x) { return x.id; }).map(function (x) { return x.id; });
    if (!ids.length) return;
    call("/api/mentor/answers?ids=" + ids.join(",")).then(function (ans) {
      var left = [];
      get(K.pending, []).forEach(function (x) {
        var a = x.id && ans[String(x.id)];
        if (a) { remember(x.q, a.answer, "mentor"); onMentorAnswer && onMentorAnswer(x.q, a.answer); }
        else left.push(x);
      });
      set(K.pending, left);
    }).catch(function () {});
  }

  // ---------- 6. новая версия модели с сервера ----------
  function storedModel() {
    var v = get(K.modelV, 0), bundled = (root.MINI_AI_TRANSFORMER && root.MINI_AI_TRANSFORMER.model_version) || 0;
    if (v <= bundled) return null;
    try { var m = JSON.parse(localStorage.getItem(K.modelJson)); return m && m.weights ? m : null; } catch (e) { return null; }
  }

  function checkModel(onUpdated) {
    call("/api/model/latest").then(function (info) {
      var have = Math.max(get(K.modelV, 0), (root.MINI_AI_TRANSFORMER && root.MINI_AI_TRANSFORMER.model_version) || 0);
      if (!info || !info.version || info.version <= have || !info.file) return;
      return call("/models/" + info.file).then(function (m) {
        if (!m || m.tokenizer !== "bpe" || !m.weights) return;
        try { localStorage.setItem(K.modelJson, JSON.stringify(m)); } catch (e) { return; }   // нет места — остаёмся на старой
        set(K.modelV, info.version);
        onUpdated && onUpdated(m, info);
      });
    }).catch(function () {});
  }

  function start(opts) {
    onMentorAnswer = opts.onMentorAnswer;
    discover().then(function () {
      flushOutbox(); pollMentor(); checkModel(opts.onModelUpdated);
    });
    setInterval(pollMentor, 60000);
  }

  function progress() { return call("/api/progress"); }

  // ---------- арифметика: считаем сами, а не «учим» ----------
  // «1+2», «1+2=», «сколько будет 7*8», «5 плюс 3», «12 / 4», «2+2=4?» → просто «3», «56», «8»…
  // Нейронке такое учить бессмысленно (бесконечно много примеров), калькулятор точен всегда.
  var WORD_OPS = [[/(умножить|умножь|умноженное)\s+на|ko'paytir(ilsa|sa)?|×|x(?=\s*\d)|х(?=\s*\d)/g, "*"],
                  [/(разделить|раздели|делить|делённое|деленное)\s+на|bo'lin(sa|gan)?|÷|:/g, "/"],
                  [/плюс|qo'sh(ilsa|sa)?/g, "+"], [/минус|ayir(ilsa|sa)?/g, "-"]];
  function mathAnswer(text, lang) {
    var t = String(text).toLowerCase().replace(/ё/g, "е").replace(/[’ʻʼ`]/g, "'").replace(/,(?=\d)/g, ".");
    t = t.replace(/^(а\s+)?(сколько|скок)\s+(будет|равно|получится)\s*|^(посчитай|вычисли|реши)\s*|^(nechchi|necha)\s+(bo'ladi)?\s*/, "");
    t = t.replace(/\s*(сколько|nechchi|necha|bo'ladi|будет)\s*\??$/, "");
    WORD_OPS.forEach(function (r) { t = t.replace(r[0], r[1]); });
    t = t.replace(/[?!\s]+$/, "");
    var claim = /=\s*(-?\d+(\.\d+)?)$/.exec(t);          // «2+2=5» — пользователь сам написал ответ
    t = t.replace(/=\s*-?\d*(\.\d+)?$/, "").replace(/\s+/g, "");
    if (!/^[\d.+\-*/()]+$/.test(t) || !/\d[+\-*/]|\)[+\-*/]/.test(t) || !/[+\-*/]\(?-?\d/.test(t)) return null;
    var i = 0;
    function num() {
      var m = /^\d+(\.\d+)?/.exec(t.slice(i));
      if (!m) throw 0;
      i += m[0].length; return parseFloat(m[0]);
    }
    function atom() {
      if (t[i] === "-") { i++; return -atom(); }
      if (t[i] === "(") { i++; var v = sum(); if (t[i++] !== ")") throw 0; return v; }
      return num();
    }
    function prod() {
      var v = atom();
      while (t[i] === "*" || t[i] === "/") {
        var op = t[i++], r = atom();
        if (op === "/" && r === 0) throw "zero";
        v = op === "*" ? v * r : v / r;
      }
      return v;
    }
    function sum() {
      var v = prod();
      while (t[i] === "+" || t[i] === "-") { var op = t[i++]; v = op === "+" ? v + prod() : v - prod(); }
      return v;
    }
    try {
      var v = sum();
      if (i !== t.length || !isFinite(v)) return null;
      var r = Math.round(v * 1e6) / 1e6;
      var s = lang === "uz" ? String(r) : String(r).replace(".", ",");
      if (claim) return parseFloat(claim[1]) === r ? (lang === "uz" ? "To'g'ri, " : "Верно, ") + s + "."
                                                   : (lang === "uz" ? "Yo'q, " : "Нет, будет ") + s + ".";
      return s;
    } catch (e) {
      if (e === "zero") return lang === "uz" ? "Nolga bo'lib bo'lmaydi." : "На ноль делить нельзя.";
      return null;
    }
  }

  root.MiniLearn = {
    mathAnswer: mathAnswer,
    progress: progress,
    start: start, afterAnswer: afterAnswer, recall: recall, remember: remember, storedModel: storedModel,
    isConfident: isConfident, needsWeb: needsWeb, intentOf: intentOf, _call: call, _setServer: function (u) { serverUrl = u; set(K.server, u); }
  };
})(typeof window !== "undefined" ? window : this);
