/* Mini AI — логика интерфейса. Всё офлайн: модели в model.js/model_transformer.js, чаты в localStorage.
 * Два «мозга»:
 *   - transformer (v0.2, по умолчанию) — настоящая генерация текста по буквам (app/www/transformer.js)
 *   - mlp          (v0.1, классификатор) — ответы-заготовки (app/www/brain.js)
 * Переключатель — в боковом меню, выбор сохраняется в localStorage.
 */
(function () {
  "use strict";

  var APP_NAME = "Mini AI";

  var $ = function (s) { return document.querySelector(s); };
  var STORE = "miniai.chats.v1";
  var ENGINE_KEY = "miniai.engine";

  var app = $("#app"), chat = $("#chat"), msgs = $("#messages"), input = $("#input"),
      form = $("#form"), send = $("#send"), chips = $("#chips");

  document.title = APP_NAME;
  document.querySelectorAll("[data-name]").forEach(function (el) { el.textContent = APP_NAME; });

  // ---------- движки ----------
  var mlpModel = window.MINI_AI_MODEL;
  var mlpBrain = new window.MiniBrain.Brain(mlpModel);
  // Новая версия модели, скачанная с сервера (прошла экзамен) — важнее встроенной.
  if (window.MiniLearn && window.MiniLearn.storedModel()) window.MINI_AI_TRANSFORMER = window.MiniLearn.storedModel();
  var txModel = window.MINI_AI_TRANSFORMER ? new window.MiniTransformer.TransformerModel(window.MINI_AI_TRANSFORMER) : null;

  var UNKNOWN = {
    ru: "Я пока маленькая нейронка и этого не знаю 🙈 Попробуй что-то попроще: «привет», «как дела», «кто ты».",
    uz: "Men hali kichkina neyron tarmoqman va buni bilmayman 🙈 Oddiyroq so'rang: «salom», «qalaysan»."
  };

  function isLatinText(t) { return window.MiniBrain.isLatin(t); }

  // 18.10.2026: BPE-трансформер (v0.3) точнее классификатора на новых фразах → он по умолчанию;
  // классификатор v0.1 — только если пользователь сам выбрал его в меню.
  var engine = (localStorage.getItem(ENGINE_KEY) !== "mlp" && txModel) ? "transformer" : "mlp";

  function paramsOfCurrentEngine() {
    return engine === "transformer" && txModel ? window.MINI_AI_TRANSFORMER.n_params : mlpModel.n_params;
  }
  function updateParamsLabel() {
    document.querySelectorAll("[data-params]").forEach(function (el) { el.textContent = paramsOfCurrentEngine().toLocaleString("ru-RU"); });
  }
  updateParamsLabel();

  function setEngine(name) {
    if (name === "transformer" && !txModel) name = "mlp";
    engine = name;
    localStorage.setItem(ENGINE_KEY, name);
    updateParamsLabel();
    renderEngineSwitch();
  }

  function renderEngineSwitch() {
    var el = $("#engineSwitch");
    if (!el) return;
    el.querySelectorAll(".engine-opt").forEach(function (b) {
      var active = b.dataset.engine === engine;
      b.classList.toggle("active", active);
      b.disabled = b.dataset.engine === "transformer" && !txModel;
    });
  }
  $("#engineSwitch") && $("#engineSwitch").querySelectorAll(".engine-opt").forEach(function (b) {
    b.onclick = function () { setEngine(b.dataset.engine); closeDrawer(); };
  });
  renderEngineSwitch();

  // ---------- тёмная тема по системной настройке ----------
  var mq = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
  function applyTheme() {
    var forced = new URLSearchParams(location.search).get("theme");
    var dark = forced ? forced === "dark" : !!(mq && mq.matches);
    document.documentElement.classList.toggle("dark", dark);
    try { window.AndroidApp && window.AndroidApp.setTheme(dark); } catch (e) {}
  }
  applyTheme();
  if (mq && mq.addEventListener) mq.addEventListener("change", applyTheme);

  // ---------- приветствие по времени суток (часы телефона) ----------
  var hr = new Date().getHours();
  $("#helloTime").textContent = hr < 5 ? "Доброй ночи" : hr < 12 ? "Доброе утро" : hr < 18 ? "Добрый день" : "Добрый вечер";

  // ---------- подсказки-чипы ----------
  ["Привет", "Как дела?", "Кто ты?", "Salom", "Расскажи шутку", "Кто тебя создал?", "Который час?"].forEach(function (t) {
    var b = document.createElement("button");
    b.className = "chip"; b.type = "button"; b.textContent = t;
    b.onclick = function () { ask(t); };
    chips.appendChild(b);
  });

  // ---------- хранилище чатов ----------
  function load() { try { return JSON.parse(localStorage.getItem(STORE)) || []; } catch (e) { return []; } }
  function save() { try { localStorage.setItem(STORE, JSON.stringify(chats.slice(0, 50))); } catch (e) {} }
  var chats = load();
  var current = null;   // {id, title, ts, msgs:[{role, text, thought}]}

  function newChat() {
    current = null;
    msgs.innerHTML = "";
    app.classList.remove("has-msgs");
    lastThought = null;
    renderList();
  }

  function openChat(id) {
    current = chats.find(function (c) { return c.id === id; }) || null;
    msgs.innerHTML = "";
    if (!current) return newChat();
    current.msgs.forEach(function (m) {
      if (m.role === "user") addUser(m.text);
      else { var el = addAI(); el.querySelector(".text").textContent = m.text; addMeta(el, m.thought, m.q); lastThought = { q: m.q, t: m.thought }; }
    });
    app.classList.toggle("has-msgs", current.msgs.length > 0);
    renderList();
    scrollDown();
  }

  function renderList() {
    var list = $("#chatList");
    list.innerHTML = "";
    if (!chats.length) { list.innerHTML = '<div class="chat-empty">Пока нет чатов</div>'; return; }
    chats.forEach(function (c) {
      var b = document.createElement("button");
      b.className = "chat-item" + (current && current.id === c.id ? " active" : "");
      var d = new Date(c.ts);
      b.textContent = c.title;
      var s = document.createElement("small");
      s.textContent = d.toLocaleDateString("ru-RU", { day: "numeric", month: "short" }) + " · " + c.msgs.length + " сообщ.";
      b.appendChild(s);
      b.onclick = function () { openChat(c.id); closeDrawer(); };
      list.appendChild(b);
    });
  }

  // ---------- сообщения ----------
  function scrollDown() { chat.scrollTop = chat.scrollHeight; }

  function addUser(text) {
    var d = document.createElement("div");
    d.className = "msg-user"; d.textContent = text;
    msgs.appendChild(d);
    return d;
  }

  function addAI() {
    var d = document.createElement("div");
    d.className = "msg-ai";
    d.innerHTML = '<span class="logo"></span><div><div class="text"></div></div>';
    msgs.appendChild(d);
    return d;
  }

  function addMeta(el, thought, q) {
    var meta = document.createElement("div");
    meta.className = "meta";
    var b = document.createElement("button");
    if (thought && thought.engine === "memory") {
      b.textContent = thought.source === "mentor" ? "выучено от наставника" : "выучено из интернета";
      b.onclick = function () {};
      meta.appendChild(b);
      el.querySelector(".text").parentNode.appendChild(meta);
      return;
    }
    if (thought && thought.engine === "transformer") {
      b.textContent = "✨ трансформер · нажми, чтобы увидеть внимание";
    } else {
      var label = thought && thought.intent ? (mlpModel.labels[thought.intent] || thought.intent) : "не уверена";
      b.textContent = "🧠 " + label + (thought ? " · " + Math.round(thought.confidence * 100) + "%" : "");
    }
    b.onclick = function () { openSheet({ q: q, t: thought }); };
    meta.appendChild(b);
    el.querySelector(".text").parentNode.appendChild(meta);
  }

  var busy = false, lastThought = null;

  function addMentorAnswer(q, answer) {
    if (!current) { current = { id: String(Date.now()), title: q.slice(0, 40), ts: Date.now(), msgs: [] }; chats.unshift(current); }
    app.classList.add("has-msgs");
    var lang = isLatinText(q) ? "uz" : "ru";
    var el = addAI();
    el.querySelector(".text").textContent = answer;
    var meta = document.createElement("div"); meta.className = "meta";
    var chip = document.createElement("button");
    chip.textContent = (lang === "uz" ? "ustoz javobi · «" : "ответ наставника · «") + (q.length > 28 ? q.slice(0, 28) + "…" : q) + "»";
    meta.appendChild(chip); el.querySelector(".text").parentNode.appendChild(meta);
    current.msgs.push({ role: "ai", text: answer, thought: { engine: "memory", source: "mentor" }, q: q });
    save(); renderList(); scrollDown();
  }
  if (window.MiniLearn) window.MiniLearn.start({ onMentorAnswer: addMentorAnswer });

  // ---------- трансформер: пошаговая генерация с реальным стримингом ----------
  // lastGen хранит «сырые» данные последнего сгенерированного ответа (внимание по шагам)
  // для тепловой карты. Не сохраняется в localStorage — слишком тяжело, и это нормально:
  // карта внимания доступна для последнего ответа в текущей сессии.
  var lastGen = null;

  function streamTransformer(question, onChar, onDone) {
    var ids = txModel.promptIds(question);
    var qLen = ids.length;                     // <bos> + вопрос + <sep>
    var cache = txModel.newCache();
    var last = null;
    for (var i = 0; i < ids.length; i++) last = txModel.step(ids[i], i, cache, i === ids.length - 1);
    var pos = ids.length, outIds = [], steps = [];
    var block = txModel.cfg.block;
    var fast = !!window.__FAST__;
    // Пары ids → текст собираем через decode(ВСЕХ id), а не склейкой по одному:
    // у BPE пробел живёт внутри токена («▁слово»), decode одного токена его срезает.
    var qIds = ids.slice(1, -1);

    function tick() {
      if (pos >= block) return onDone(txModel.decode(outIds), steps);
      var nxt = txModel.pickNext(last.logits, 0);
      if (nxt === window.MiniTransformer.EOS) return onDone(txModel.decode(outIds), steps);
      var ch = txModel.piece(nxt);
      var topk = txModel.topK(last.logits, 5);
      outIds.push(nxt);
      steps.push({ ch: ch, attn: last.attn, topk: topk, qLen: qLen, qIds: qIds });
      onChar(ch, steps[steps.length - 1], txModel.decode(outIds));
      last = txModel.step(nxt, pos, cache, true);
      pos++;
      setTimeout(tick, fast ? 0 : 26 + Math.random() * 24);
    }
    tick();
  }

  // Сентинелы даты/времени: модель трансформера (v0.2) обучена писать ОДИН фиксированный
  // символ-плейсхолдер вместо реальных цифр (баг-фикс 18.10: раньше она должна была сама
  // «угадывать» произвольные цифры и иногда выдумывала их — «сегодня 0.0.2026»).
  // Эти символы должны совпадать с TIME_TOK/DATE_TOK в mini_ai/transformer.py.
  var TIME_TOK = "\u2460";   // ①
  var DATE_TOK = "\u2461";   // ②

  function fillPlaceholders(text, now) {
    function pad(n) { return (n < 10 ? "0" : "") + n; }
    return text
      .split(TIME_TOK).join(pad(now.getHours()) + ":" + pad(now.getMinutes()))
      .split(DATE_TOK).join(pad(now.getDate()) + "." + pad(now.getMonth() + 1) + "." + now.getFullYear());
  }

  function ask(text) {
    text = (text || "").trim();
    if (!text || busy) return;
    busy = true;
    input.value = ""; autosize(); updateSend();
    app.classList.add("has-msgs");
    if (!current) {
      current = { id: String(Date.now()), title: text.slice(0, 40), ts: Date.now(), msgs: [] };
      chats.unshift(current);
    }
    addUser(text);
    current.msgs.push({ role: "user", text: text });
    current.ts = Date.now();
    save(); renderList();

    var el = addAI();
    var logo = el.querySelector(".logo");
    var out = el.querySelector(".text");
    logo.classList.add("spin");
    out.innerHTML = '<span class="typing">печатает <span class="dots"><i></i><i></i><i></i></span></span>';
    scrollDown();

    var lang = isLatinText(text) ? "uz" : "ru";
    var fast = !!window.__FAST__;

    function finish(finalText, thought, raw, fromMemory) {
      out.classList.remove("caret");
      logo.classList.remove("spin");
      addMeta(el, thought, text);
      current.msgs.push({ role: "ai", text: finalText, thought: thought, q: text });
      save();
      busy = false;
      scrollDown();
      if (window.MiniLearn) window.MiniLearn.afterAnswer({
        q: text, answer: finalText, raw: raw != null ? raw : finalText, lang: lang, engine: thought && thought.engine,
        thought: thought, memory: !!fromMemory, host: el.querySelector(".text").parentNode, scroll: scrollDown
      });
    }

    // Арифметика — считаем сами, коротко: «1+2» → «3».
    var calc = window.MiniLearn && window.MiniLearn.mathAnswer(text, lang);
    if (calc != null) {
      setTimeout(function () {
        out.textContent = calc;
        out.classList.remove("caret"); logo.classList.remove("spin");
        current.msgs.push({ role: "ai", text: calc, thought: { engine: "calc" }, q: text });
        save(); busy = false; scrollDown();
      }, fast ? 0 : 250);
      return;
    }

    // Выученное (от наставника / подтверждено в интернете) — отвечаем сразу, до ночного дообучения.
    var mem = window.MiniLearn && window.MiniLearn.recall(text);
    if (mem) {
      setTimeout(function () {
        out.textContent = mem.a;
        finish(mem.a, { engine: "memory", answer: mem.a, source: mem.source }, mem.a, true);
      }, fast ? 0 : 350);
      return;
    }

    if (engine === "transformer" && txModel) {
      var startDelay = fast ? 0 : 280 + Math.random() * 220;
      setTimeout(function () {
        out.textContent = "";
        out.classList.add("caret");
        streamTransformer(text, function (ch, step, soFar) {
          out.textContent = fillPlaceholders(soFar, new Date());
          scrollDown();
        }, function (finalText, steps) {
          var rawText = finalText;
          finalText = fillPlaceholders(finalText, new Date());
          out.textContent = finalText;
          lastGen = { question: text, cleaned: window.MiniTransformer.cleanQuestion(text), answer: finalText, steps: steps, engine: "transformer" };
          var thought = { engine: "transformer", answer: finalText };
          lastThought = { q: text, t: thought };
          finish(finalText, thought, rawText);
        });
      }, startDelay);
    } else {
      var r = mlpBrain.reply(text);
      var thought = r.thought; thought.engine = "mlp";
      lastThought = { q: text, t: thought };
      var delay = fast ? 0 : 450 + Math.random() * 350;
      setTimeout(function () {
        out.textContent = "";
        out.classList.add("caret");
        var chars = Array.from(r.text), i = 0;
        (function tick2() {
          if (fast) i = chars.length; else i = Math.min(chars.length, i + 2);
          out.textContent = chars.slice(0, i).join("");
          scrollDown();
          if (i < chars.length) return setTimeout(tick2, 18);
          finish(r.text, thought);
        })();
      }, delay);
    }
  }

  // ---------- шторка «Как я работаю» ----------
  function pct(x) { return (x * 100).toFixed(x > 0.995 || x < 0.1 ? 1 : 0) + "%"; }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }

  function openSheet(data) {
    $("#sheetTitle").textContent = "Как я работаю";
    data = data || lastThought;
    if (data && data.t && data.t.engine === "transformer") return openSheetTransformer(data);
    return openSheetMLP(data);
  }

  function openSheetMLP(data) {
    var body = $("#sheetBody"), html = "";
    var acc = Math.round((mlpModel.test_accuracy_chat || mlpModel.test_accuracy) * 100);
    html += engineBadgeHtml();
    if (data && data.t) {
      var t = data.t, top = t.top3;
      var verdict = t.intent ? (mlpModel.labels[t.intent] || t.intent) : "не знаю (уверенность ниже " + Math.round(mlpModel.threshold * 100) + "%)";
      html += '<h3>Твоё сообщение</h3><div class="q">' + esc(data.q) + "</div>";
      html += '<div class="verdict"><b>' + esc(verdict) + "</b><span>уверенность " + pct(t.confidence) + "</span></div>";
      html += "<h3>Топ-3 намерения</h3>";
      top.forEach(function (p, k) {
        html += '<div class="bar' + (k ? " rest" : "") + '"><div class="bar-top"><span>' + esc(mlpModel.labels[p[0]] || p[0]) + ' <small>' + p[0] + "</small></span><b>" + pct(p[1]) + '</b></div><div class="bar-track"><div class="bar-fill" style="width:' + (p[1] * 100).toFixed(1) + '%"></div></div></div>';
      });
      if (t.hidden) {
        var mx = Math.max.apply(null, t.hidden.concat([1e-9])), on = t.hidden.filter(function (v) { return v > 0; }).length;
        html += "<h3>Скрытый слой: " + on + " из " + t.hidden.length + " нейронов активны</h3><div class=\"neurons\">";
        t.hidden.forEach(function (v) { html += '<i style="opacity:' + (0.08 + 0.92 * Math.max(0, v) / mx).toFixed(2) + '"></i>'; });
        html += "</div>";
      }
      var toks = window.MiniBrain.tokens(data.q).filter(function (f) { return f[0] === "w" || f[0] === "c"; }).slice(0, 18);
      html += "<h3>Как я вижу фразу (" + (t.features || toks.length) + " признаков)</h3><div class=\"tags\">" +
        toks.map(function (f) { return "<code>" + esc(f.slice(2).replace(/ /g, "␣")) + "</code>"; }).join("") + "</div>";
      html += '<p class="small-note">Знакомых кусочков: ' + Math.round(t.familiar * 100) + "%. Если мало — честно говорю «не знаю».</p>";
    } else {
      html += "<p>Напиши мне что-нибудь — и здесь появится, что именно подумала нейросеть.</p>";
    }
    html += "<h3>Архитектура</h3><div class=\"layers\">" +
      '<div class="layer"><b>' + mlpModel.dim + "</b><small>вход<br>слова + триграммы</small></div><span class=\"arrow\">→</span>" +
      '<div class="layer"><b>' + mlpModel.hidden + "</b><small>скрытый слой<br>ReLU</small></div><span class=\"arrow\">→</span>" +
      '<div class="layer"><b>' + mlpModel.intents.length + "</b><small>намерений<br>softmax</small></div></div>";
    html += "<p>" + mlpModel.n_params.toLocaleString("ru-RU") + " параметров · точность на тесте ≈ " + acc + "% · написано с нуля на numpy (backprop + Adam) · работает без интернета.</p>";
    html += '<p class="small-note">Автор: Фарход. Это классификатор (v0.1): он выбирает тип фразы и отвечает заготовкой, а не пишет текст сам.</p>';
    body.innerHTML = html;
    app.classList.add("sheet-open");
    $("#sheet").setAttribute("aria-hidden", "false");
  }

  function engineBadgeHtml() {
    return '<div class="engine-badge">' +
      '<button data-e="transformer"' + (engine === "transformer" ? " class=\"active\"" : "") + (txModel ? "" : " disabled") + '>Трансформер v0.2</button>' +
      '<button data-e="mlp"' + (engine === "mlp" ? " class=\"active\"" : "") + '>Классификатор v0.1</button></div>';
  }

  function wireEngineBadge() {
    var body = $("#sheetBody");
    body.querySelectorAll(".engine-badge button").forEach(function (b) {
      b.onclick = function () { setEngine(b.dataset.e); };
    });
  }

  // ---------- тепловая карта внимания (для трансформера) ----------
  var heatState = { layer: 0, head: 0, stepIdx: 0 };

  function openSheetTransformer(data) {
    var body = $("#sheetBody"), m = window.MINI_AI_TRANSFORMER, cfg = m.config;
    var html = engineBadgeHtml();
    var gen = (data && data.t && lastGen && lastGen.question === data.q) ? lastGen : null;

    html += "<h3>Твой вопрос</h3><div class=\"q\">" + esc(data.q || "") + "</div>";
    if (data && data.t && data.t.answer) {
      html += "<h3>Ответ модели</h3><div class=\"q\">" + esc(data.t.answer) + "</div>";
    }

    if (gen && gen.steps.length) {
      heatState.stepIdx = Math.min(heatState.stepIdx, gen.steps.length - 1);
      heatState.layer = Math.min(heatState.layer, cfg.n_layer - 1);
      heatState.head = Math.min(heatState.head, cfg.n_head - 1);

      html += "<h3>Тепловая карта внимания</h3>";
      html += '<p class="small-note">На какие буквы твоего вопроса смотрела модель, когда писала выбранную букву ответа. Светлее фон — больше внимания.</p>';
      html += '<div class="heat-controls">';
      html += '<div class="heat-row"><span>Слой</span><div class="seg" id="heatLayer">';
      for (var l = 0; l < cfg.n_layer; l++) html += '<button data-v="' + l + '"' + (l === heatState.layer ? ' class="active"' : '') + '>' + (l + 1) + '</button>';
      html += '</div></div>';
      html += '<div class="heat-row"><span>Голова</span><div class="seg" id="heatHead">';
      for (var h = 0; h < cfg.n_head; h++) html += '<button data-v="' + h + '"' + (h === heatState.head ? ' class="active"' : '') + '>' + (h + 1) + '</button>';
      html += '</div></div>';
      html += '</div>';

      html += '<p class="small-note">Буква ответа (нажми на другую, чтобы посмотреть внимание в её момент):</p>';
      html += '<div class="answer-chars" id="answerChars">';
      gen.steps.forEach(function (s, i) {
        html += '<button data-i="' + i + '"' + (i === heatState.stepIdx ? ' class="active"' : '') + '>' + esc(s.ch === " " ? "␣" : s.ch) + '</button>';
      });
      html += '</div>';

      html += '<div class="heatmap" id="heatmap"></div>';

      html += '<h3>Топ-5 вероятностей следующей буквы</h3><div class="topk" id="topk"></div>';
    } else {
      html += '<p class="small-note">Тепловая карта доступна для последнего ответа, написанного в этой сессии (внимание не сохраняется между перезапусками — слишком много данных).</p>';
    }

    html += '<h3>Архитектура</h3><div class="tf-diagram">' +
      '<div class="tf-row"><b>' + (txModel.tokenizerType === "bpe" ? "кусочки слов вопроса (BPE, словарь " + cfg.V + ")" : "символы вопроса") + '</b></div>' +
      '<div class="tf-arrow">↓ эмбеддинг символа + позиции</div>' +
      '<div class="tf-block">×' + cfg.n_layer + ' блока:<br>LayerNorm → внимание (' + cfg.n_head + ' головы) → +residual<br>LayerNorm → MLP (GELU) → +residual</div>' +
      '<div class="tf-arrow">↓ LayerNorm → умножение на эмбеддинги</div>' +
      '<div class="tf-row"><b>softmax → вероятность следующего ' + (txModel.tokenizerType === "bpe" ? "кусочка" : "символа") + '</b></div></div>';
    html += '<p>' + m.n_params.toLocaleString("ru-RU") + ' параметров · d_model=' + cfg.d + ' · контекст=' + cfg.block +
      ' · perplexity на тесте ≈ ' + (m.test_ppl || "?") + ' · точность (точн. совпадение) ≈ ' + Math.round((m.test_exact_accuracy || 0) * 100) + '%' + (txModel.tokenizerType === "bpe" ? ' · свой BPE-токенизатор (написан с нуля)' : '') + '.</p>';
    html += '<p class="small-note">Автор: Фарход. Это настоящая генерация текста по буквам (как крошечный GPT), написанная с нуля на numpy — не выбор заготовки.</p>';

    body.innerHTML = html;
    wireEngineBadge();
    app.classList.add("sheet-open");
    $("#sheet").setAttribute("aria-hidden", "false");

    if (gen && gen.steps.length) {
      wireHeatControls(gen);
      renderHeatmap(gen);
    }
  }

  function wireHeatControls(gen) {
    $("#heatLayer") && $("#heatLayer").querySelectorAll("button").forEach(function (b) {
      b.onclick = function () { heatState.layer = +b.dataset.v; refreshHeatButtons(); renderHeatmap(gen); };
    });
    $("#heatHead") && $("#heatHead").querySelectorAll("button").forEach(function (b) {
      b.onclick = function () { heatState.head = +b.dataset.v; refreshHeatButtons(); renderHeatmap(gen); };
    });
    $("#answerChars") && $("#answerChars").querySelectorAll("button").forEach(function (b) {
      b.onclick = function () { heatState.stepIdx = +b.dataset.i; refreshHeatButtons(); renderHeatmap(gen); };
    });
  }

  function refreshHeatButtons() {
    [["#heatLayer", "layer"], ["#heatHead", "head"], ["#answerChars", "stepIdx"]].forEach(function (pair) {
      var el = $(pair[0]); if (!el) return;
      el.querySelectorAll("button").forEach(function (b) {
        var val = pair[1] === "stepIdx" ? +b.dataset.i : +b.dataset.v;
        b.classList.toggle("active", val === heatState[pair[1]]);
      });
    });
  }

  function renderHeatmap(gen) {
    var step = gen.steps[heatState.stepIdx];
    var hm = $("#heatmap"), tk = $("#topk");
    if (!step || !hm) return;
    var attn = step.attn && step.attn[heatState.layer];   // Float64Array(H * T)
    var qLen = step.qLen;
    // кусочки вопроса (без <bos>/<sep>): буквы для char-модели, BPE-токены для bpe-модели
    var qChars = step.qIds ? step.qIds.map(function (id) { return txModel.piece(id).trim() || "␣"; })
                           : Array.from(gen.cleaned);
    var html = "";
    if (attn) {
      var T = attn.length / txModel.cfg.n_head;
      var head = heatState.head;
      var weights = [];
      var mx = 1e-9;
      // позиции 1..qLen-2 — символы вопроса (0 = <bos>, qLen-1 = <sep>)
      for (var i = 0; i < qChars.length && i + 1 < qLen - 1; i++) {
        var w = attn[head * T + (i + 1)] || 0;
        weights.push(w);
        if (w > mx) mx = w;
      }
      html += '<div class="heat-strip">';
      for (i = 0; i < weights.length; i++) {
        var op = 0.06 + 0.9 * (weights[i] / mx);
        html += '<span style="background:rgba(217,119,87,' + op.toFixed(2) + ')">' + esc(qChars[i] === " " ? "␣" : qChars[i]) + '</span>';
      }
      html += '</div>';
    } else {
      html = '<p class="small-note">Нет данных внимания для этого шага.</p>';
    }
    hm.innerHTML = html;

    if (tk) {
      var topkHtml = "";
      step.topk.forEach(function (t) {
        var ch = t.ch.length === 1 ? t.ch : t.ch;
        topkHtml += '<div class="bar rest"><div class="bar-top"><span><code>' + esc(ch === " " ? "␣" : ch) + '</code></span><b>' + pct(t.p) + '</b></div><div class="bar-track"><div class="bar-fill" style="width:' + (t.p * 100).toFixed(1) + '%"></div></div></div>';
      });
      tk.innerHTML = topkHtml;
    }
  }

  function closeSheet() { app.classList.remove("sheet-open"); $("#sheet").setAttribute("aria-hidden", "true"); }

  // ---------- меню ----------
  function openDrawer() { renderList(); renderEngineSwitch(); app.classList.add("drawer-open"); }
  function closeDrawer() { app.classList.remove("drawer-open"); }

  $("#menuBtn").onclick = openDrawer;
  $("#scrim").onclick = closeDrawer;
  $("#newBtn").onclick = newChat;
  $("#drawerNew").onclick = function () { newChat(); closeDrawer(); };
  $("#drawerHow").onclick = function () { closeDrawer(); openSheet(); };

  // ---------- «Как я умнею»: экзамены на вопросах, которых модель не видела ----------
  function openGrowth() {
    var body = $("#sheetBody");
    $("#sheetTitle").textContent = "Как я умнею";
    body.innerHTML = "<h3>Как я умнею</h3><p class=\"small-note\">Загружаю историю экзаменов…</p>";
    app.classList.add("sheet-open"); $("#sheet").setAttribute("aria-hidden", "false");
    var m = window.MINI_AI_TRANSFORMER || {};
    var render = function (hist) {
      var html = "<h3>Экзамены на новом <span class=\"hint\" title=\"Каждую ночь я сдаю экзамен на вопросах, которых никогда не видела. Новая версия ставится, только если сдала.\">!</span></h3>";
      html += "<p class=\"small-note\">Экзамен — только на формулировках, которых я <b>не видела</b> при обучении. Так видно, что я понимаю смысл, а не зубрю.</p>";
      html += "<div class=\"grow-now\"><div><b>" + Math.round((m.test_intent_accuracy || 0) * 100) + "%</b><small>понимаю старые темы</small></div>" +
        "<div><b>" + (m.new_topics_accuracy == null ? "—" : Math.round(m.new_topics_accuracy * 100) + "%") + "</b><small>понимаю новые темы</small></div>" +
        "<div><b>" + (m.learned_topics || 0) + "</b><small>выучено тем</small></div></div>";
      var ok = (hist || []).filter(function (h) { return h.kind !== "rejected"; });
      if (ok.length) {
        var W = 300, H = 120, n = ok.length;
        var x = function (i) { return n === 1 ? W / 2 : 10 + i * (W - 20) / (n - 1); };
        var y = function (v) { return H - 10 - v * (H - 20); };
        var line = function (key) { return ok.map(function (h, i) { return h[key] == null ? "" : (i ? "L" : "M") + x(i).toFixed(1) + "," + y(h[key]).toFixed(1); }).join(""); };
        html += "<svg class=\"grow-chart\" viewBox=\"0 0 " + W + " " + H + "\"><path d=\"" + line("base") + "\" class=\"g-base\"/><path d=\"" + line("new") + "\" class=\"g-new\"/>" +
          ok.map(function (h, i) { return h.new == null ? "" : "<circle cx=\"" + x(i) + "\" cy=\"" + y(h.new) + "\" r=\"3\" class=\"g-dot\"/>"; }).join("") + "</svg>";
        html += "<p class=\"small-note\"><span class=\"key g-new\"></span> новые темы · <span class=\"key g-base\"></span> старые темы</p>";
      }
      html += "<h3>Экзамены</h3>";
      if (!hist || !hist.length) html += "<p class=\"small-note\">Пока не было — первый пройдёт ночью, когда появятся новые знания.</p>";
      (hist || []).slice().reverse().slice(0, 10).forEach(function (h) {
        var verdict = h.kind === "rejected" ? "не сдала — осталась прежняя версия" : (h.kind === "full" ? "переобучилась с нуля, v" : "дообучилась, v") + (h.kind === "rejected" ? "" : h.version);
        html += "<div class=\"grow-row\"><span>" + esc(h.at) + "</span><span>" + esc(verdict) + "</span><b>" +
          (h.new_total ? h.new_hits + "/" + h.new_total + " новых" : "") + "</b></div>";
      });
      body.innerHTML = html;
    };
    if (window.MiniLearn) window.MiniLearn.progress().then(render).catch(function () { render(null); });
    else render(null);
  }
  $("#drawerGrow").onclick = function () { closeDrawer(); openGrowth(); };
  $("#howBtn").onclick = function () { openSheet(); };
  $("#modelBtn").onclick = function () { openSheet(); };
  $("#sheetScrim").onclick = closeSheet;

  // ---------- ввод ----------
  function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 140) + "px"; }
  function updateSend() { send.disabled = !input.value.trim(); }
  input.addEventListener("input", function () { autosize(); updateSend(); });
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(input.value); }
  });
  form.addEventListener("submit", function (e) { e.preventDefault(); ask(input.value); });

  // Кнопка «назад» на Android: сначала закрываем шторку/меню
  window.miniBack = function () {
    if (app.classList.contains("sheet-open")) { closeSheet(); return 1; }
    if (app.classList.contains("drawer-open")) { closeDrawer(); return 1; }
    return 0;
  };

  window.MiniApp = { ask: ask, openSheet: openSheet, openDrawer: openDrawer, newChat: newChat, closeSheet: closeSheet, setEngine: setEngine };
  renderList();
})();
