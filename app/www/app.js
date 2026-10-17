/* Mini AI — логика интерфейса. Всё офлайн: модель в model.js, чаты в localStorage. */
(function () {
  "use strict";

  // ===== Название в одном месте — поменяй здесь, и оно изменится везде =====
  var APP_NAME = "Mini AI";

  var $ = function (s) { return document.querySelector(s); };
  var model = window.MINI_AI_MODEL;
  var brain = new window.MiniBrain.Brain(model);
  var STORE = "miniai.chats.v1";

  var app = $("#app"), chat = $("#chat"), msgs = $("#messages"), input = $("#input"),
      form = $("#form"), send = $("#send"), chips = $("#chips");

  document.title = APP_NAME;
  document.querySelectorAll("[data-name]").forEach(function (el) { el.textContent = APP_NAME; });
  document.querySelectorAll("[data-params]").forEach(function (el) { el.textContent = model.n_params.toLocaleString("ru-RU"); });

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
      else { var el = addAI(m.thought); el.querySelector(".text").textContent = m.text; addMeta(el, m.thought, m.q); lastThought = { q: m.q, t: m.thought }; }
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
    var label = thought.intent ? (model.labels[thought.intent] || thought.intent) : "не уверена";
    b.textContent = "🧠 " + label + " · " + Math.round(thought.confidence * 100) + "%";
    b.onclick = function () { openSheet({ q: q, t: thought }); };
    meta.appendChild(b);
    el.querySelector(".text").parentNode.appendChild(meta);
  }

  var busy = false, lastThought = null;

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

    var r = brain.reply(text);           // ← вся «магия»: прямой проход нейросети
    var thought = r.thought;
    lastThought = { q: text, t: thought };
    var delay = window.__FAST__ ? 0 : 450 + Math.random() * 350;

    setTimeout(function () {
      out.textContent = "";
      out.classList.add("caret");
      var chars = Array.from(r.text), i = 0;
      (function tick() {
        if (window.__FAST__) i = chars.length; else i = Math.min(chars.length, i + 2);
        out.textContent = chars.slice(0, i).join("");
        scrollDown();
        if (i < chars.length) return setTimeout(tick, 18);
        out.classList.remove("caret");
        logo.classList.remove("spin");
        addMeta(el, thought, text);
        current.msgs.push({ role: "ai", text: r.text, thought: slim(thought), q: text });
        save();
        busy = false;
        scrollDown();
      })();
    }, delay);
  }

  function slim(t) { return { intent: t.intent, confidence: t.confidence, top3: t.top3, familiar: t.familiar, features: t.features, hidden: t.hidden }; }

  // ---------- шторка «Как я работаю» ----------
  function pct(x) { return (x * 100).toFixed(x > 0.995 || x < 0.1 ? 1 : 0) + "%"; }
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }

  function openSheet(data) {
    data = data || lastThought;
    var body = $("#sheetBody"), html = "";
    var acc = Math.round((model.test_accuracy_chat || model.test_accuracy) * 100);
    if (data) {
      var t = data.t, top = t.top3;
      var verdict = t.intent ? (model.labels[t.intent] || t.intent) : "не знаю (уверенность ниже " + Math.round(model.threshold * 100) + "%)";
      html += '<h3>Твоё сообщение</h3><div class="q">' + esc(data.q) + "</div>";
      html += '<div class="verdict"><b>' + esc(verdict) + "</b><span>уверенность " + pct(t.confidence) + "</span></div>";
      html += "<h3>Топ-3 намерения</h3>";
      top.forEach(function (p, k) {
        html += '<div class="bar' + (k ? " rest" : "") + '"><div class="bar-top"><span>' + esc(model.labels[p[0]] || p[0]) + ' <small>' + p[0] + "</small></span><b>" + pct(p[1]) + '</b></div><div class="bar-track"><div class="bar-fill" style="width:' + (p[1] * 100).toFixed(1) + '%"></div></div></div>';
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
      html += "<p>Напиши мне что-нибудь — и здесь появится, что именно подумала нейросеть: какое намерение она узнала и насколько уверена.</p>";
    }
    html += "<h3>Архитектура</h3><div class=\"layers\">" +
      '<div class="layer"><b>' + model.dim + "</b><small>вход<br>слова + триграммы</small></div><span class=\"arrow\">→</span>" +
      '<div class="layer"><b>' + model.hidden + "</b><small>скрытый слой<br>ReLU</small></div><span class=\"arrow\">→</span>" +
      '<div class="layer"><b>' + model.intents.length + "</b><small>намерений<br>softmax</small></div></div>";
    html += "<p>" + model.n_params.toLocaleString("ru-RU") + " параметров · точность на тесте ≈ " + acc + "% · написано с нуля на numpy (backprop + Adam) · работает без интернета.</p>";
    html += '<p class="small-note">Автор: Фарход. Это не ChatGPT — это крошечная сеть-классификатор: она выбирает тип фразы и отвечает заготовкой.</p>';
    body.innerHTML = html;
    app.classList.add("sheet-open");
    $("#sheet").setAttribute("aria-hidden", "false");
  }
  function closeSheet() { app.classList.remove("sheet-open"); $("#sheet").setAttribute("aria-hidden", "true"); }

  // ---------- меню ----------
  function openDrawer() { renderList(); app.classList.add("drawer-open"); }
  function closeDrawer() { app.classList.remove("drawer-open"); }

  $("#menuBtn").onclick = openDrawer;
  $("#scrim").onclick = closeDrawer;
  $("#newBtn").onclick = newChat;
  $("#drawerNew").onclick = function () { newChat(); closeDrawer(); };
  $("#drawerHow").onclick = function () { closeDrawer(); openSheet(); };
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

  window.MiniApp = { ask: ask, openSheet: openSheet, openDrawer: openDrawer, newChat: newChat, closeSheet: closeSheet };
  renderList();
})();
