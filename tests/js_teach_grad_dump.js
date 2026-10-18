// tests/js_teach_grad_dump.js
// Берёт на stdin JSON {model: <экспорт model_transformer.json-формата>, q, a}
// Считает forward+backward (как lossAndGradsOne с trainFrom=0, т.е. "обучаем всё")
// для ОДНОГО примера (B=1) — ровно то же самое, что делает Python тест для сверки.
// Печатает в stdout JSON {loss, grads: {key: [...]}} для запрошенных ключей.
const core = require("./../app/www/teach_core.js");

const input = JSON.parse(require("fs").readFileSync(0, "utf8"));
const m = input.model;
const p = core.loadParamsF64(m);
const cfg = m.config;
const stoi = {};
for (let i = 0; i < m.vocab.length; i++) stoi[m.vocab[i]] = i;

const keys = input.keys || Object.keys(p);
const grads = core.zerosLike(p, keys);
const r = core.lossAndGradsOne(p, cfg, stoi, { q: input.q, a: input.a }, grads, 0, input.totalTok);

const out = { loss: r.loss, nTok: r.nTok, grads: {} };
keys.forEach((k) => { out.grads[k] = Array.from(grads[k]); });
process.stdout.write(JSON.stringify(out));
