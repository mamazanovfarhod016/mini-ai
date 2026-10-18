// Сверка Python<->JS для BPE: node tests/js_bpe_encode.js <model.json> [raw] < texts.json
// stdin: JSON-массив строк (уже clean_question'ed или сырые ответы с ①②, либо произвольный
// текст/код при raw-режиме).
// stdout: JSON [{text, ids, decoded}, ...]
const fs = require("fs");
const path = require("path");
const MiniAIBPE = require(path.join(__dirname, "..", "app", "www", "bpe.js"));

const modelPath = process.argv[2];
const raw = process.argv[3] === "raw";
const m = JSON.parse(fs.readFileSync(modelPath, "utf8"));
const tok = new MiniAIBPE.BPETokenizer({ vocab: m.vocab, merges: m.merges || [] });

const texts = JSON.parse(fs.readFileSync(0, "utf8"));
const out = texts.map((t) => {
  const ids = tok.encode(t, raw);
  return { text: t, ids: ids, decoded: tok.decode(ids, raw) };
});
process.stdout.write(JSON.stringify(out));
