// Сверка Python<->JS для BPE: node tests/js_bpe_encode.js <model.json> < texts.json
// stdin: JSON-массив строк (уже clean_question'ed или сырые ответы с ①②).
// stdout: JSON [{text, ids, decoded}, ...]
const fs = require("fs");
const path = require("path");
const MiniAIBPE = require(path.join(__dirname, "..", "app", "www", "bpe.js"));

const modelPath = process.argv[2];
const m = JSON.parse(fs.readFileSync(modelPath, "utf8"));
const tok = new MiniAIBPE.BPETokenizer({ vocab: m.vocab, merges: m.merges || [] });

const texts = JSON.parse(fs.readFileSync(0, "utf8"));
const out = texts.map((t) => {
  const ids = tok.encode(t);
  return { text: t, ids: ids, decoded: tok.decode(ids) };
});
process.stdout.write(JSON.stringify(out));
