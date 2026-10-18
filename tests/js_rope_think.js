// Сверка rope: node tests/js_rope_think.js model.json < ids_lists.json
// → JSON [{out, firstLogits}] — greedy-генерация JS-трансформера по готовым id-токенам.
const fs = require("fs");
const path = require("path");
const { TransformerModel } = require(path.join(__dirname, "..", "app", "www", "transformer.js"));

const modelPath = process.argv[2];
const m = JSON.parse(fs.readFileSync(modelPath, "utf8"));
const model = new TransformerModel(m);

const idsLists = JSON.parse(fs.readFileSync(0, "utf8"));
const out = idsLists.map((ids) => {
  const gen = model.generate(ids, { temperature: 0, maxNew: 30 });
  const firstLogits = gen.steps.length ? Array.from(gen.steps[0].logits) : null;
  return { out: gen.ids, firstLogits: firstLogits };
});
process.stdout.write(JSON.stringify(out));
