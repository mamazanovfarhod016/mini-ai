// Сверка: node tests/js_transformer_think.js < questions.json
// → JSON [{question, answer, firstLogits}] — ответы JS-трансформера (greedy).
const fs = require("fs");
const path = require("path");
const { TransformerModel } = require(path.join(__dirname, "..", "app", "www", "transformer.js"));

const modelPath = path.join(__dirname, "..", "model_transformer.json");
const m = JSON.parse(fs.readFileSync(modelPath, "utf8"));
const model = new TransformerModel(m);

const questions = JSON.parse(fs.readFileSync(0, "utf8"));
const out = questions.map((q) => {
  const ids = model.promptIds(q);
  const gen = model.generate(ids, { temperature: 0 });
  const firstLogits = gen.steps.length ? Array.from(gen.steps[0].logits) : null;
  return { question: q, answer: gen.text, firstLogits: firstLogits };
});
process.stdout.write(JSON.stringify(out));
