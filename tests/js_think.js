// Сверка: node tests/js_think.js < phrases.json → JSON с ответами JS-мозга
const fs = require("fs");
const path = require("path");
const { Brain } = require(path.join(__dirname, "..", "app", "www", "brain.js"));
const model = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "model.json"), "utf8"));
const brain = new Brain(model);
const phrases = JSON.parse(fs.readFileSync(0, "utf8"));
const out = phrases.map((p) => {
  const t = brain.think(p);
  return { phrase: p, intent: t.intent, confidence: t.confidence, top3: t.top3, familiar: t.familiar };
});
process.stdout.write(JSON.stringify(out));
