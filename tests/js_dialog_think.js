// Паритет Python<->JS для диалога: node tests/js_dialog_think.js <model.json>
// stdin: JSON {"cases": [{"history":[{"role","text"}...], "last_q":"...", "max_history_turns":8}, ...],
//              "antirepeat_cases": [{"logits":[...], "gen_ids":[...], "rp":1.3, "ngram":3}, ...]}
// stdout: JSON {"prompt_ids": [[...], ...], "antirepeat_logits": [[...], ...]}
const fs = require("fs");
const path = require("path");
const MiniTransformer = require(path.join(__dirname, "..", "app", "www", "transformer.js"));

const modelPath = process.argv[2];
const m = JSON.parse(fs.readFileSync(modelPath, "utf8"));
const model = new MiniTransformer.TransformerModel(m);

const input = JSON.parse(fs.readFileSync(0, "utf8"));
const promptIds = (input.cases || []).map((c) =>
  model.promptIdsDialog(c.history, c.last_q, c.max_history_turns || 8)
);
const antirepeat = (input.antirepeat_cases || []).map((c) => {
  const logits = Float64Array.from(c.logits);
  MiniTransformer.applyAntiRepeat(logits, c.gen_ids, c.rp, c.ngram);
  return Array.from(logits);
});
process.stdout.write(JSON.stringify({ prompt_ids: promptIds, antirepeat_logits: antirepeat }));
