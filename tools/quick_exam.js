// Быстрый экзамен в ДВИЖКЕ ПРИЛОЖЕНИЯ: node tools/quick_exam.js → сейчас в приложении vs новая школьная модель
const fs = require("fs"), path = require("path");
const ROOT = path.join(__dirname, "..");
global.MiniAIBPE = require(path.join(ROOT, "app/www/bpe.js"));
const { TransformerModel } = require(path.join(ROOT, "app/www/transformer.js"));
const load = (f) => new TransformerModel(JSON.parse(fs.readFileSync(path.join(ROOT, f), "utf8")));
const QS = ["привет", "как дела", "кто ты", "что ты умеешь", "сколько будет 7 + 5", "12 * 3", "100 - 37",
  "salom", "qalaysan", "2+2", "hello", "how are you", "45 : 5", "спасибо", "пока",
  "расскажи шутку", "помоги", "18 + 27", "у Ali было 5 яблок, он отдал 2. сколько осталось", "9 * 8"];
const ans = (m, q) => m.generate(m.promptIds(q), { temperature: 0 }).text;
const old = load("model_transformer.json"), neu = load("model_transformer_school.json");
const rows = QS.map((q) => ({ q, old: ans(old, q), new: ans(neu, q) }));
process.stdout.write(JSON.stringify(rows, null, 1));
