#!/usr/bin/env bash
# Учёба по книгам по очереди (19.10.2026): ждёт окончания обучения логике на Kaggle, потом четверть за четвертью.
# Стоп: всё пройдено, или 3 провала подряд на одной четверти (тогда разбираемся вручную), или 40 шагов.
cd "$(dirname "$0")/.." || exit 1
while pgrep -f "kaggle_train --logic" >/dev/null; do sleep 120; done
LOGIC=server_data/models/_candidate_logic1.json
if [ -f "$LOGIC" ] && [ ! -f server_data/book_school.json ]; then
  printf '{"model": "%s", "grade": 1, "quarter": 1, "attempts": 0, "history": []}\n' "$(pwd)/$LOGIC" > server_data/book_school.json
fi
for i in $(seq 40); do
  .venv/bin/python -u -m server.book_school || exit 1
  .venv/bin/python - <<'EOF' || exit 0
import json, sys
s = json.load(open("server_data/book_school.json"))
sys.exit(1 if s.get("done") or s.get("attempts", 0) >= 3 else 0)
EOF
done
