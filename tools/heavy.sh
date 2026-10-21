#!/bin/bash
# heavy.sh — ЕДИНСТВЕННЫЙ способ запускать тяжёлое Mini AI на сервере (2 ядра / 8 ГБ, рядом живёт gateway).
# Гарантии: (1) один тяжёлый процесс за раз; (2) стартует только при ≥4 ГБ свободных; (3) ulimit -v 3.5 ГБ —
# при превышении процесс умирает сам (MemoryError), а не сервер; (4) nice 19 + ionice; (5) OOM-killer выберет его первым.
#   tools/heavy.sh tools/school.py n=3000 state=...      → лог в ~/Developer/mini-ai-runs/learning-core/heavy_<имя>.log
set -u
cd "$(dirname "$0")/.." || exit 1
LOCK=/tmp/mini-ai-heavy.lock; LOG_DIR=$HOME/Developer/mini-ai-runs/learning-core; mkdir -p "$LOG_DIR"
name=$(basename "${1:-job}" .py); LOG="$LOG_DIR/heavy_${name}.log"
if [ -e "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then echo "BUSY: heavy job pid $(cat "$LOCK") is running ($LOG)"; exit 2; fi
avail=$(free -m | awk 'NR==2{print $7}')
if [ "$avail" -lt 4000 ]; then echo "LOW MEMORY: only ${avail} MB available, need 4000"; exit 3; fi
echo $$ > "$LOCK"
echo "$(date -u +%FT%TZ) start: $* (avail ${avail} MB)" >> "$LOG"
( ulimit -v 3500000; echo 1000 > /proc/self/oom_score_adj 2>/dev/null; exec nice -n 19 ionice -c 3 "$PWD/.venv/bin/python" "$@" ) >> "$LOG" 2>&1
rc=$?
echo "$(date -u +%FT%TZ) end rc=$rc" >> "$LOG"; rm -f "$LOCK"
grep -v "Warning\|warn(" "$LOG" | tail -6
exit $rc
