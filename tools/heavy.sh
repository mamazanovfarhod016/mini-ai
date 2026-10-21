#!/bin/bash
# heavy.sh — ЕДИНСТВЕННЫЙ способ запускать тяжёлое Mini AI на сервере (2 ядра / 8 ГБ, рядом живёт gateway).
# Гарантии: (1) один тяжёлый процесс за раз (лок); (2) стартует только при ≥4 ГБ свободных; (3) СТОРОЖ по реальной памяти:
# RSS процесса > 3.2 ГБ или свободной < 1.2 ГБ → процесс убивается (SIGKILL), сервер жив; (4) nice 19 + ionice, 1 поток BLAS;
# (5) OOM-killer выберет его первым (oom_score_adj 1000). ulimit -v не используем: torch/OpenBLAS резервируют адреса, не память.
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
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
( echo 1000 > /proc/self/oom_score_adj 2>/dev/null; exec nice -n 19 ionice -c 3 "$PWD/.venv/bin/python" "$@" ) >> "$LOG" 2>&1 &
pid=$!
# сторож
while kill -0 "$pid" 2>/dev/null; do
  rss=$(awk '/VmRSS/{print int($2/1024)}' /proc/$pid/status 2>/dev/null || echo 0)
  av=$(free -m | awk 'NR==2{print $7}')
  if [ "${rss:-0}" -gt 3200 ] || [ "$av" -lt 1200 ]; then
    echo "$(date -u +%FT%TZ) WATCHDOG KILL: rss=${rss}MB avail=${av}MB" >> "$LOG"; kill -9 "$pid" 2>/dev/null; break
  fi
  sleep 2
done
wait "$pid" 2>/dev/null; rc=$?
echo "$(date -u +%FT%TZ) end rc=$rc" >> "$LOG"; rm -f "$LOCK"
grep -v "Warning\|warn(\|Loading weights\|it/s" "$LOG" | tail -8
exit $rc
