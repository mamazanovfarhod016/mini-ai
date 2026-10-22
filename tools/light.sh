#!/bin/bash
# light.sh — лёгкий прогон (один процесс, ≤1.6 ГБ RSS, nice 19, без лока heavy) для помощников-агентов.
# Использование: tools/light.sh script.py [args]  — вывод в stdout. Отличие от heavy.sh: меньший лимит памяти и нет глобального
# лока, поэтому до ДВУХ light-процессов могут идти параллельно с одним heavy. Третий — отказ. Без MiniLM/DistilBERT (MINI_AI_NEURAL=0),
# иначе память не влезет.
set -u
cd "$(dirname "$0")/.." || exit 1
n=$(pgrep -fc "LIGHT_RUN=1" || true); n=${n:-0}
if [ "$n" -ge 3 ]; then echo "BUSY: $n light jobs running"; exit 2; fi
avail=$(free -m | awk 'NR==2{print $7}')
if [ "$avail" -lt 2500 ]; then echo "LOW MEMORY: ${avail} MB"; exit 3; fi
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 LIGHT_RUN=1 MINI_AI_NEURAL=${MINI_AI_NEURAL:-0}
echo 900 > /proc/self/oom_score_adj 2>/dev/null
# сторож в фоне наблюдает за НАШИМ pid; сам процесс — в foreground через exec после запуска сторожа
( parent=$$   # $$ и в подоболочке = pid этого скрипта, который после exec станет python; $PPID смотрел бы на внешний shell (и вешал пайп)
  while kill -0 "$parent" 2>/dev/null; do
    rss=$(awk '/VmRSS/{print int($2/1024)}' /proc/$parent/status 2>/dev/null || echo 0); av=$(free -m | awk 'NR==2{print $7}')
    if [ "${rss:-0}" -gt 1600 ] || [ "$av" -lt 1000 ]; then echo "WATCHDOG KILL: rss=${rss}MB avail=${av}MB" >&2; kill -9 "$parent"; break; fi
    sleep 2
  done ) &
exec nice -n 19 ionice -c 3 "$HOME/Developer/mini-ai/.venv/bin/python" "$@"
