#!/usr/bin/env bash
# Mini AI: сервер самообучения + публичный адрес (Cloudflare quick tunnel) — запускать в tmux.
#   tmux new -d -s miniai 'bash ~/Developer/mini-ai/server/run.sh'
# Quick-tunnel даёт новый адрес при каждом запуске → скрипт публикует его в ветку `live`
# (файл server.json), приложение берёт адрес оттуда (app/www/learn.js, DISCOVERY).
set -uo pipefail
cd "$(dirname "$0")/.."
ROOT=$(pwd); DATA="$ROOT/server_data"; PORT=8787
CF="${CLOUDFLARED:-/opt/hermes-data/bin/cloudflared}"
mkdir -p "$DATA/models" "$DATA/logs"

# веса для скачивания: если ещё ничего не опубликовано — базовая BPE-модель как версия 0 не нужна
python3 -m server.app --port $PORT --data "$DATA" >> "$DATA/logs/server.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; kill $TUN 2>/dev/null' EXIT

publish() {  # $1 = https-адрес
  local tmp; tmp=$(mktemp -d)
  git -C "$ROOT" worktree add -q --detach "$tmp" 2>/dev/null || return 1
  ( cd "$tmp" && git checkout -q --orphan live-tmp && git rm -rq --cached . >/dev/null 2>&1; find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +
    printf '{"url": "%s", "updated": "%s"}\n' "$1" "$(date -u +%FT%TZ)" > server.json
    printf 'Ветка только для адреса сервера самообучения Mini AI (обновляется server/run.sh).\n' > README.md
    git add server.json README.md && git commit -qm "live: адрес сервера" && git push -qf origin HEAD:live )
  local rc=$?
  git -C "$ROOT" worktree remove --force "$tmp"; git -C "$ROOT" branch -D live-tmp >/dev/null 2>&1
  return $rc
}

while true; do
  : > "$DATA/logs/tunnel.log"
  "$CF" tunnel --no-autoupdate --url "http://127.0.0.1:$PORT" >> "$DATA/logs/tunnel.log" 2>&1 &
  TUN=$!
  URL=""
  for _ in $(seq 60); do
    URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' "$DATA/logs/tunnel.log" | head -1)
    [ -n "$URL" ] && break; sleep 1
  done
  if [ -n "$URL" ]; then
    echo "$(date '+%F %T') туннель: $URL" >> "$DATA/logs/server.log"
    for _ in 1 2 3; do publish "$URL" && break; sleep 10; done
  fi
  wait $TUN
  echo "$(date '+%F %T') туннель упал — перезапуск" >> "$DATA/logs/server.log"
  kill -0 $SRV 2>/dev/null || { python3 -m server.app --port $PORT --data "$DATA" >> "$DATA/logs/server.log" 2>&1 & SRV=$!; }
  sleep 5
done
