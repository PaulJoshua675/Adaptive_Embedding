#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${RAG_API_HOST:-0.0.0.0}"
PORT="${RAG_API_PORT:-8000}"
PYTHON="${RAG_PYTHON:-$ROOT/.venv/bin/python}"
PID_FILE="$ROOT/data/api-$PORT.pid"
LOG_FILE="$ROOT/data/api-$PORT.log"
STARTUP_TIMEOUT="${RAG_API_STARTUP_TIMEOUT:-90}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python executable not found: $PYTHON" >&2
  exit 1
fi
mkdir -p "$ROOT/data"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "API already running with PID $(cat "$PID_FILE")." >&2
  exit 1
fi

cd "$ROOT"
nohup "$PYTHON" -m uvicorn main_api:app --host "$HOST" --port "$PORT" >>"$LOG_FILE" 2>&1 &
PID=$!
echo "$PID" > "$PID_FILE"

for ((attempt = 0; attempt < STARTUP_TIMEOUT * 2; attempt++)); do
  if ! kill -0 "$PID" 2>/dev/null; then
    tail -n 60 "$LOG_FILE" >&2 || true
    rm -f "$PID_FILE"
    echo "Uvicorn exited during startup." >&2
    exit 1
  fi
  if curl --fail --silent "http://127.0.0.1:$PORT/health" >/dev/null; then
    echo "API ready at http://127.0.0.1:$PORT (PID $PID); listening on $HOST."
    exit 0
  fi
  sleep 0.5
done

kill "$PID" 2>/dev/null || true
rm -f "$PID_FILE"
tail -n 60 "$LOG_FILE" >&2 || true
echo "API did not become healthy within ${STARTUP_TIMEOUT}s." >&2
exit 1