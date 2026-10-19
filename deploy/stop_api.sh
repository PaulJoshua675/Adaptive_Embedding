#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${RAG_API_PORT:-8000}"
PID_FILE="$ROOT/data/api-$PORT.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "No managed API PID file found for port $PORT."
  exit 0
fi

PID="$(cat "$PID_FILE")"
if kill -0 "$PID" 2>/dev/null; then
  COMMAND="$(ps -p "$PID" -o args= || true)"
  if [[ "$COMMAND" == *"uvicorn main_api:app"* ]]; then
    kill "$PID"
    echo "Stopped API process $PID."
  else
    echo "PID $PID is not the managed Uvicorn process; leaving it untouched." >&2
    exit 1
  fi
fi
rm -f "$PID_FILE"