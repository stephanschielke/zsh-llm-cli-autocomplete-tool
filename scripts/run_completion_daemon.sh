#!/bin/bash
# Start the completion daemon for low-latency Tab (Cursor-style).
# Run in background or in a dedicated terminal.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"
PYTHON="${PYTHON_CMD:-$ROOT/venv/bin/python3}"
[[ -x "$PYTHON" ]] || PYTHON="$(command -v python3)"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PYTHON" -m model_completer.daemon
