#!/usr/bin/env bash
# start.sh — Launch TennisBot77 (bot engine + web dashboard + optional terminal TUI)
# Usage:
#   ./start.sh          → bot + web server only   (open http://localhost:8000)
#   ./start.sh tui      → bot + web server + terminal TUI in this window

set -e
cd "$(dirname "$0")"

VENV_PY="./venv/bin/python"
if [ ! -f "$VENV_PY" ]; then
    echo "[start.sh] venv not found — creating it..."
    python3 -m venv venv
fi

# Ensure all deps are installed (fast no-op if already satisfied)
"$VENV_PY" -m pip install -q -r requirements.txt

echo "[start.sh] Starting TennisBot77 engine + web server..."
echo "[start.sh] Web dashboard → http://localhost:8000"
echo "[start.sh] Press Ctrl+C to stop."

if [ "$1" = "tui" ]; then
    # Start the bot+server in background, launch TUI in foreground
    "$VENV_PY" main.py &
    BOT_PID=$!
    echo "[start.sh] Bot PID: $BOT_PID — waiting 3s for startup..."
    sleep 3
    trap "kill $BOT_PID 2>/dev/null" EXIT
    "$VENV_PY" tui_dashboard.py
else
    # Run everything in foreground (bot starts web server internally)
    exec "$VENV_PY" main.py
fi
