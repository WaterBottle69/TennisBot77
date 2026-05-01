#!/usr/bin/env bash
# start.sh — Launch TennisBot77 (bot engine + web dashboard + optional terminal TUI)
# Usage:
#   ./start.sh          → bot + web server only   (open http://localhost:8000)
#   ./start.sh tui      → bot + web server + terminal TUI in this window

set -e
cd "$(dirname "$0")"

# ── Find Python 3.10+ ─────────────────────────────────────────────────────────
# 3.10 minimum: codebase uses X | Y union type syntax (PEP 604).
PYTHON=""
for candidate in python3 python3.13 python3.12 python3.11 python3.10 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(sys.version_info >= (3,10))" 2>/dev/null)
        if [ "$ver" = "True" ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[start.sh] ERROR: Python 3.10+ not found. Install it first:"
    echo "  macOS:         brew install python@3.11"
    echo "  Debian/Ubuntu: sudo apt install python3.11 python3.11-venv"
    exit 1
fi

# ── Create venv if missing ────────────────────────────────────────────────────
VENV_PY="./venv/bin/python"
if [ ! -f "$VENV_PY" ]; then
    echo "[start.sh] Creating virtual environment..."
    "$PYTHON" -m venv venv
fi

# ── Install / update dependencies ────────────────────────────────────────────
echo "[start.sh] Checking dependencies..."
"$VENV_PY" -m pip install -q --upgrade pip
"$VENV_PY" -m pip install -q -r requirements.txt

# ── Playwright: Chromium browser for tennisstats.com fallback scraping ────────
# Downloads the binary once (~130 MB) into ~/.cache/ms-playwright.
# On Linux also installs the OS-level shared libraries Chromium needs.
if "$VENV_PY" -c "import playwright" 2>/dev/null; then
    PW_CACHE="$HOME/.cache/ms-playwright"
    if [ ! -d "$PW_CACHE" ] || [ -z "$(ls -A "$PW_CACHE" 2>/dev/null)" ]; then
        echo "[start.sh] Installing Playwright Chromium browser (~130 MB, one-time)..."
        "$VENV_PY" -m playwright install chromium
    fi
    if [ "$(uname)" = "Linux" ]; then
        echo "[start.sh] Installing Playwright Linux system dependencies..."
        "$VENV_PY" -m playwright install-deps chromium 2>/dev/null || \
            echo "[start.sh] WARNING: playwright install-deps failed — run manually if Chromium crashes."
    fi
fi

echo "[start.sh] Starting TennisBot77 engine + web server..."
echo "[start.sh] Web dashboard → http://localhost:8000"
echo "[start.sh] Press Ctrl+C to stop."

if [ "$1" = "tui" ]; then
    # Start the bot+server in background, launch TUI in foreground
    "$VENV_PY" main.py &
    BOT_PID=$!
    echo "[start.sh] Bot PID: $BOT_PID — waiting 4s for startup..."
    sleep 4
    trap "kill $BOT_PID 2>/dev/null; exit" EXIT INT TERM
    "$VENV_PY" tui_dashboard.py
else
    # Run everything in foreground (main.py starts uvicorn internally)
    exec "$VENV_PY" main.py
fi
