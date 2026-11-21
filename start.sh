#!/usr/bin/env bash
# =============================================================================
# Asteroids AI - Clean, Rebuild, and Restart Script
# =============================================================================
# Usage:
#   ./start.sh              # Clean, verify tests, and start Streamlit (port 8501)
#   ./start.sh --web        # Clean, verify tests, and start standalone web server (port 8000)
#   ./start.sh --rebuild    # Force full virtualenv recreation from requirements.txt
#   ./start.sh --skip-tests # Skip unit & syntax test suite
#   ./start.sh --stop       # Stop any running project servers and exit
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

MODE="streamlit"
FORCE_REBUILD=false
SKIP_TESTS=false
STOP_ONLY=false
STREAMLIT_PORT="${STREAMLIT_PORT:-8501}"
WEB_PORT="${WEB_PORT:-8000}"

for arg in "$@"; do
    case "$arg" in
        --web)
            MODE="web"
            ;;
        --streamlit)
            MODE="streamlit"
            ;;
        --rebuild)
            FORCE_REBUILD=true
            ;;
        --skip-tests)
            SKIP_TESTS=true
            ;;
        --stop)
            STOP_ONLY=true
            ;;
        -h|--help)
            sed -n '2,11p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Run './start.sh --help' for usage."
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "🚀 Asteroids AI: Clean, Rebuild & Restart"
echo "============================================================"

# -----------------------------------------------------------------------------
# 1. Stop any running instances
# -----------------------------------------------------------------------------
echo "[1/4] Stopping running server instances..."
stopped_any=false

if pgrep -f "streamlit run app.py" >/dev/null 2>&1; then
    pkill -f "streamlit run app.py" || true
    stopped_any=true
fi

if pgrep -f "http.server $WEB_PORT" >/dev/null 2>&1; then
    pkill -f "http.server $WEB_PORT" || true
    stopped_any=true
fi

# Free ports if still occupied
for port in "$STREAMLIT_PORT" "$WEB_PORT"; do
    pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        kill -9 $pids 2>/dev/null || true
        stopped_any=true
    fi
done

if [ "$stopped_any" = true ]; then
    sleep 1
    echo "  ✓ Stopped existing processes."
else
    echo "  ✓ No running instances found."
fi

if [ "$STOP_ONLY" = true ]; then
    echo "Done (--stop)."
    exit 0
fi

# -----------------------------------------------------------------------------
# 2. Clean temporary caches and logs
# -----------------------------------------------------------------------------
echo "[2/4] Cleaning caches and temporary files..."
find "$PROJECT_DIR" -maxdepth 3 -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$PROJECT_DIR" -maxdepth 3 -name "*.pyc" -delete 2>/dev/null || true
rm -f "$PROJECT_DIR/streamlit.log"
echo "  ✓ Cleaned __pycache__, *.pyc, and old logs."

# -----------------------------------------------------------------------------
# 3. Repair or Rebuild Python Virtual Environment (.venv)
# -----------------------------------------------------------------------------
echo "[3/4] Checking Python environment (.venv)..."

if [ "$FORCE_REBUILD" = true ] || [ ! -d "$PROJECT_DIR/.venv" ]; then
    echo "  ↻ Building fresh virtual environment in .venv..."
    rm -rf "$PROJECT_DIR/.venv"
    python3 -m venv "$PROJECT_DIR/.venv"
    "$PROJECT_DIR/.venv/bin/pip" install --upgrade pip
    "$PROJECT_DIR/.venv/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
    echo "  ✓ Virtual environment rebuilt."
else
    # Fix stale paths if the project directory was moved or renamed
    OLD_PATH="/Users/admin/project-ai/asteroids-ML/Asteroids-AI"
    if grep -q "$OLD_PATH" "$PROJECT_DIR/.venv/pyvenv.cfg" 2>/dev/null || \
       grep -q "$OLD_PATH" "$PROJECT_DIR/.venv/bin/streamlit" 2>/dev/null; then
        echo "  ↻ Updating relocated virtualenv paths to $PROJECT_DIR/.venv..."
        for f in "$PROJECT_DIR/.venv/pyvenv.cfg" "$PROJECT_DIR/.venv/bin/"*; do
            if [ -f "$f" ] && [ ! -L "$f" ]; then
                sed -i '' "s|$OLD_PATH|$PROJECT_DIR|g" "$f" 2>/dev/null || true
            fi
        done
    fi

    # Verify streamlit imports cleanly
    if ! "$PROJECT_DIR/.venv/bin/python" -c "import streamlit" >/dev/null 2>&1; then
        echo "  ↻ Streamlit missing or broken in .venv; installing requirements..."
        "$PROJECT_DIR/.venv/bin/pip" install -r "$PROJECT_DIR/requirements.txt"
    fi
    echo "  ✓ Virtual environment ready ($("$PROJECT_DIR/.venv/bin/python" --version))."
fi

# -----------------------------------------------------------------------------
# 4. Run Verification Suite & Start Server
# -----------------------------------------------------------------------------
if [ "$SKIP_TESTS" = false ]; then
    echo "[4/4] Running RL & syntax verification suite..."
    "$PROJECT_DIR/run_tests.sh"
    "$PROJECT_DIR/.venv/bin/python" "$PROJECT_DIR/tests/test_app_template.py"
else
    echo "[4/4] Skipping tests (--skip-tests)."
fi

echo "============================================================"
if [ "$MODE" = "web" ]; then
    echo "🌐 Starting Standalone 3D Web Server at http://localhost:$WEB_PORT"
    echo "   Press Ctrl+C to stop"
    echo "============================================================"
    exec "$PROJECT_DIR/.venv/bin/python" -m http.server "$WEB_PORT"
else
    echo "🎮 Starting Streamlit App at http://localhost:$STREAMLIT_PORT"
    echo "   Press Ctrl+C to stop"
    echo "============================================================"
    exec "$PROJECT_DIR/.venv/bin/streamlit" run "$PROJECT_DIR/app.py" \
        --server.port="$STREAMLIT_PORT" \
        --server.address="localhost" \
        --browser.gatherUsageStats=false
fi
