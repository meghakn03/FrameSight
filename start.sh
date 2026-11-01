#!/usr/bin/env bash
# Bootstrap FrameSight and replace this process with its Streamlit server.
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

supports_python() {
  "$1" -c 'import sys; raise SystemExit(not ((3, 10) <= sys.version_info[:2] <= (3, 12)))' >/dev/null 2>&1
}

if [[ -x .venv/bin/python ]]; then
  if ! supports_python .venv/bin/python; then
    echo 'The existing .venv needs Python 3.10–3.12. Rename or remove it, then run ./start.sh again.' >&2
    exit 1
  fi
else
  PYTHON_BIN=''
  for candidate in python3.11 python3.12 python3.10 "$HOME/.local/bin/python3.11" python3; do
    if command -v "$candidate" >/dev/null 2>&1 && supports_python "$candidate"; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
  if [[ -z "$PYTHON_BIN" ]]; then
    echo 'Install Python 3.10–3.12 (3.11 recommended), then run ./start.sh again.' >&2
    exit 1
  fi
  echo 'Creating the FrameSight virtual environment…'
  "$PYTHON_BIN" -m venv .venv
fi

mkdir -p .runtime/cache .runtime/yolo .runtime/matplotlib
export PIP_CACHE_DIR="$PROJECT_DIR/.runtime/cache"
export YOLO_CONFIG_DIR="$PROJECT_DIR/.runtime/yolo"
export MPLCONFIGDIR="$PROJECT_DIR/.runtime/matplotlib"

REQUIREMENTS_HASH="$(.venv/bin/python -c 'import hashlib; print(hashlib.sha256(open("requirements.txt", "rb").read()).hexdigest())')"
INSTALLED_HASH="$(cat .venv/.framesight-requirements 2>/dev/null || true)"
if [[ "$REQUIREMENTS_HASH" != "$INSTALLED_HASH" ]]; then
  echo 'Installing FrameSight dependencies (first run or updated requirements)…'
  .venv/bin/python -m pip install -r requirements.txt
  printf '%s\n' "$REQUIREMENTS_HASH" > .venv/.framesight-requirements
fi

echo 'Starting FrameSight. Press Ctrl+C to stop.'
exec .venv/bin/python -m streamlit run app.py --browser.gatherUsageStats false "$@"
