#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv-h200}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 10), "Python >=3.10 required; set PYTHON_BIN"'
"$PYTHON_BIN" -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --upgrade pip
"$VENV_DIR/bin/python" -m pip install torch==2.8.0 --index-url "$TORCH_INDEX_URL"
"$VENV_DIR/bin/python" -m pip install -e '.[dev]'
"$VENV_DIR/bin/python" -m pip check
echo "Environment ready: $VENV_DIR"
