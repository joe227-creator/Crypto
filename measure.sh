#!/usr/bin/env bash
set -euo pipefail

VENV="${VENV:-.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi

"$VENV/bin/python" - <<'PY'
import importlib.util
import subprocess
import sys

required = ("numpy", "pandas", "requests", "optuna")
if any(importlib.util.find_spec(name) is None for name in required):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
PY

exec "$VENV/bin/python" measure.py "$@"
