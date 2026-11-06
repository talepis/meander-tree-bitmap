#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
task_python="${PYTHON_BIN:-python3}"
"$task_python" -c 'import sys;assert (3,11)<=sys.version_info[:2]<=(3,14) and sys.maxsize>2**32,"Use 64-bit Python 3.11–3.14"'
"$task_python" -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check --only-binary=:all: --index-url https://pypi.org/simple -r requirements.txt
.venv/bin/python verify_bundle.py
printf '\nReady. Start with: bash START_M38.sh\n'
