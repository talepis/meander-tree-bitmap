#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
python_bin="${PYTHON_BIN:-python3.12}"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3,12) and sys.maxsize > 2**32, "Use 64-bit Python 3.12"'
"$python_bin" -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check -r requirements.txt
.venv/bin/python verify_bundle.py
printf '\nSetup complete. Start with: bash START_M37.sh\n'
