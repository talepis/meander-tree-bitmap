#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
python_bin="${PYTHON_BIN:-python3.13}"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3,13) and sys.maxsize > 2**32, "Use 64-bit Python 3.13"'
"$python_bin" -m venv .venv
.venv/bin/python -m pip install --disable-pip-version-check -r source_frozen/requirements.txt
.venv/bin/python source_frozen/verify_bundle.py
printf '
Setup complete. See README.md for a small-order self-test.
'
