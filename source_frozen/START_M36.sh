#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
py="${MEANDER_PYTHON:-$PWD/.venv/bin/python}"
threads="${MEANDER_THREADS:-16}"
if [[ ! -x "$py" ]]; then
  echo 'First run: bash setup_linux.sh, or set MEANDER_PYTHON to an existing Python 3.12 environment.' >&2
  exit 2
fi
"$py" verify_bundle.py
"$py" server_meander.py 36 --inspect --threads "$threads"
"$py" self_test.py --threads "$threads" --output SELF_TEST_RESULT.json
exec "$py" server_meander.py 36 --threads "$threads" --reserve-gib 16 --timeout-hours 72 --output runs/M36
