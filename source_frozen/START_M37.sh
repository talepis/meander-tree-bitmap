#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
py="${MEANDER_PYTHON:-$PWD/.venv/bin/python}"
threads="${MEANDER_THREADS:-16}"
"$py" verify_bundle.py
"$py" server_meander.py 37 --inspect --threads "$threads"
"$py" self_test.py --threads "$threads" --extended --output SELF_TEST_RESULT.json
"$py" prepare_memory_preflight.py --output M37_MEMORY_PREFLIGHT.json
exec "$py" server_meander.py 37 --threads "$threads" --geometry-engine bitmap   --numeric-mode parallel --memory-policy full --reserve-gib 16 --timeout-hours 720   --memory-preflight M37_MEMORY_PREFLIGHT.json --output runs/M37_MEMORY
