#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [ -n "${MEANDER_PYTHON:-}" ]; then
  task_python="$MEANDER_PYTHON"
else
  if [ ! -x .venv/bin/python ]; then bash setup_linux.sh; fi
  task_python=.venv/bin/python
fi
task_threads="${MEANDER_THREADS:-auto}"
task_groups="${MEANDER_GROUPS:-4}"
"$task_python" self_test.py --threads "$task_threads"
exec "$task_python" server_meander.py 38 --groups "$task_groups" --threads "$task_threads" --resume "$@"
