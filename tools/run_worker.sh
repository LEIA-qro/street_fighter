#!/usr/bin/env bash
# run_worker.sh -- launch an ES evaluation worker on this machine.
#
#   tools/run_worker.sh --coordinator http://<coordinator-ip>:8823 --procs 8
#
# Activates the repo-local .venv when present (WSL2/mac dev boxes); falls
# back to whatever `python3` is on PATH otherwise. All arguments pass
# through to src/es/worker.py.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

exec python3 src/es/worker.py "$@"
