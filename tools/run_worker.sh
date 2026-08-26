#!/usr/bin/env bash
# run_worker.sh -- launch an ES evaluation worker on this machine.
#
#   tools/run_worker.sh --coordinator http://madre:8080                  # auto-sized
#   tools/run_worker.sh --coordinator http://madre:8080 --cpu-share 0.5  # donate half
#   tools/run_worker.sh --coordinator http://madre:8080 --procs 12       # explicit
#
# Sizing flags (all handled in src/es/resources.py, see tools/setup_worker.md):
#   --procs auto|N     emulator processes; 'auto' (default) sizes the machine
#   --reserve-cores K  cores left free for the owner of the box (default 2)
#   --cpu-share F      0.0-1.0, donate this fraction of the machine instead
#   --max-procs N      hard cap, whatever the sizing decides
#   --nice N           niceness of each emulator process (default 10 on POSIX)
#
# Activates the repo-local .venv when present (WSL2/mac dev boxes); falls
# back to whatever `python3` is on PATH otherwise. All arguments pass
# through to src/es/worker.py unchanged.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

exec python3 src/es/worker.py "$@"
