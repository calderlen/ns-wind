#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# RANKS can be overridden for a smaller test, for example: RANKS=4 ./run_rhd_suite.sh
exec python3 "$SCRIPT_DIR/run_rhd_experiments.py" --ranks "${RANKS:-8}" "$@"
