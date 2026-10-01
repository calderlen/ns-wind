#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Phenomenological nonrotating comparison: Gamma=1.1, CS_REL_0=0.136.
# Completion at tstop is not a steadiness determination.
exec python3 "$SCRIPT_DIR/run_rmhd_experiments.py" \
    --suite gamma1p1 --ranks "${RANKS:-8}" "$@"
