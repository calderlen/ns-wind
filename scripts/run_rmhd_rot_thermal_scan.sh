#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# RANKS overrides the default of 16 MPI ranks per sequential case.
exec python3 "$SCRIPT_DIR/run_rmhd_experiments.py" \
    --suite rot_thermal_m2 --ranks "${RANKS:-16}" "$@"
