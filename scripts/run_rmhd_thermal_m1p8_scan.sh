#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# RANKS can be overridden, for example: RANKS=4 ./scripts/run_rmhd_thermal_m1p8_scan.sh
exec python3 "$SCRIPT_DIR/run_rmhd_experiments.py" \
    --suite thermal_m1p8 --ranks "${RANKS:-16}" "$@"
