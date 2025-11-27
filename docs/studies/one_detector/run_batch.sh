#!/usr/bin/env bash
set -euo pipefail

# Configure runs here.
GPS_LIST=(
  1186741733
  1186741737
)
# To run 100 sequential GPS offsets, replace GPS_LIST above with e.g.:
# GPS_LIST=($(seq 1186741733 4 $((1186741733 + 4*99))))
INJECT_LIST=(signal glitch noise)
DETECTOR="H1"

# Optional: set seeds per run. If empty, uses loop index.
SEEDS=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_PY="${SCRIPT_DIR}/one_detector_analysis.py"

run_idx=0
for gps in "${GPS_LIST[@]}"; do
  for inject in "${INJECT_LIST[@]}"; do
    seed="${SEEDS[$run_idx]:-$run_idx}"
    echo ">>> Running detector=${DETECTOR} gps=${gps} inject=${inject} seed=${seed}"
    python "${ANALYSIS_PY}" --detector "${DETECTOR}" --gps "${gps}" --inject "${inject}" --seed "${seed}"
    ((run_idx++))
  done
done


