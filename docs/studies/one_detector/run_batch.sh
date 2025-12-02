#!/usr/bin/env bash
# set -euo pipefail

# Configure runs here.
# GPS_LIST=(
#   1368350730
#   1186741733
#   1186741737
# )
# To run 100 sequential GPS offsets, replace GPS_LIST above with e.g.:
GPS_LIST=($(seq 1187721218 4 $((1187721218 + 1*5000))))

O2END=1187733618

INJECT_LIST=(glitch) # noise signal glitch)
DETECTOR="H1"

# Optional: set seeds per run. If empty, uses loop index.
SEEDS=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_PY="${SCRIPT_DIR}/one_detector_analysis.py"

run_idx=0
for gps in "${GPS_LIST[@]}"; do
  for inject in "${INJECT_LIST[@]}"; do
    seed="${SEEDS[$run_idx]:-$run_idx}"
    echo ""
    echo ">>> Running detector=${DETECTOR} gps=${gps} inject=${inject} seed=${seed}"
    python "${ANALYSIS_PY}" --detector "${DETECTOR}" --gps "${gps}" --inject "${inject}" --seed "${seed}" --snr-min 20
    echo ">>> DONE <<<"
    echo ""
    ((run_idx++))
  done
done


