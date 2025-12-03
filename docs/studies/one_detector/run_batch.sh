#!/usr/bin/env bash
# set -euo pipefail

# Configure runs here.
# GPS_LIST=(
#   1368350730
#   1186741733
#   1186741737
# )
# Cache coverage (single file bundled with this script)
CACHE_FILE="cache/H-H1_GWOSC_O2_4KHZ_R1-1187721216-4096.hdf5"
CACHE_START=1187721216
CACHE_DUR=4096          # seconds in cache file
DURATION=32             # must match one_detector_analysis.py
CACHE_END=$((CACHE_START + CACHE_DUR))
ALLOWED_MAX=$((CACHE_END - DURATION))

# GPS start times to run (within cache span). Adjust step/count as needed.
GPS_LIST=($(seq -f "%.0f" ${CACHE_START} 4 ${ALLOWED_MAX}))

INJECT_LIST=(glitch signal glitch)
DETECTOR="H1"

# Optional: set seeds per run. If empty, uses loop index.
SEEDS=()

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_PY="${SCRIPT_DIR}/one_detector_analysis.py"
echo "Using cache ${CACHE_FILE}"
echo "Cache covers [${CACHE_START}, ${CACHE_END}] => allowed GPS start in [${CACHE_START}, ${ALLOWED_MAX}]"

run_idx=0
for gps in "${GPS_LIST[@]}"; do
  for inject in "${INJECT_LIST[@]}"; do
    seed="${SEEDS[$run_idx]:-$run_idx}"
    echo ""
    echo ">>> Running detector=${DETECTOR} gps=${gps} inject=${inject} seed=${seed}"
    python "${ANALYSIS_PY}" --detector "${DETECTOR}" --gps "${gps}" --inject "${inject}" --seed "${seed}" --cache-file "${CACHE_FILE}"
    echo ">>> DONE <<<"
    echo ""
    ((run_idx++))
  done
done
