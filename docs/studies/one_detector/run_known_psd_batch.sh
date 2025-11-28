#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ANALYSIS_PY="${SCRIPT_DIR}/one_detector_analysis_known_psd.py"

DETECTOR="H1"
INJECT="signal"   # change to glitch/noise if desired
SNR_MIN=3
SNR_MAX=100
N_RUNS=1000

for ((i=0; i<${N_RUNS}; i++)); do
  seed=${i}
  echo ">>> Run $i seed=$seed"
  python "${ANALYSIS_PY}" \
    --detector "${DETECTOR}" \
    --inject "${INJECT}" \
    --seed "${seed}" \
    --snr-min "${SNR_MIN}" \
    --snr-max "${SNR_MAX}"
done
