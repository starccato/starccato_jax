#!/usr/bin/env bash
set -euo pipefail

# Selected by the July 2026 retraining and inference-geometry study. Both runs
# intentionally use seed 2: lower-MSE alternatives produced multimodal or
# slowly mixing LVK posteriors and failed four-chain convergence.
uv run train-vae \
  --dataset ccsne \
  --latent-dim 5 \
  --epochs 1000 \
  --cycles 0 \
  --batch-size 64 \
  --seed 2 \
  --data-seed 0 \
  --use-capacity \
  --capacity-start 0 \
  --capacity-end 12 \
  --capacity-warmup-epochs 500 \
  --beta-capacity 5 \
  --normalize-decoder-output \
  --outdir out_ccsne

uv run train-vae \
  --dataset blip \
  --latent-dim 5 \
  --epochs 1000 \
  --cycles 0 \
  --batch-size 64 \
  --seed 2 \
  --data-seed 0 \
  --use-capacity \
  --capacity-start 0 \
  --capacity-end 12 \
  --capacity-warmup-epochs 500 \
  --beta-capacity 5 \
  --normalize-decoder-output \
  --outdir out_blip
