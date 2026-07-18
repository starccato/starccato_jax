#!/usr/bin/env bash
set -euo pipefail

# Selected by the July 2026 retraining study. The blip run intentionally uses
# 1000 epochs: the matched 2000-epoch run stopped at epoch 1408 and had worse
# held-out deterministic reconstruction MSE.
uv run train-vae \
  --dataset ccsne \
  --latent-dim 5 \
  --epochs 2000 \
  --cycles 3 \
  --batch-size 64 \
  --seed 1 \
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
  --cycles 3 \
  --batch-size 64 \
  --seed 0 \
  --data-seed 0 \
  --use-capacity \
  --capacity-start 0 \
  --capacity-end 12 \
  --capacity-warmup-epochs 500 \
  --beta-capacity 5 \
  --normalize-decoder-output \
  --outdir out_blip
