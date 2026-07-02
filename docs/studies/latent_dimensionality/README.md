# Latent Dimensionality Sweep

This study trains Starccato VAEs at multiple latent dimensions and records the
numbers used to decide whether low-dimensional latents are sufficient.

Full run:

```bash
uv run python docs/studies/latent_dimensionality/run_latent_sweep.py \
  --latent-dims 2,3,4,5,6,8,12,16 \
  --epochs 1000
```

Fast smoke test:

```bash
uv run python docs/studies/latent_dimensionality/run_latent_sweep.py \
  --latent-dims 2 \
  --epochs 2 \
  --outdir /tmp/starccato_latent_sweep_smoke
```

The script writes:

- `out/latent_sweep_metrics.csv`: fit, KL, active-dimension, correlation, and
  total-correlation metrics for each trained model.
- `out/latent_sweep_separability.csv`: LDA ROC-AUC and Gaussian JSD for
  cross-model separability at each latent dimension.
- `docs/assets/latent_dimensionality_sweep.png`: the README figure, plotted
  from the CSV files.

No values are hard-coded in the plotting path. Re-running with `--force`
re-trains existing model directories.
