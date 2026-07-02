"""Train VAE latent-dimension sweeps and plot the recorded measurements.

This script is intentionally heavier than the README examples: it trains the
CCSNe and blip VAEs at each requested latent dimension, records metrics to CSV,
then creates the summary figure from those CSVs. It does not contain any
pre-baked results.

Example full run:

    uv run python docs/studies/latent_dimensionality/run_latent_sweep.py \
      --latent-dims 2,3,4,5,6,8,12,16 \
      --epochs 1000

Fast smoke test:

    uv run python docs/studies/latent_dimensionality/run_latent_sweep.py \
      --latent-dims 2 \
      --epochs 2 \
      --outdir /tmp/starccato_latent_sweep_smoke
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from starccato_jax.data import TrainValData
from starccato_jax.vae import Config
from starccato_jax.vae.core import VAE, encode, load_model, train_vae
from starccato_jax.vae.core.io import LOSS_FNAME, MODEL_FNAME, load_loss_h5
from starccato_jax.vae.core.metrics import compute_latent_stats

ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "docs" / "assets"
DEFAULT_OUTDIR = Path(__file__).resolve().parent / "out"
METRICS_CSV = "latent_sweep_metrics.csv"
SEPARABILITY_CSV = "latent_sweep_separability.csv"
SUMMARY_FIGURE = "latent_dimensionality_sweep.png"


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def parse_str_list(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def model_dir(outdir: Path, dataset: str, latent_dim: int) -> Path:
    return outdir / "models" / f"{dataset}_z{latent_dim}"


def train_if_needed(
    outdir: Path,
    dataset: str,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    force: bool,
) -> Path:
    target = model_dir(outdir, dataset, latent_dim)
    if (target / MODEL_FNAME).exists() and not force:
        print(f"reusing {target}")
        return target

    config = Config(
        latent_dim=latent_dim,
        epochs=epochs,
        batch_size=batch_size,
        dataset=dataset,
        use_capacity=True,
        capacity_start=0.0,
        capacity_end=4.0,
        capacity_warmup_epochs=min(500, epochs),
        beta_capacity=5.0,
    )
    print(f"training {dataset} z={latent_dim} -> {target}")
    train_vae(config.data, config=config, save_dir=str(target))
    return target


def last_recorded_index(values) -> int:
    arr = np.asarray(values, dtype=float)
    finite = np.flatnonzero(np.isfinite(arr) & (arr != 0.0))
    if finite.size == 0:
        return len(arr) - 1
    return int(finite[-1])


def loss_value(losses, split: str, field: str, idx: int) -> float:
    metrics = getattr(losses, f"{split}_metrics")
    return float(np.asarray(getattr(metrics, field))[idx])


def measure_model(savedir: Path, dataset: str, latent_dim: int) -> dict:
    model_data = load_model(str(savedir))
    model = VAE(model_data.latent_dim, data_dim=model_data.data_dim)
    data = TrainValData.load(source=dataset)
    rng = jax.random.PRNGKey(0)
    stats = compute_latent_stats(model_data.params, model, data.val, rng)

    losses = load_loss_h5(str(savedir / LOSS_FNAME))
    idx = last_recorded_index(losses.val_metrics.loss)
    kl_per_dim = np.asarray(stats["kl_per_dim"], dtype=float)

    return {
        "dataset": dataset,
        "latent_dim": latent_dim,
        "model_dir": str(savedir),
        "epoch_recorded": idx + 1,
        "train_loss": loss_value(losses, "train", "loss", idx),
        "val_loss": loss_value(losses, "val", "loss", idx),
        "train_reconstruction_loss": loss_value(
            losses, "train", "reconstruction_loss", idx
        ),
        "val_reconstruction_loss": loss_value(
            losses, "val", "reconstruction_loss", idx
        ),
        "train_kl_divergence": loss_value(
            losses, "train", "kl_divergence", idx
        ),
        "val_kl_divergence": loss_value(losses, "val", "kl_divergence", idx),
        "capacity": loss_value(losses, "val", "capacity", idx),
        "active_dims": stats["active"],
        "n80": stats["n80"],
        "n90": stats["n90"],
        "n100": stats["n100"],
        "mean_abs_corr": stats["mean_abs_corr"],
        "max_abs_corr": stats["max_abs_corr"],
        "total_corr": stats["total_corr"],
        "total_kl": float(np.sum(kl_per_dim)),
        "kl_per_dim": json.dumps(kl_per_dim.tolist()),
    }


def fit_gaussian(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = z.mean(axis=0)
    cov = np.cov(z, rowvar=False)
    cov = np.atleast_2d(cov)
    cov = cov + 1e-6 * np.eye(cov.shape[0])
    return mean, cov


def logpdf_gaussian(
    x: np.ndarray, mean: np.ndarray, cov: np.ndarray
) -> np.ndarray:
    diff = x - mean
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        raise ValueError("Gaussian covariance is not positive definite")
    solved = np.linalg.solve(cov, diff.T).T
    quad = np.sum(diff * solved, axis=1)
    dim = mean.shape[0]
    return -0.5 * (dim * np.log(2.0 * np.pi) + logdet + quad)


def gaussian_jsd(a: np.ndarray, b: np.ndarray, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    mean_a, cov_a = fit_gaussian(a)
    mean_b, cov_b = fit_gaussian(b)
    n = min(2000, len(a), len(b))
    sample_a = rng.multivariate_normal(mean_a, cov_a, size=n)
    sample_b = rng.multivariate_normal(mean_b, cov_b, size=n)

    log_pa_a = logpdf_gaussian(sample_a, mean_a, cov_a)
    log_pb_a = logpdf_gaussian(sample_a, mean_b, cov_b)
    log_pa_b = logpdf_gaussian(sample_b, mean_a, cov_a)
    log_pb_b = logpdf_gaussian(sample_b, mean_b, cov_b)

    log_m_a = np.logaddexp(log_pa_a, log_pb_a) - np.log(2.0)
    log_m_b = np.logaddexp(log_pa_b, log_pb_b) - np.log(2.0)
    return float(
        0.5 * np.mean(log_pa_a - log_m_a) + 0.5 * np.mean(log_pb_b - log_m_b)
    )


def auc_from_scores(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(bool)
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_ranks = ranks[labels]
    n_pos = int(labels.sum())
    n_neg = int((~labels).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    auc = (pos_ranks.sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(max(auc, 1.0 - auc))


def lda_auc_and_jsd(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    mean_a, cov_a = fit_gaussian(a)
    mean_b, cov_b = fit_gaussian(b)
    pooled = 0.5 * (cov_a + cov_b)
    w = np.linalg.pinv(pooled) @ (mean_b - mean_a)
    z = np.vstack([a, b])
    labels = np.concatenate([np.zeros(len(a)), np.ones(len(b))])
    scores = z @ w
    return auc_from_scores(labels, scores), gaussian_jsd(a, b)


def encoded_latents(savedir: Path, x: np.ndarray) -> np.ndarray:
    model_data = load_model(str(savedir))
    model = VAE(model_data.latent_dim, data_dim=model_data.data_dim)
    z = encode(
        jnp.asarray(x), model_data, rng=jax.random.PRNGKey(0), model=model
    )
    return np.asarray(z)


def measure_separability(
    outdir: Path,
    latent_dim: int,
    n_validation: int,
) -> list[dict]:
    ccsne_dir = model_dir(outdir, "ccsne", latent_dim)
    blip_dir = model_dir(outdir, "blip", latent_dim)
    if (
        not (ccsne_dir / MODEL_FNAME).exists()
        or not (blip_dir / MODEL_FNAME).exists()
    ):
        return []

    ccsne_val = np.asarray(TrainValData.load(source="ccsne").val)
    blip_val = np.asarray(TrainValData.load(source="blip").val)
    n = min(n_validation, len(ccsne_val), len(blip_val))
    ccsne_val = ccsne_val[:n]
    blip_val = blip_val[:n]

    ccsne_in_signal = encoded_latents(ccsne_dir, ccsne_val)
    blip_in_signal = encoded_latents(ccsne_dir, blip_val)
    auc, jsd = lda_auc_and_jsd(ccsne_in_signal, blip_in_signal)
    rows = [
        {
            "latent_dim": latent_dim,
            "space": "ccsne_encoder",
            "auc": auc,
            "gaussian_jsd": jsd,
            "n_per_class": n,
        }
    ]

    blip_in_glitch = encoded_latents(blip_dir, blip_val)
    ccsne_in_glitch = encoded_latents(blip_dir, ccsne_val)
    auc, jsd = lda_auc_and_jsd(blip_in_glitch, ccsne_in_glitch)
    rows.append(
        {
            "latent_dim": latent_dim,
            "space": "blip_encoder",
            "auc": auc,
            "gaussian_jsd": jsd,
            "n_per_class": n,
        }
    )
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def numeric_rows(rows: list[dict]) -> list[dict]:
    converted = []
    for row in rows:
        item = dict(row)
        for key, value in row.items():
            if key in {"dataset", "model_dir", "kl_per_dim", "space"}:
                continue
            item[key] = float(value)
        converted.append(item)
    return converted


def grouped_mean(
    rows: list[dict], x_key: str, y_key: str
) -> tuple[np.ndarray, np.ndarray]:
    groups: dict[float, list[float]] = {}
    for row in rows:
        groups.setdefault(float(row[x_key]), []).append(float(row[y_key]))
    xs = np.array(sorted(groups))
    ys = np.array([np.mean(groups[x]) for x in xs])
    return xs, ys


def plot_summary(outdir: Path, figure_path: Path) -> None:
    metrics = numeric_rows(read_csv(outdir / METRICS_CSV))
    separability = numeric_rows(read_csv(outdir / SEPARABILITY_CSV))
    if not metrics:
        raise FileNotFoundError(f"No metrics found at {outdir / METRICS_CSV}")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), constrained_layout=True)

    xs, ys = grouped_mean(metrics, "latent_dim", "total_corr")
    axes[0].plot(xs, ys, "o-", label="total correlation")
    axes[0].set_yscale("symlog", linthresh=1e-2)
    ax0b = axes[0].twinx()
    xs, ys = grouped_mean(metrics, "latent_dim", "max_abs_corr")
    ax0b.plot(xs, ys, "s--", color="C1", label="max |corr|")
    axes[0].set_title("Latent geometry")
    axes[0].set_xlabel("latent dimension")
    axes[0].set_ylabel("total correlation")
    ax0b.set_ylabel("max |corr|")
    axes[0].grid(alpha=0.2, which="both")

    handles_a, labels_a = axes[0].get_legend_handles_labels()
    handles_b, labels_b = ax0b.get_legend_handles_labels()
    axes[0].legend(handles_a + handles_b, labels_a + labels_b, frameon=False)

    if separability:
        xs, ys = grouped_mean(separability, "latent_dim", "auc")
        axes[1].plot(xs, ys, "o-", label="LDA ROC-AUC")
        ax1b = axes[1].twinx()
        xs, ys = grouped_mean(separability, "latent_dim", "gaussian_jsd")
        ax1b.plot(xs, ys, "s--", color="C1", label="Gaussian JSD")
        axes[1].set_ylabel("ROC-AUC")
        ax1b.set_ylabel("JSD (nats)")
        handles_a, labels_a = axes[1].get_legend_handles_labels()
        handles_b, labels_b = ax1b.get_legend_handles_labels()
        axes[1].legend(
            handles_a + handles_b, labels_a + labels_b, frameon=False
        )
    else:
        axes[1].text(0.5, 0.5, "separability not available", ha="center")
    axes[1].set_title("Cross-model separability")
    axes[1].set_xlabel("latent dimension")
    axes[1].grid(alpha=0.2)

    xs, ys = grouped_mean(metrics, "latent_dim", "val_reconstruction_loss")
    axes[2].plot(xs, ys, "o-", label="validation reconstruction")
    ax2b = axes[2].twinx()
    xs, ys = grouped_mean(metrics, "latent_dim", "total_kl")
    ax2b.plot(xs, ys, "s--", color="C1", label="total KL")
    axes[2].set_title("Fit and information")
    axes[2].set_xlabel("latent dimension")
    axes[2].set_ylabel("validation reconstruction loss")
    ax2b.set_ylabel("total KL (nats)")
    axes[2].grid(alpha=0.2)
    handles_a, labels_a = axes[2].get_legend_handles_labels()
    handles_b, labels_b = ax2b.get_legend_handles_labels()
    axes[2].legend(handles_a + handles_b, labels_a + labels_b, frameon=False)

    for ax in axes:
        ax.axvline(5, color="0.25", ls=":", lw=1.2)

    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    print(f"wrote {figure_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-dims", default="2,3,4,5,6,8,12,16")
    parser.add_argument("--datasets", default="ccsne,blip")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--n-validation", type=int, default=1000)
    parser.add_argument(
        "--force", action="store_true", help="retrain existing models"
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="skip training/measurement and replot existing CSVs",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=ASSETS / SUMMARY_FIGURE,
        help="path for the summary PNG",
    )
    args = parser.parse_args()

    latent_dims = parse_int_list(args.latent_dims)
    datasets = parse_str_list(args.datasets)
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        metric_rows = []
        for latent_dim in latent_dims:
            for dataset in datasets:
                savedir = train_if_needed(
                    args.outdir,
                    dataset,
                    latent_dim,
                    args.epochs,
                    args.batch_size,
                    args.force,
                )
                metric_rows.append(measure_model(savedir, dataset, latent_dim))
        write_csv(args.outdir / METRICS_CSV, metric_rows)

        separability_rows = []
        for latent_dim in latent_dims:
            separability_rows.extend(
                measure_separability(
                    args.outdir, latent_dim, args.n_validation
                )
            )
        write_csv(args.outdir / SEPARABILITY_CSV, separability_rows)

    plot_summary(args.outdir, args.figure)


if __name__ == "__main__":
    main()
