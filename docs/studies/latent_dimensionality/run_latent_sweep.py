"""Train VAE latent-dimension sweeps and plot the recorded measurements.

This script is intentionally heavier than the README examples: it trains the
CCSNe and blip VAEs at each requested latent dimension, records metrics to CSV,
then creates the summary figure from those CSVs. It does not contain any
pre-baked results.

Example full run:

    uv run python docs/studies/latent_dimensionality/run_latent_sweep.py \
      --latent-dims 4,5,6,8 \
      --seeds 0,1,2 \
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
import matplotlib.pyplot as plt  # noqa: E402

from starccato_jax.data import TrainValData  # noqa: E402
from starccato_jax.vae import Config  # noqa: E402
from starccato_jax.vae.core import (  # noqa: E402
    VAE,
    decoder_collision_summary,
    encode_mean,
    generate,
    load_model,
    load_model_metadata,
    reconstruction_fidelity,
    summarize_decoder_geometry,
    summarize_fidelity,
    train_vae,
)
from starccato_jax.vae.core.io import (  # noqa: E402
    LOSS_FNAME,
    MODEL_FNAME,
    load_loss_h5,
)
from starccato_jax.vae.core.metrics import compute_latent_stats  # noqa: E402

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


def model_dir(
    outdir: Path,
    dataset: str,
    latent_dim: int,
    capacity_end: float,
    seed: int,
) -> Path:
    capacity_label = str(capacity_end).replace(".", "p")
    return (
        outdir
        / "models"
        / f"{dataset}_z{latent_dim}_c{capacity_label}_seed{seed}"
    )


def train_if_needed(
    outdir: Path,
    dataset: str,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    capacity_end: float,
    seed: int,
    data_seed: int,
    force: bool,
) -> Path:
    target = model_dir(outdir, dataset, latent_dim, capacity_end, seed)
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
        capacity_end=capacity_end,
        capacity_warmup_epochs=min(500, epochs),
        beta_capacity=5.0,
        seed=seed,
        data_seed=data_seed,
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


def measure_model(
    savedir: Path,
    dataset: str,
    latent_dim: int,
    seed: int,
    data_seed: int,
    sample_rate: float,
    flow: float,
    fmax: float,
) -> dict:
    model_data = load_model(str(savedir))
    model = VAE(
        model_data.latent_dim,
        data_dim=model_data.data_dim,
        normalize_decoder_output=model_data.normalize_decoder_output,
    )
    data = TrainValData.load(source=dataset, seed=data_seed)
    rng = jax.random.PRNGKey(0)
    stats = compute_latent_stats(model_data.params, model, data.val, rng)

    losses = load_loss_h5(str(savedir / LOSS_FNAME))
    artifact = load_model_metadata(str(savedir))
    training_metadata = artifact["artifact_metadata"].get("training", {})
    best_epoch = int(training_metadata.get("best_epoch", 0))
    idx = (
        best_epoch - 1
        if best_epoch > 0
        else last_recorded_index(losses.val_metrics.loss)
    )
    kl_per_dim = np.asarray(stats["kl_per_dim"], dtype=float)
    z_mean = encode_mean(data.val, model_data, model=model)
    reconstruction = generate(model_data, z_mean, model=model)
    fidelity = summarize_fidelity(
        reconstruction_fidelity(
            np.asarray(data.val),
            np.asarray(reconstruction),
            sample_rate=sample_rate,
            flow=flow,
            fmax=fmax,
        )
    )
    prior_z = np.random.default_rng(seed + 20260718).normal(
        size=(128, latent_dim)
    )
    prior_geometry = summarize_decoder_geometry(model_data, prior_z)
    validation_geometry = summarize_decoder_geometry(
        model_data, np.asarray(z_mean)[:128]
    )
    collisions = decoder_collision_summary(model_data, prior_z)

    return {
        "dataset": dataset,
        "latent_dim": latent_dim,
        "training_seed": seed,
        "data_seed": data_seed,
        "model_dir": str(savedir),
        "best_epoch": idx + 1,
        "recorded_epochs": int(
            training_metadata.get(
                "recorded_epochs",
                last_recorded_index(losses.val_metrics.loss) + 1,
            )
        ),
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
        **fidelity,
        "prior_condition_number_median": prior_geometry[
            "condition_number_median"
        ],
        "prior_condition_number_p90": prior_geometry["condition_number_p90"],
        "prior_effective_rank_min": prior_geometry["effective_rank_min"],
        "prior_roundtrip_rmse_median": prior_geometry["roundtrip_rmse_median"],
        "validation_condition_number_median": validation_geometry[
            "condition_number_median"
        ],
        "validation_roundtrip_rmse_median": validation_geometry[
            "roundtrip_rmse_median"
        ],
        "prior_collision_pairs": collisions["collision_pairs"],
        "prior_minimum_mismatch": collisions["minimum_mismatch"],
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
    model = VAE(
        model_data.latent_dim,
        data_dim=model_data.data_dim,
        normalize_decoder_output=model_data.normalize_decoder_output,
    )
    z = encode_mean(jnp.asarray(x), model_data, model=model)
    return np.asarray(z)


def measure_separability(
    outdir: Path,
    latent_dim: int,
    n_validation: int,
    capacity_end: float,
    seed: int,
    data_seed: int,
) -> list[dict]:
    ccsne_dir = model_dir(outdir, "ccsne", latent_dim, capacity_end, seed)
    blip_dir = model_dir(outdir, "blip", latent_dim, capacity_end, seed)
    if (
        not (ccsne_dir / MODEL_FNAME).exists()
        or not (blip_dir / MODEL_FNAME).exists()
    ):
        return []

    ccsne_val = np.asarray(
        TrainValData.load(source="ccsne", seed=data_seed).val
    )
    blip_val = np.asarray(TrainValData.load(source="blip", seed=data_seed).val)
    n = min(n_validation, len(ccsne_val), len(blip_val))
    ccsne_val = ccsne_val[:n]
    blip_val = blip_val[:n]

    ccsne_in_signal = encoded_latents(ccsne_dir, ccsne_val)
    blip_in_signal = encoded_latents(ccsne_dir, blip_val)
    auc, jsd = lda_auc_and_jsd(ccsne_in_signal, blip_in_signal)
    rows = [
        {
            "latent_dim": latent_dim,
            "training_seed": seed,
            "data_seed": data_seed,
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
            "training_seed": seed,
            "data_seed": data_seed,
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


def upsert_csv(
    path: Path, rows: list[dict], key_fields: tuple[str, ...]
) -> None:
    """Merge resumable study rows, replacing matching configurations."""
    if not rows:
        return
    merged = {
        tuple(str(row[field]) for field in key_fields): row
        for row in read_csv(path)
    }
    for row in rows:
        key = tuple(str(row[field]) for field in key_fields)
        merged[key] = row
    write_csv(path, list(merged.values()))


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


def subsets(rows: list[dict], key: str):
    for value in sorted({row[key] for row in rows}):
        yield value, [row for row in rows if row[key] == value]


def plot_summary(outdir: Path, figure_path: Path) -> None:
    metrics = numeric_rows(read_csv(outdir / METRICS_CSV))
    if not metrics:
        raise FileNotFoundError(f"No metrics found at {outdir / METRICS_CSV}")

    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), constrained_layout=True)

    for dataset, rows in subsets(metrics, "dataset"):
        xs, ys = grouped_mean(
            rows, "latent_dim", "prior_condition_number_median"
        )
        axes[0].plot(xs, ys, "o-", label=f"{dataset} condition")
    axes[0].set_yscale("log")
    ax0b = axes[0].twinx()
    for index, (dataset, rows) in enumerate(subsets(metrics, "dataset")):
        xs, ys = grouped_mean(
            rows, "latent_dim", "prior_roundtrip_rmse_median"
        )
        ax0b.plot(
            xs,
            ys,
            "s--",
            color=f"C{index + 2}",
            label=f"{dataset} round-trip RMSE",
        )
    axes[0].set_title("Decoder geometry")
    axes[0].set_xlabel("latent dimension")
    axes[0].set_ylabel("median Jacobian condition number")
    ax0b.set_ylabel("median latent round-trip RMSE")
    axes[0].grid(alpha=0.2, which="both")

    handles_a, labels_a = axes[0].get_legend_handles_labels()
    handles_b, labels_b = ax0b.get_legend_handles_labels()
    axes[0].legend(handles_a + handles_b, labels_a + labels_b, frameon=False)

    for dataset, rows in subsets(metrics, "dataset"):
        xs, ys = grouped_mean(rows, "latent_dim", "val_reconstruction_loss")
        axes[1].plot(xs, ys, "o-", label=f"{dataset} validation MSE")
    axes[1].set_ylabel("validation MSE")
    ax1b = axes[1].twinx()
    for index, (dataset, rows) in enumerate(subsets(metrics, "dataset")):
        dimensions, active = grouped_mean(rows, "latent_dim", "active_dims")
        ax1b.plot(
            dimensions,
            active / dimensions,
            "s--",
            color=f"C{index + 2}",
            label=f"{dataset} active fraction",
        )
    ax1b.set_ylim(0.0, 1.05)
    ax1b.set_ylabel("active latent fraction")
    handles_a, labels_a = axes[1].get_legend_handles_labels()
    handles_b, labels_b = ax1b.get_legend_handles_labels()
    axes[1].legend(handles_a + handles_b, labels_a + labels_b, frameon=False)
    axes[1].set_title("Held-out reconstruction")
    axes[1].set_xlabel("latent dimension")
    axes[1].grid(alpha=0.2)

    for dataset, rows in subsets(metrics, "dataset"):
        xs, ys = grouped_mean(rows, "latent_dim", "mismatch_median")
        axes[2].plot(xs, ys, "o-", label=f"{dataset} median mismatch")
    for index, (dataset, rows) in enumerate(subsets(metrics, "dataset")):
        xs, ys = grouped_mean(rows, "latent_dim", "mismatch_p90")
        axes[2].plot(
            xs,
            ys,
            "s--",
            color=f"C{index + 2}",
            label=f"{dataset} 90th percentile",
        )
    axes[2].set_title("Band-limited fidelity")
    axes[2].set_xlabel("latent dimension")
    axes[2].set_ylabel("300–800 Hz mismatch")
    axes[2].grid(alpha=0.2)
    handles_a, labels_a = axes[2].get_legend_handles_labels()
    axes[2].legend(handles_a, labels_a, frameon=False)

    for ax in axes:
        ax.axvline(5, color="0.25", ls=":", lw=1.2)

    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    print(f"wrote {figure_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-dims", default="4,5,6,8")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--datasets", default="ccsne,blip")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--capacity-end",
        type=float,
        default=12.0,
        help="final total-KL capacity in nats",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--n-validation", type=int, default=1000)
    parser.add_argument("--sample-rate", type=float, default=4096.0)
    parser.add_argument("--flow", type=float, default=300.0)
    parser.add_argument("--fmax", type=float, default=800.0)
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
    seeds = parse_int_list(args.seeds)
    datasets = parse_str_list(args.datasets)
    args.outdir.mkdir(parents=True, exist_ok=True)

    if not args.plot_only:
        metric_rows = []
        for latent_dim in latent_dims:
            for dataset in datasets:
                for seed in seeds:
                    savedir = train_if_needed(
                        args.outdir,
                        dataset,
                        latent_dim,
                        args.epochs,
                        args.batch_size,
                        args.capacity_end,
                        seed,
                        args.data_seed,
                        args.force,
                    )
                    metric_rows.append(
                        measure_model(
                            savedir,
                            dataset,
                            latent_dim,
                            seed,
                            args.data_seed,
                            args.sample_rate,
                            args.flow,
                            args.fmax,
                        )
                    )
        upsert_csv(
            args.outdir / METRICS_CSV,
            metric_rows,
            ("dataset", "latent_dim", "training_seed", "data_seed"),
        )

        separability_rows = []
        for latent_dim in latent_dims:
            for seed in seeds:
                separability_rows.extend(
                    measure_separability(
                        args.outdir,
                        latent_dim,
                        args.n_validation,
                        args.capacity_end,
                        seed,
                        args.data_seed,
                    )
                )
        upsert_csv(
            args.outdir / SEPARABILITY_CSV,
            separability_rows,
            ("latent_dim", "training_seed", "data_seed", "space"),
        )

    plot_summary(args.outdir, args.figure)


if __name__ == "__main__":
    main()
