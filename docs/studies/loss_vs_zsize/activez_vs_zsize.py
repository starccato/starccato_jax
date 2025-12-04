"""
Investigate how many latent dimensions are active as we shrink the latent size.

For each latent size in [16, 15, ..., 2] we:
- Train a VAE on CCSNe and on Blip with the same config.
- Compute per-dimension KL on the training set.
- Count "active" latents with KL >= threshold (default 0.1).
- Plot active count vs latent size for both datasets on one chart.
"""

from __future__ import annotations

import os
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import h5py

from starccato_jax.vae.config import Config
from starccato_jax.vae.core.loss import vae_loss
from starccato_jax.vae.starccato_vae import StarccatoVAE

LATENT_SIZES = [128, 64, 32, 16, 8, 4, 2]
KL_THRESHOLD = 0.1
OUTPUT_DIR = Path(__file__).parent
PLOT_PATH = OUTPUT_DIR / "activez_vs_zsize.png"
LOSS_PLOT_PATH = OUTPUT_DIR / "losses_vs_zsize.png"
DIMS_PLOT_PATH = OUTPUT_DIR / "dims_needed_vs_zsize.png"


def _load_saved_config(model_dir: Path) -> Config | None:
    model_path = model_dir / "model.h5"
    if not model_path.exists():
        return None
    try:
        with h5py.File(model_path, "r") as f:
            cfg_json = f["config"][()].decode("utf-8")
            return Config(**json.loads(cfg_json))
    except Exception:
        return None


def get_model_and_config(
    dataset: str, latent_dim: int, base_config: Config
) -> tuple[StarccatoVAE, Config]:
    """Load existing model if present, otherwise train. Return model and its config."""
    outdir = OUTPUT_DIR / f"runs/{dataset}_z{latent_dim}"
    outdir.mkdir(parents=True, exist_ok=True)
    saved_cfg = _load_saved_config(outdir)
    if (outdir / "model.h5").exists() and saved_cfg is not None:
        model = StarccatoVAE(model_dir=str(outdir))
        cfg = saved_cfg
    else:
        cfg = replace(base_config, latent_dim=latent_dim, dataset=dataset)
        model = StarccatoVAE.train(model_dir=str(outdir), config=cfg, plot_every=np.inf)
    return model, cfg


def kl_stats(
    model: StarccatoVAE, dataset: str, threshold: float = KL_THRESHOLD
) -> tuple[int, np.ndarray, int, int]:
    """
    Compute per-dim KL, active count, and dims needed to reach 80%/90% KL mass.
    """
    data = model._data
    vae = model._model
    ds = Config(dataset=dataset).data.train

    rng = jax.random.PRNGKey(0)
    _, mean, logvar = vae.apply(
        {"params": data.params}, ds, rng, True, method=vae.__call__
    )
    kl_per_dim = 0.5 * (jnp.exp(logvar) + mean**2 - 1.0 - logvar).mean(axis=0)
    kl_sorted = jnp.sort(kl_per_dim)[::-1]
    kl_cumsum = jnp.cumsum(kl_sorted)
    kl_frac = kl_cumsum / (kl_cumsum[-1] + 1e-8)
    n80 = int(jnp.argmax(kl_frac >= 0.80)) + 1
    n90 = int(jnp.argmax(kl_frac >= 0.90)) + 1
    active = int(jnp.sum(kl_per_dim >= threshold))
    return active, np.array(kl_per_dim), n80, n90


def compute_end_losses(
    model: StarccatoVAE, dataset: str, cfg: Config
) -> Tuple[float, float, float, float]:
    """Compute final recon/KL on train/val using the trained params."""
    data = Config(dataset=dataset).data
    rng = jax.random.PRNGKey(0)
    # Use the final beta/capacity values from the schedule
    beta = cfg.beta_schedule[-1]
    capacity = cfg.capacity_schedule[-1]
    losses_train = vae_loss(
        model._data.params,
        data.train,
        rng,
        model._model,
        beta,
        cfg.kl_free_bits,
        cfg.use_capacity,
        capacity,
        cfg.beta_capacity,
        deterministic=True,
    )
    losses_val = vae_loss(
        model._data.params,
        data.val,
        rng,
        model._model,
        beta,
        cfg.kl_free_bits,
        cfg.use_capacity,
        capacity,
        cfg.beta_capacity,
        deterministic=True,
    )
    return (
        float(losses_train.reconstruction_loss),
        float(losses_val.reconstruction_loss),
        float(losses_train.kl_divergence),
        float(losses_val.kl_divergence),
    )


def run_study():
    base_config = Config(
        epochs=1500,
        batch_size=128,
        use_capacity=True,
    )

    results_active: Dict[str, List[Tuple[int, int]]] = {"ccsne": [], "blip": []}
    results_losses: Dict[str, List[Tuple[int, float, float, float, float]]] = {
        "ccsne": [],
        "blip": [],
    }  # (z, recon_train, recon_val, kl_train, kl_val)
    results_dims: Dict[str, List[Tuple[int, int, int, int]]] = {
        "ccsne": [],
        "blip": [],
    }  # (z, n80, n90, n100)

    for dataset in results_active.keys():
        for z_dim in LATENT_SIZES:
            print(f"Processing {dataset} with z={z_dim}")
            model, cfg = get_model_and_config(dataset, z_dim, base_config)
            active, kl_per_dim, n80, n90 = kl_stats(model, dataset)
            recon_tr, recon_val, kl_tr, kl_val = compute_end_losses(
                model, dataset, cfg
            )
            results_active[dataset].append((z_dim, active))
            results_losses[dataset].append((z_dim, recon_tr, recon_val, kl_tr, kl_val))
            results_dims[dataset].append((z_dim, n80, n90, len(kl_per_dim)))

            save_sorted_kl_plot(
                kl_per_dim, dataset=dataset, z_dim=z_dim, threshold=KL_THRESHOLD
            )

            print(
                f"{dataset} z={z_dim}: {active} active dims (>= {KL_THRESHOLD}), "
                f"n80={n80}, n90={n90}"
            )

    plot_active(results_active)
    plot_losses(results_losses)
    plot_dims_needed(results_dims)


def plot_active(results: Dict[str, List[Tuple[int, int]]]):
    plt.figure(figsize=(8, 4))
    for dataset, vals in results.items():
        xs, ys = zip(*sorted(vals, key=lambda x: x[0]))
        plt.plot(xs, ys, marker="o", label=f"{dataset.upper()} Active (>= {KL_THRESHOLD})")
    plt.xlabel("Latent size (z-dim)")
    plt.ylabel("Active dims")
    plt.title("Active latent dimensions vs latent size")
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.gca().invert_xaxis()  # show decreasing z-size left->right
    plt.legend(frameon=False)
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(PLOT_PATH, bbox_inches="tight")
    print(f"Saved plot to {PLOT_PATH}")


def plot_losses(results: Dict[str, List[Tuple[int, float, float, float, float]]]):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    ax_kl, ax_recon = axes  # left: KL, right: recon
    colors = {"ccsne": "tab:blue", "blip": "tab:orange"}
    for dataset, vals in results.items():
        vals_sorted = sorted(vals, key=lambda x: x[0])
        xs = [v[0] for v in vals_sorted]
        recon_tr = [v[1] for v in vals_sorted]
        recon_val = [v[2] for v in vals_sorted]
        kl_tr = [v[3] for v in vals_sorted]
        kl_val = [v[4] for v in vals_sorted]

        color = colors.get(dataset, None)
        ax_recon.plot(xs, recon_tr, marker="o", ls="-", color=color, label=f"{dataset.upper()} Train")
        ax_recon.plot(xs, recon_val, marker="o", ls="--", color=color, label=f"{dataset.upper()} Val")
        ax_kl.plot(xs, kl_tr, marker="o", ls="-", color=color, label=f"{dataset.upper()} Train")
        ax_kl.plot(xs, kl_val, marker="o", ls="--", color=color, label=f"{dataset.upper()} Val")

    ax_kl.set_ylabel(r"KL divergence")
    ax_recon.set_ylabel(r"Recon loss")
    ax_kl.set_xlabel("Latent size (z-dim)")
    ax_recon.set_xlabel("Latent size (z-dim)")
    for ax in (ax_kl, ax_recon):
        ax.invert_xaxis()
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, loc="upper right")
    fig.suptitle("Final epoch losses vs latent size")
    LOSS_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(LOSS_PLOT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {LOSS_PLOT_PATH}")


def plot_dims_needed(results: Dict[str, List[Tuple[int, int, int, int]]]):
    plt.figure(figsize=(8, 4))
    for dataset, vals in results.items():
        vals_sorted = sorted(vals, key=lambda x: x[0])
        xs = [v[0] for v in vals_sorted]
        n80 = [v[1] for v in vals_sorted]
        n90 = [v[2] for v in vals_sorted]
        n100 = [v[3] for v in vals_sorted]
        plt.plot(xs, n80, marker="o", ls="-.", label=f"{dataset.upper()} n80")
        plt.plot(xs, n90, marker="o", ls="--", label=f"{dataset.upper()} n90")
        plt.plot(xs, n100, marker="o", ls=":", label=f"{dataset.upper()} n100")
    plt.gca().invert_xaxis()
    plt.xscale("log", base=2)
    plt.yscale("log", base=2)
    plt.xlabel("Latent size (z-dim)")
    plt.ylabel("Dims to retain KL mass")
    plt.title("Latent dims needed for 80% / 90% / 100% KL")
    plt.legend(frameon=False, loc="upper right")
    plt.grid(alpha=0.2)
    DIMS_PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(DIMS_PLOT_PATH, bbox_inches="tight")
    print(f"Saved plot to {DIMS_PLOT_PATH}")


def save_sorted_kl_plot(
    kl_per_dim: np.ndarray, dataset: str, z_dim: int, threshold: float
):
    kl_sorted = np.sort(np.array(kl_per_dim))[::-1]
    kl_cumsum = np.cumsum(kl_sorted)
    kl_frac = kl_cumsum / (kl_cumsum[-1] + 1e-8)

    fig, (ax_top, ax_frac) = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
    ax_top.plot(np.arange(len(kl_sorted)), kl_sorted, marker="o")
    ax_top.axhline(threshold, color="tab:red", ls="--", alpha=0.6, label="Active threshold")
    ax_top.set_ylabel("KL per dim (sorted)")
    ax_top.legend(frameon=False)

    ax_frac.plot(np.arange(len(kl_frac)), kl_frac, marker="o", color="tab:green")
    ax_frac.axhline(0.8, color="tab:gray", ls="--", alpha=0.6, label="80%")
    ax_frac.axhline(0.9, color="tab:gray", ls=":", alpha=0.6, label="90%")
    ax_frac.set_xlabel("Latent dim (sorted)")
    ax_frac.set_ylabel("Cumulative KL fraction")
    ax_frac.legend(frameon=False, loc="lower right")

    fig.suptitle(f"{dataset.upper()} z={z_dim} KL per dim")
    outdir = OUTPUT_DIR / f"runs/{dataset}_z{z_dim}"
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "kl_sorted.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    run_study()
