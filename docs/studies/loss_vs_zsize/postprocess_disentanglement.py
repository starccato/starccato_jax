"""
Postprocess saved VAEs to compute disentanglement-style diagnostics:
- Active dims, n80/n90/n100
- Mean/max absolute correlation of latents
- Gaussian total correlation (TC)

Runs over existing models under docs/studies/loss_vs_zsize/runs/*_z*.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from starccato_jax.vae.config import Config
from starccato_jax.vae.core.metrics import compute_latent_stats
from starccato_jax.vae.starccato_vae import StarccatoVAE

RUNS_DIR = Path(__file__).parent / "runs"
OUT_PATH = Path(__file__).parent / "disentanglement_metrics.json"
PLOT_DIR = Path(__file__).parent
KL_THRESHOLD = 0.1


def load_config(model_dir: Path) -> Config:
    with h5py.File(model_dir / "model.h5", "r") as f:
        cfg_json = f["config"][()].decode("utf-8")
    return Config(**json.loads(cfg_json))


def compute_stats(model_dir: Path) -> Dict:
    # Infer dataset and z from path name like ccsne_z10
    name = model_dir.name
    dataset = "ccsne" if "ccsne" in name else "blip"

    cfg = load_config(model_dir)
    model = StarccatoVAE(model_dir=str(model_dir))
    data = Config(dataset=dataset).data.train

    stats = compute_latent_stats(model._data.params, model._model, data, jax.random.PRNGKey(0), kl_threshold=KL_THRESHOLD)

    return dict(
        model_dir=str(model_dir),
        dataset=dataset,
        latent_dim=model._data.latent_dim,
        active=stats["active"],
        n80=stats["n80"],
        n90=stats["n90"],
        n100=stats["n100"],
        mean_abs_corr=stats["mean_abs_corr"],
        max_abs_corr=stats["max_abs_corr"],
        total_corr=stats["total_corr"],
        kl_per_dim=np.array(stats["kl_per_dim"]).tolist(),
    )


def main():
    model_dirs = sorted(
        [p for p in RUNS_DIR.glob("*_z*") if (p / "model.h5").exists()],
        key=lambda p: p.name,
    )
    results = [compute_stats(p) for p in model_dirs]
    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} entries to {OUT_PATH}")
    plot_from_results(results)


def plot_from_results(results: List[Dict]):
    # Scatter TC and corr vs latent_dim, separated by dataset
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    for dataset in ["ccsne", "blip"]:
        ds_results = [r for r in results if r["dataset"] == dataset]
        ds_results = sorted(ds_results, key=lambda r: r["latent_dim"])
        xs = [r["latent_dim"] for r in ds_results]
        tc = [r["total_corr"] for r in ds_results]
        max_corr = [r["max_abs_corr"] for r in ds_results]
        axes[0].plot(xs, tc, marker="o", label=dataset.upper())
        axes[1].plot(xs, max_corr, marker="o", label=dataset.upper())
    # Guides and scaling
    axes[0].set_yscale("log")
    axes[0].axhline(1.0, color="gray", ls="--", alpha=0.4, label="TC=1 (low)")
    axes[0].axhline(10.0, color="gray", ls=":", alpha=0.4, label="TC=10 (higher)")
    axes[1].axhline(0.1, color="gray", ls="--", alpha=0.4, label="|corr|=0.1")
    axes[1].axhline(0.3, color="gray", ls=":", alpha=0.4, label="|corr|=0.3")
    for ax, ylabel in zip(axes, ["Total correlation (log scale)", "Max |corr|"]):
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Latent size (z-dim)")
        ax.set_ylabel(ylabel)
        ax.invert_xaxis()
        ax.legend(frameon=False, loc='lower right')
        ax.grid(alpha=0.2)
    axes[0].text(
        0.02,
        0.9,
        "Lower TC = more factorised\n(log scale)",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    axes[1].text(
        0.02,
        0.9,
        "Off-diag correlation target:\n<0.1 good, >0.3 bad",
        transform=axes[1].transAxes,
        ha="left",
        va="top",
        fontsize=9,
    )
    fig.suptitle("Disentanglement proxies vs latent size")
    fig.tight_layout()
    out = PLOT_DIR / "postprocess_disentanglement_summary.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {out}")


if __name__ == "__main__":
    main()
