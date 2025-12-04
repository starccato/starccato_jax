"""
Cross-model outlier diagnostics for CCSNe vs Blip VAEs.

Processes all saved runs in docs/studies/loss_vs_zsize/runs/*_z*:
- Encodes CCSNe and Blip through each VAE.
- Computes latent norms for in/out separation (ROC-AUC).
- Computes Gaussian JSD, RBF MMD between CCSNe/Blip latents.
- Computes LDA-based separability.
- Logs latent disentanglement proxies from compute_latent_stats.

Outputs JSON summary + plots vs latent size.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from starccato_jax.vae.config import Config
from starccato_jax.vae.core.metrics import compute_latent_stats
from starccato_jax.vae.starccato_vae import StarccatoVAE

RUNS_DIR = Path(__file__).parent / "runs"
OUT_PATH = Path(__file__).parent / "cross_model_outlier_metrics.json"
PLOT_PATH = Path(__file__).parent / "cross_model_outlier_summary.png"
SAMPLE_N = 4000  # subsample per dataset for speed


def encode_dataset(vae: StarccatoVAE, dataset: str, n: int) -> jnp.ndarray:
    data = Config(dataset=dataset).data.train
    if data.shape[0] > n:
        data = data[:n]
    rng = jax.random.PRNGKey(0)
    return vae.encode(data, rng=rng)


def roc_auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    # Labels: 1 = positive (in-domain), 0 = negative (out-of-domain)
    order = np.argsort(scores)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(scores))
    pos = labels == 1
    neg = labels == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    auc = (ranks[pos].sum() - n_pos * (n_pos - 1) / 2) / (n_pos * n_neg)
    return float(auc)


def gaussian_jsd(mu1, cov1, mu2, cov2):
    def kl(mu_a, cov_a, mu_b, cov_b):
        k = mu_a.shape[0]
        cov_b_inv = np.linalg.inv(cov_b)
        diff = (mu_b - mu_a)[:, None]
        term_trace = np.trace(cov_b_inv @ cov_a)
        term_quad = float(np.asarray(diff.T @ cov_b_inv @ diff).item())
        sign_a, logdet_a = np.linalg.slogdet(cov_a)
        sign_b, logdet_b = np.linalg.slogdet(cov_b)
        return 0.5 * (term_trace + term_quad - k + logdet_b - logdet_a)

    cov_m = 0.5 * (cov1 + cov2)
    mu_m = 0.5 * (mu1 + mu2)
    return float(0.5 * (kl(mu1, cov1, mu_m, cov_m) + kl(mu2, cov2, mu_m, cov_m)))


def mmd_rbf(x: np.ndarray, y: np.ndarray):
    xy = np.concatenate([x, y], axis=0)
    dists = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=-1)
    median_dist = np.median(dists)
    sigma2 = median_dist if median_dist > 0 else 1.0
    k = np.exp(-dists / (2 * sigma2))
    nx, ny = x.shape[0], y.shape[0]
    kxx = k[:nx, :nx]
    kyy = k[nx:, nx:]
    kxy = k[:nx, nx:]
    mmd2 = kxx.mean() + kyy.mean() - 2 * kxy.mean()
    return float(mmd2)


def lda_direction(z_pos: np.ndarray, z_neg: np.ndarray):
    mu_pos = z_pos.mean(axis=0)
    mu_neg = z_neg.mean(axis=0)
    cov_pos = np.cov(z_pos, rowvar=False)
    cov_neg = np.cov(z_neg, rowvar=False)
    cov = cov_pos + cov_neg + 1e-6 * np.eye(z_pos.shape[1])
    w = np.linalg.solve(cov, mu_pos - mu_neg)
    return w


def evaluate_cross(vae: StarccatoVAE, dataset_in: str, dataset_out: str, n: int):
    z_in = np.array(encode_dataset(vae, dataset_in, n))
    z_out = np.array(encode_dataset(vae, dataset_out, n))

    # LDA projection scores
    labels = np.concatenate([np.ones(len(z_in)), np.zeros(len(z_out))])
    w = lda_direction(z_in, z_out)
    lda_scores = np.concatenate([z_in @ w, z_out @ w])
    lda_auc = roc_auc_from_scores(lda_scores, labels)

    # Distribution stats
    mu_in = z_in.mean(axis=0)
    cov_in = np.cov(z_in, rowvar=False)
    mu_out = z_out.mean(axis=0)
    cov_out = np.cov(z_out, rowvar=False)
    jsd = gaussian_jsd(mu_in, cov_in, mu_out, cov_out)

    return {
        "lda_auc": lda_auc,
        "jsd": jsd,
    }


def main():
    results = []
    for model_path in sorted(
        [p for p in RUNS_DIR.glob("*_z*") if (p / "model.h5").exists()],
        key=lambda p: int(p.name.split("_z")[-1]),
    ):
        ds = "ccsne" if "ccsne" in model_path.name else "blip"
        other = "blip" if ds == "ccsne" else "ccsne"
        vae = StarccatoVAE(model_dir=str(model_path))
        stats = compute_latent_stats(
            vae._data.params, vae._model, Config(dataset=ds).data.train, jax.random.PRNGKey(0)
        )
        cross = evaluate_cross(vae, dataset_in=ds, dataset_out=other, n=SAMPLE_N)
        entry = {
            "model": ds,
            "latent_dim": vae.latent_dim,
            **stats,
            **{
                f"cross_{k}": float(v) if isinstance(v, (np.floating, np.ndarray)) else float(v)
                for k, v in cross.items()
            },
        }
        entry["kl_per_dim"] = np.array(entry["kl_per_dim"]).tolist()
        results.append(entry)

    OUT_PATH.write_text(json.dumps(results, indent=2))
    print(f"Wrote cross-model metrics to {OUT_PATH}")
    for r in results:
        print(
            f"{r['model'].upper()} z={r['latent_dim']}: "
            f"active={r['active']} n80={r['n80']} n90={r['n90']} "
            f"mean|corr|={r['mean_abs_corr']:.3f} TC={r['total_corr']:.3f} "
            f"lda AUC={r['cross_lda_auc']:.3f} "
            f"JSD={r['cross_jsd']:.3f}"
        )
    plot_metrics(results)


def plot_metrics(results: List[Dict]):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
    metrics = [
        ("cross_lda_auc", "LDA AUC"),
        ("cross_jsd", "Gaussian JSD"),
    ]
    for (metric, label), ax in zip(metrics, axes):
        for ds in ["ccsne", "blip"]:
            ds_results = sorted([r for r in results if r["model"] == ds], key=lambda r: r["latent_dim"])
            xs = [r["latent_dim"] for r in ds_results]
            ys = [r[metric] for r in ds_results]
            ax.plot(xs, ys, marker="o", label=ds.upper())
        ax.set_xscale("log", base=2)
        ax.invert_xaxis()
        ax.set_ylabel(label)
        if metric == "cross_lda_auc":
            ax.axhline(0.5, color="gray", ls="--", alpha=0.4, label="Random (0.5)")
            ax.axhline(0.8, color="gray", ls=":", alpha=0.4, label="Strong (0.8)")
            ax.axhline(0.95, color="gray", ls="-.", alpha=0.4, label="Excellent (0.95)")
            ax.text(0.02, 0.9, "Higher AUC = better separation", transform=ax.transAxes, ha="left", va="top", fontsize=9)
        else:
            ax.axhline(1.0, color="gray", ls="--", alpha=0.4, label="JSD 1")
            ax.axhline(5.0, color="gray", ls=":", alpha=0.4, label="JSD 5")
            ax.axhline(10.0, color="gray", ls="-.", alpha=0.4, label="JSD 10")
            ax.text(0.02, 0.9, "Higher JSD = more separated\n(Gaussian assumption)", transform=ax.transAxes, ha="left", va="top", fontsize=9)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)
    axes[-1].set_xlabel("Latent size (z-dim)")
    fig.suptitle("Cross-model separability vs latent size")
    fig.tight_layout()
    plt.savefig(PLOT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {PLOT_PATH}")
    plot_metrics(results)


if __name__ == "__main__":
    main()
