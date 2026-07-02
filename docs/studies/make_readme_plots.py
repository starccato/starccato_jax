"""Regenerate the figures embedded in README.rst.

This script uses the public package APIs and shipped default model artifacts.
It records the numerical values used in captions to
``docs/studies/readme_plot_metrics.csv`` and writes figures to ``docs/assets``.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from starccato_jax.data import TrainValData
from starccato_jax.waveforms import StarccatoBlip, StarccatoCCSNe

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs" / "assets"
METRICS = Path(__file__).resolve().parent / "readme_plot_metrics.csv"
ASSETS.mkdir(parents=True, exist_ok=True)


def latent_norms(encoder, waveforms):
    z = np.asarray(encoder.encode(waveforms))
    return np.linalg.norm(z, axis=1)


def histogram_overlap(a, b, bins=60):
    values = np.concatenate([a, b])
    lo, hi = np.percentile(values, [1, 99])
    edges = np.geomspace(max(lo, 1e-6), hi, bins)
    ha, _ = np.histogram(a, bins=edges, density=True)
    hb, _ = np.histogram(b, bins=edges, density=True)
    widths = np.diff(edges)
    return float(np.sum(np.minimum(ha, hb) * widths)), edges


def reconstruction_mse(model, x):
    xhat = np.asarray(model.reconstruct(x))
    return np.mean((xhat - x) ** 2, axis=1)


def pca2(z):
    z0 = z - z.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(z0, full_matrices=False)
    return z0 @ vt[:2].T


def write_metrics(rows):
    with METRICS.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    metric_rows = []

    signal_vae = StarccatoCCSNe()
    glitch_vae = StarccatoBlip()

    # Generated waveform examples.
    n_waveforms = 6
    signal_x = np.asarray(signal_vae.generate(n=n_waveforms))
    glitch_x = np.asarray(glitch_vae.generate(n=n_waveforms))

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True, sharey=True)
    for row in signal_x:
        axes[0].plot(row, lw=1.4, alpha=0.8)
    for row in glitch_x:
        axes[1].plot(row, lw=1.4, alpha=0.8)
    axes[0].set_title("Signal VAE samples (CCSNe)")
    axes[1].set_title("Glitch VAE samples (blips)")
    for ax in axes:
        ax.set_xlabel("sample index")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("standardized strain")
    fig.tight_layout()
    fig.savefig(ASSETS / "waveform_samples.png", dpi=180)
    plt.close(fig)

    # Latent norm overlap on generated samples.
    n_generated = 3000
    signal_generated = np.asarray(signal_vae.generate(n=n_generated))
    glitch_generated = np.asarray(glitch_vae.generate(n=n_generated))

    signal_in_signal = latent_norms(signal_vae, signal_generated)
    glitch_in_signal = latent_norms(signal_vae, glitch_generated)
    glitch_in_glitch = latent_norms(glitch_vae, glitch_generated)
    signal_in_glitch = latent_norms(glitch_vae, signal_generated)

    signal_overlap, signal_edges = histogram_overlap(
        signal_in_signal, glitch_in_signal
    )
    glitch_overlap, glitch_edges = histogram_overlap(
        glitch_in_glitch, signal_in_glitch
    )
    metric_rows.extend(
        [
            {"metric": "signal_latent_norm_overlap", "value": signal_overlap},
            {"metric": "glitch_latent_norm_overlap", "value": glitch_overlap},
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    axes[0].hist(
        signal_in_signal,
        bins=signal_edges,
        density=True,
        histtype="step",
        lw=2.2,
        label="signal -> signal VAE",
    )
    axes[0].hist(
        glitch_in_signal,
        bins=signal_edges,
        density=True,
        histtype="step",
        lw=2.2,
        label="glitch -> signal VAE",
    )
    axes[0].set_title(f"Signal latent space\noverlap={signal_overlap:.2f}")
    axes[1].hist(
        glitch_in_glitch,
        bins=glitch_edges,
        density=True,
        histtype="step",
        lw=2.2,
        label="glitch -> glitch VAE",
    )
    axes[1].hist(
        signal_in_glitch,
        bins=glitch_edges,
        density=True,
        histtype="step",
        lw=2.2,
        label="signal -> glitch VAE",
    )
    axes[1].set_title(f"Glitch latent space\noverlap={glitch_overlap:.2f}")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("latent vector norm")
        ax.set_ylabel("density")
        ax.grid(alpha=0.2, which="both")
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(ASSETS / "embedding_overlap.png", dpi=180)
    plt.close(fig)

    # Cross-reconstruction score on validation data.
    signal_val = np.asarray(TrainValData.load(source="ccsne").val)
    glitch_val = np.asarray(TrainValData.load(source="blip").val)
    n_val = min(len(signal_val), len(glitch_val), 350)
    signal_val = signal_val[:n_val]
    glitch_val = glitch_val[:n_val]

    signal_by_signal = reconstruction_mse(signal_vae, signal_val)
    signal_by_glitch = reconstruction_mse(glitch_vae, signal_val)
    glitch_by_signal = reconstruction_mse(signal_vae, glitch_val)
    glitch_by_glitch = reconstruction_mse(glitch_vae, glitch_val)

    signal_score = np.log10(signal_by_glitch / signal_by_signal)
    glitch_score = np.log10(glitch_by_glitch / glitch_by_signal)
    signal_accuracy = float(np.mean(signal_score > 0))
    glitch_accuracy = float(np.mean(glitch_score < 0))
    balanced_accuracy = 0.5 * (signal_accuracy + glitch_accuracy)
    metric_rows.extend(
        [
            {"metric": "cross_recon_n_validation", "value": n_val},
            {
                "metric": "cross_recon_signal_accuracy",
                "value": signal_accuracy,
            },
            {
                "metric": "cross_recon_glitch_accuracy",
                "value": glitch_accuracy,
            },
            {
                "metric": "cross_recon_balanced_accuracy",
                "value": balanced_accuracy,
            },
            {
                "metric": "cross_recon_signal_score_median",
                "value": np.median(signal_score),
            },
            {
                "metric": "cross_recon_glitch_score_median",
                "value": np.median(glitch_score),
            },
            {
                "metric": "median_mse_signal_by_signal",
                "value": np.median(signal_by_signal),
            },
            {
                "metric": "median_mse_signal_by_glitch",
                "value": np.median(signal_by_glitch),
            },
            {
                "metric": "median_mse_glitch_by_signal",
                "value": np.median(glitch_by_signal),
            },
            {
                "metric": "median_mse_glitch_by_glitch",
                "value": np.median(glitch_by_glitch),
            },
        ]
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), constrained_layout=True)
    bins = np.linspace(
        np.percentile(np.concatenate([signal_score, glitch_score]), 1),
        np.percentile(np.concatenate([signal_score, glitch_score]), 99),
        60,
    )
    axes[0].hist(
        signal_score,
        bins=bins,
        histtype="step",
        lw=2.2,
        density=True,
        label="held-out CCSNe",
    )
    axes[0].hist(
        glitch_score,
        bins=bins,
        histtype="step",
        lw=2.2,
        density=True,
        label="held-out blips",
    )
    axes[0].axvline(0, color="0.25", ls="--", lw=1.5, label="equal MSE")
    axes[0].set_title(
        f"Cross-reconstruction score\nbalanced accuracy={balanced_accuracy:.2f}"
    )
    axes[0].set_xlabel("log10(glitch-VAE MSE / signal-VAE MSE)")
    axes[0].set_ylabel("density")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].grid(alpha=0.2)

    labels = [
        "CCSNe\nsignal VAE",
        "CCSNe\nglitch VAE",
        "Blips\nsignal VAE",
        "Blips\nglitch VAE",
    ]
    values = [
        np.median(signal_by_signal),
        np.median(signal_by_glitch),
        np.median(glitch_by_signal),
        np.median(glitch_by_glitch),
    ]
    axes[1].bar(labels, values, color=["C0", "C1", "C0", "C1"], alpha=0.75)
    axes[1].set_yscale("log")
    axes[1].set_title("Median reconstruction MSE")
    axes[1].set_ylabel("MSE")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.2, which="both")
    fig.savefig(ASSETS / "cross_reconstruction_score.png", dpi=180)
    plt.close(fig)

    # PCA projection of full latent vectors.
    n_pca = 2500
    signal_generated = np.asarray(signal_vae.generate(n=n_pca))
    glitch_generated = np.asarray(glitch_vae.generate(n=n_pca))
    signal_space_z = np.vstack(
        [
            np.asarray(signal_vae.encode(signal_generated)),
            np.asarray(signal_vae.encode(glitch_generated)),
        ]
    )
    glitch_space_z = np.vstack(
        [
            np.asarray(glitch_vae.encode(glitch_generated)),
            np.asarray(glitch_vae.encode(signal_generated)),
        ]
    )
    signal_space_xy = pca2(signal_space_z)
    glitch_space_xy = pca2(glitch_space_z)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    axes[0].scatter(
        signal_space_xy[:n_pca, 0],
        signal_space_xy[:n_pca, 1],
        s=5,
        alpha=0.25,
        label="signal waveforms",
    )
    axes[0].scatter(
        signal_space_xy[n_pca:, 0],
        signal_space_xy[n_pca:, 1],
        s=5,
        alpha=0.25,
        label="glitch waveforms",
    )
    axes[0].set_title("Signal VAE latent PCA")
    axes[0].set_xlabel("PC 1")
    axes[0].set_ylabel("PC 2")
    axes[0].legend(frameon=False, markerscale=3)
    axes[0].grid(alpha=0.2)

    axes[1].scatter(
        glitch_space_xy[:n_pca, 0],
        glitch_space_xy[:n_pca, 1],
        s=5,
        alpha=0.25,
        label="glitch waveforms",
    )
    axes[1].scatter(
        glitch_space_xy[n_pca:, 0],
        glitch_space_xy[n_pca:, 1],
        s=5,
        alpha=0.25,
        label="signal waveforms",
    )
    axes[1].set_title("Glitch VAE latent PCA")
    axes[1].set_xlabel("PC 1")
    axes[1].set_ylabel("PC 2")
    axes[1].legend(frameon=False, markerscale=3)
    axes[1].grid(alpha=0.2)
    fig.savefig(ASSETS / "latent_pca_projection.png", dpi=180)
    plt.close(fig)

    write_metrics(metric_rows)
    print(f"wrote figures to {ASSETS}")
    print(f"wrote metrics to {METRICS}")


if __name__ == "__main__":
    main()
