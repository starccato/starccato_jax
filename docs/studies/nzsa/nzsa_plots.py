"""
Utility script to produce figs for NZSA:

- Load and visualize the glitch (blip) and signal (ccsne) datasets.
- Train two separate VAEs (one per dataset) and keep their training artifacts
  (loss curves, reconstruction frames, GIFs).
- Compare signals and glitches side-by-side.
- Cross-encode examples to see how "outlier-ish" they look in the other model.

Run directly:
    PYTHONPATH=src python docs/studies/nzsa/nzsa_plots.py
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.random import PRNGKey

from starccato_jax.vae.config import Config
from starccato_jax.vae.core import encode, load_model, train_vae
from starccato_jax.vae.core.data_containers import ModelData
from starccato_jax.data import TrainValData

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def load_datasets() -> Tuple[TrainValData, TrainValData]:
    """Load signal and glitch datasets using the library loader."""
    signal_data = TrainValData.load(source="ccsne")
    glitch_data = TrainValData.load(source="blip")
    return signal_data, glitch_data


def plot_dataset_examples(data: np.ndarray, name: str, fname: Path, n: int = 6):
    """Plot a grid of example waveforms."""
    n = min(n, data.shape[0])
    nrows = int(np.ceil(n / 3))
    ncols = 3
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 2 * nrows))
    axes = axes.flatten()
    for i in range(n):
        axes[i].plot(data[i])
        axes[i].set_title(f"{name} sample {i}")
        axes[i].set_axis_off()
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle(f"{name} examples")
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Training helpers
# --------------------------------------------------------------------------- #
def maybe_train(dataset: str, save_dir: Path, force: bool = False) -> ModelData:
    """
    Train a VAE for the given dataset unless artifacts already exist.
    Returns the loaded ModelData.
    """
    model_path = save_dir / "model.h5"
    if model_path.exists() and not force:
        return load_model(save_dir)

    cfg = Config(
        dataset=dataset,
        epochs=1000,
        batch_size=32,
        latent_dim=32,
        learning_rate=1e-3,
        cyclical_annealing_cycles=3,
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    train_vae(cfg.data, config=cfg, save_dir=str(save_dir), plot_every=50)
    return load_model(save_dir)


# --------------------------------------------------------------------------- #
# Analysis helpers
# --------------------------------------------------------------------------- #
def compare_waveforms(signal: np.ndarray, glitch: np.ndarray, fname: Path):
    """Side-by-side plot of a signal and a blip glitch."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    axes[0].plot(signal, color="tab:blue")
    axes[0].set_title("Signal example")
    axes[1].plot(glitch, color="tab:orange")
    axes[1].set_title("Blip glitch example")
    for ax in axes:
        ax.set_axis_off()
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def cross_encode_outlier_plot(
    model_data: ModelData,
    ref_data: np.ndarray,
    other_sample: np.ndarray,
    title: str,
    fname: Path,
    max_ref: int = 512,
):
    """
    Encode a reference dataset and a cross-domain sample; highlight how far
    the cross sample sits relative to the reference latent distribution.
    """
    rng = PRNGKey(0)
    model = None  # model will be constructed inside encode/reconstruct helpers
    ref_batch = ref_data[: max_ref]
    ref_z = encode(ref_batch, model_data, rng=rng, model=model)
    sample_z = encode(other_sample[None, :], model_data, rng=rng, model=model)

    ref_mean = jnp.mean(ref_z, axis=0)
    ref_std = jnp.std(ref_z, axis=0) + 1e-8
    zscore = jnp.abs((sample_z - ref_mean) / ref_std).mean()

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(
        np.asarray(ref_z)[:, 0],
        np.asarray(ref_z)[:, 1],
        alpha=0.4,
        s=8,
        label="reference latents",
    )
    ax.scatter(
        np.asarray(sample_z)[0, 0],
        np.asarray(sample_z)[0, 1],
        color="red",
        s=60,
        label=f"cross sample (|z| z-score ~ {float(zscore):.2f})",
    )
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.legend()
    ax.set_title(title)
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def main(force: bool = False):
    signal_data, glitch_data = load_datasets()

    # Quick data snapshots
    plot_dataset_examples(signal_data.train, "Signal (CCSNe)", ARTIFACT_DIR / "signal_examples.png")
    plot_dataset_examples(glitch_data.train, "Blip Glitches", ARTIFACT_DIR / "glitch_examples.png")
    compare_waveforms(
        signal_data.train[0],
        glitch_data.train[0],
        ARTIFACT_DIR / "signal_vs_glitch.png",
    )

    # Train (or load) VAEs
    signal_dir = ARTIFACT_DIR / "signal_vae"
    glitch_dir = ARTIFACT_DIR / "glitch_vae"
    signal_model = maybe_train("ccsne", signal_dir, force=force)
    glitch_model = maybe_train("blip", glitch_dir, force=force)

    # Cross-encoding diagnostics
    cross_encode_outlier_plot(
        model_data=signal_model,
        ref_data=signal_data.train,
        other_sample=glitch_data.train[0],
        title="Glitch encoded by Signal VAE",
        fname=ARTIFACT_DIR / "glitch_in_signal_vae.png",
    )
    cross_encode_outlier_plot(
        model_data=glitch_model,
        ref_data=glitch_data.train,
        other_sample=signal_data.train[0],
        title="Signal encoded by Glitch VAE",
        fname=ARTIFACT_DIR / "signal_in_glitch_vae.png",
    )

    print("Artifacts written to:", ARTIFACT_DIR)
    print("Training outputs (loss curves, GIFs) live in:", signal_dir, "and", glitch_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NZSA study plots and training")
    parser.add_argument(
        "--force", action="store_true", help="Force re-training even if model.h5 exists"
    )
    args = parser.parse_args()
    main(force=args.force)
