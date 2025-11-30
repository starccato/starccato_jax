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
from typing import Iterable, Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.random import PRNGKey

from starccato_jax.plotting import generate_gif
from starccato_jax.vae.config import Config
from starccato_jax.vae.core import VAE, encode, load_model, train_vae
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


def _plot_waveform(x: np.ndarray, title: str, fname: Path):
    """Utility for saving a single waveform."""
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(x.shape[-1]), x, lw=1.2)
    ax.set_title(title)
    ax.set_xlabel("Time (arb)")
    ax.set_ylabel("Amplitude (norm.)")
    fig.tight_layout()
    fname.parent.mkdir(parents=True, exist_ok=True)
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
# Shape helpers (for older saved ModelData missing data_dim)
# --------------------------------------------------------------------------- #
def _model_data_dim(model_data: ModelData) -> int:
    """Return data_dim, inferring from params if missing."""
    if hasattr(model_data, "data_dim"):
        return model_data.data_dim
    return _infer_data_dim_from_params(model_data.params)


def _infer_data_dim_from_params(params) -> int:
    """Infer input/output dimension from saved parameters."""
    enc = params.get("encoder", {})
    fc1 = enc.get("fc1", {})
    if "kernel" in fc1:
        return fc1["kernel"].shape[0]
    dec = params.get("decoder", {})
    # fallback to last dense layer output dim
    for layer in dec.values():
        if isinstance(layer, dict) and "kernel" in layer:
            return layer["kernel"].shape[1]
    raise ValueError("Could not infer data_dim from saved parameters.")


def _compute_ylim(data: np.ndarray, lower: float = 0.01, upper: float = 0.99, pad_frac: float = 0.1):
    """Fixed y-limits based on data quantiles with a small padding."""
    q_lo, q_hi = np.quantile(data, [lower, upper])
    pad = (q_hi - q_lo) * pad_frac
    return (float(q_lo - pad), float(q_hi + pad))


def _plot_waveform_with_latent(
    x: np.ndarray,
    z: np.ndarray,
    dims: Iterable[int],
    y_lim: tuple[float, float],
    fname: Path,
    title: str,
    amplitude: float,
):
    """Two-panel frame: waveform (fixed y) + current point in latent subspace."""
    dims = tuple(dims)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))

    axes[0].plot(np.arange(x.shape[-1]), x, lw=1.2)
    axes[0].set_ylim(*y_lim)
    axes[0].set_title(title)
    axes[0].set_xlabel("Time (arb)")
    axes[0].set_ylabel("Amplitude")

    # show movement in the first two dims we're animating
    if len(dims) >= 2:
        xdim, ydim = dims[0], dims[1]
    else:
        xdim, ydim = dims[0], dims[0]
    axes[1].scatter(
        np.asarray(z[..., xdim]),
        np.asarray(z[..., ydim]),
        color="red",
        s=60,
    )
    lim = amplitude * 1.2
    axes[1].set_xlim(-lim, lim)
    axes[1].set_ylim(-lim, lim)
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].axvline(0, color="gray", lw=0.5)
    axes[1].set_xlabel(f"z[{xdim}]")
    axes[1].set_ylabel(f"z[{ydim}]")
    axes[1].set_title("Latent walk position")

    fig.tight_layout()
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Animations
# --------------------------------------------------------------------------- #
def latent_walk(
    model_data: ModelData,
    outdir: Path,
    steps: int = 64,
    dims: Iterable[int] = (0, 1, 2),
    amplitude: float = 2.0,
    seed: int = 0,
    y_lim: tuple[float, float] | None = None,
):
    """Animate a smooth walk in a few latent dimensions, decoded to waveform space."""
    model = VAE(latents=model_data.latent_dim)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = jax.random.PRNGKey(seed)
    dims = tuple(dims)
    phases = jax.random.uniform(rng, (len(dims),), minval=0.0, maxval=2 * jnp.pi)

    for i in range(steps):
        t = 2 * jnp.pi * (i / steps)
        z = jnp.zeros((1, model_data.latent_dim))
        for idx, d in enumerate(dims):
            z = z.at[0, d].set(amplitude * jnp.sin(t + phases[idx]))
        x = model.apply({"params": model_data.params}, z, method=model.generate)
        frame_path = outdir / f"frame_{i:04d}.png"
        _plot_waveform_with_latent(
            np.asarray(x[0]),
            np.asarray(z[0]),
            dims=dims,
            y_lim=y_lim if y_lim is not None else _compute_ylim(np.asarray(x[0])),
            fname=frame_path,
            title=f"Latent walk step {i}",
            amplitude=amplitude,
        )

    generate_gif(str(outdir / "frame_*.png"), str(outdir / "latent_walk.gif"), duration=80, final_pause=800)
    for frame in outdir.glob("frame_*.png"):
        frame.unlink(missing_ok=True)


def interpolate_samples(
    model_data: ModelData,
    x_a: np.ndarray,
    x_b: np.ndarray,
    outdir: Path,
    steps: int = 48,
    y_lim: tuple[float, float] | None = None,
):
    """Encode two samples, interpolate their latents, decode along the path."""
    model = VAE(latents=model_data.latent_dim)
    outdir.mkdir(parents=True, exist_ok=True)
    z_a = encode(x_a[None, :], model_data, model=model)
    z_b = encode(x_b[None, :], model_data, model=model)

    for i, alpha in enumerate(jnp.linspace(0.0, 1.0, steps)):
        z = (1 - alpha) * z_a + alpha * z_b
        x = model.apply({"params": model_data.params}, z, method=model.generate)
        frame_path = outdir / f"frame_{i:04d}.png"
        _plot_waveform_with_latent(
            np.asarray(x[0]),
            np.asarray(z[0]),
            dims=(0, 1),
            y_lim=y_lim if y_lim is not None else _compute_ylim(np.asarray(x[0])),
            fname=frame_path,
            title=f"Interpolation α={float(alpha):.2f}",
            amplitude=float(np.max(np.abs(np.concatenate([z_a, z_b])))) + 1.0,
        )

    generate_gif(
        str(outdir / "frame_*.png"),
        str(outdir / "latent_interp.gif"),
        duration=100,
        final_pause=800,
    )
    for frame in outdir.glob("frame_*.png"):
        frame.unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def main(force: bool = False):
    signal_data, glitch_data = load_datasets()
    signal_ylim = _compute_ylim(signal_data.train)
    glitch_ylim = _compute_ylim(glitch_data.train)
    combined_ylim = _compute_ylim(np.concatenate([signal_data.train, glitch_data.train], axis=0))

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

    # Animations
    anim_dir = ARTIFACT_DIR / "animations"
    latent_walk(signal_model, anim_dir / "signal_latent_walk", steps=64, y_lim=signal_ylim)
    latent_walk(glitch_model, anim_dir / "glitch_latent_walk", steps=64, y_lim=glitch_ylim)
    interpolate_samples(
        model_data=signal_model,
        x_a=np.asarray(signal_data.train[0]),
        x_b=np.asarray(glitch_data.train[0]),
        outdir=anim_dir / "interp_signal_model",
        steps=48,
        y_lim=combined_ylim,
    )
    interpolate_samples(
        model_data=glitch_model,
        x_a=np.asarray(glitch_data.train[0]),
        x_b=np.asarray(signal_data.train[0]),
        outdir=anim_dir / "interp_glitch_model",
        steps=48,
        y_lim=combined_ylim,
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
