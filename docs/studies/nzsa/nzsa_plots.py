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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from jax.random import PRNGKey

from starccato_jax.plotting import generate_gif
from starccato_jax.vae.config import Config
from starccato_jax.vae.core import VAE, encode, load_model, reconstruct, train_vae
from starccato_jax.vae.core.data_containers import ModelData
from starccato_jax.data import TrainValData

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Global plot style for a cleaner look
mpl.rcParams.update(
    {
        "figure.figsize": (6, 3),
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linestyle": ":",
        "grid.alpha": 0.3,
        "lines.linewidth": 1.4,
        "savefig.dpi": 200,
    }
)

SIGNAL_COLOR = "#1f77b4"
GLITCH_COLOR = "#d55e00"
COLOR_ORIG = "#222222"
COLOR_RECON = "#1f77b4"
COLOR_RESID = "#d62728"


def _color_for_label(label: str) -> str:
    l = label.lower()
    if "signal" in l:
        return SIGNAL_COLOR
    if "glitch" in l:
        return GLITCH_COLOR
    return COLOR_RECON


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
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.4 * nrows))
    axes = axes.flatten()
    for i in range(n):
        axes[i].plot(data[i], color=COLOR_ORIG)
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
    ax.plot(np.arange(x.shape[-1]), x, lw=1.2, color=COLOR_ORIG)
    ax.set_title(title)
    ax.set_xlabel("Time (arb)")
    ax.set_ylabel("Amplitude (norm.)")
    fig.tight_layout(pad=1.2)
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
        batch_size=64,
        latent_dim=32,
        learning_rate=1e-3,
        cyclical_annealing_cycles=1,
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
    show_reconstruction: bool = False,
):
    """
    Encode a reference dataset and a cross-domain sample; highlight how far
    the cross sample sits relative to the reference latent distribution.
    """
    model = VAE(latents=model_data.latent_dim, data_dim=_model_data_dim(model_data))
    rng = PRNGKey(0)
    ref_batch = ref_data[: max_ref]
    ref_z = encode(ref_batch, model_data, rng=rng, model=model)
    sample_z = encode(other_sample[None, :], model_data, rng=rng, model=model)

    ref_mean = jnp.mean(ref_z, axis=0)
    ref_std = jnp.std(ref_z, axis=0) + 1e-8
    zscore = jnp.abs((sample_z - ref_mean) / ref_std).mean()

    if show_reconstruction:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1.3, 1]})
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=(5, 4))

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
    ax.set_title(f"{title}\n|z|-zscore≈{float(zscore):.2f}")
    if show_reconstruction:
        recon = reconstruct(other_sample, model_data, rng=rng, model=model)
        recon = np.asarray(recon)
        if recon.ndim > 1:
            recon = recon[0]
        axes[1].plot(other_sample, color=COLOR_ORIG, label="Original")
        axes[1].plot(recon, color=COLOR_RECON, alpha=0.9, label="Recon")
        y_min, y_max = _compute_ylim(np.vstack([other_sample, recon]))
        axes[1].set_ylim(y_min, y_max)
        axes[1].set_title("Cross reconstruction")
        axes[1].legend(frameon=False)
        axes[1].set_xlabel("Time (arb)")
        axes[1].set_ylabel("Amplitude")
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def cross_reconstruction_plot(
    sample: np.ndarray,
    model_data: ModelData,
    title: str,
    fname: Path,
    rng: PRNGKey = None,
):
    """Plot original sample and its reconstruction under a cross model."""
    rng = rng if rng is not None else PRNGKey(0)
    model = VAE(latents=model_data.latent_dim, data_dim=_model_data_dim(model_data))
    recon = reconstruct(sample, model_data, rng=rng, model=model)
    recon = np.asarray(recon)
    if recon.ndim > 1:
        recon = recon[0]

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    t = np.arange(sample.shape[-1])
    ax.plot(t, sample, color=COLOR_ORIG, label="Original")
    ax.plot(t, recon, color=COLOR_RECON, alpha=0.9, label="Reconstruction")
    y_min, y_max = _compute_ylim(np.vstack([sample, recon]))
    ax.set_ylim(y_min, y_max)
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    ax.set_xlabel("Time (arb)")
    ax.set_ylabel("Amplitude (norm.)")
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=1.1)
    fig.savefig(fname)
    plt.close(fig)


def reconstruction_with_residuals(
    sample: np.ndarray,
    recon: np.ndarray,
    title: str,
    fname: Path,
):
    """Plot original, reconstruction, and residual."""
    if recon.ndim > 1:
        recon = recon[0]
    resid = sample - recon
    t = np.arange(sample.shape[-1])
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 6), sharex=True)
    axes[0].plot(t, sample, color=COLOR_ORIG)
    axes[0].set_title("Original")
    axes[1].plot(t, recon, color=COLOR_RECON)
    axes[1].set_title("Reconstruction")
    axes[2].plot(t, resid, color=COLOR_RESID)
    axes[2].axhline(0, color="#888", lw=0.8, ls="--")
    axes[2].set_title("Residual (orig - recon)")
    axes[2].set_xlabel("Time (arb)")
    y_min, y_max = _compute_ylim(np.vstack([sample, recon]))
    axes[0].set_ylim(y_min, y_max)
    axes[1].set_ylim(y_min, y_max)
    axes[2].set_ylim(*_compute_ylim(resid[None, :]))
    fig.suptitle(title)
    fig.tight_layout(pad=1.2)
    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fname)
    plt.close(fig)


def latent_distribution_plot(
    model_data: ModelData,
    data_a: np.ndarray,
    data_b: np.ndarray,
    labels: tuple[str, str],
    fname: Path,
    rng: PRNGKey = None,
    clip_percentile: float = 99.0,
    show_zscore_hist: bool = False,
    model_type: str = "CCSNe",
):
    """Scatter z[0], z[1] for two datasets under a given model."""
    rng = rng if rng is not None else PRNGKey(0)
    model = VAE(latents=model_data.latent_dim, data_dim=_model_data_dim(model_data))
    z_a = encode(data_a, model_data, rng=rng, model=model)
    z_b = encode(data_b, model_data, rng=rng, model=model)
    z_a = np.asarray(z_a)
    z_b = np.asarray(z_b)

    # Define robust limits to de-emphasize extreme outliers.
    all_x = np.concatenate([z_a[:, 0], z_b[:, 0]])
    all_y = np.concatenate([z_a[:, 1], z_b[:, 1]])
    x_lo, x_hi = np.percentile(all_x, [100 - clip_percentile, clip_percentile])
    y_lo, y_hi = np.percentile(all_y, [100 - clip_percentile, clip_percentile])
    x_pad = 0.05 * (x_hi - x_lo)
    y_pad = 0.05 * (y_hi - y_lo)
    x_lim = (x_lo - x_pad, x_hi + x_pad)
    y_lim = (y_lo - y_pad, y_hi + y_pad)

    def _scatter(ax, z, label, color):
        in_bounds = (
            (z[:, 0] >= x_lim[0]) & (z[:, 0] <= x_lim[1]) & (z[:, 1] >= y_lim[0]) & (z[:, 1] <= y_lim[1])
        )
        ax.scatter(z[in_bounds, 0], z[in_bounds, 1], alpha=0.45, s=10, label=label, color=color)
        if np.any(~in_bounds):
            # draw outliers at clipped edges
            clipped = np.column_stack(
                [np.clip(z[~in_bounds, 0], *x_lim), np.clip(z[~in_bounds, 1], *y_lim)]
            )
            ax.scatter(clipped[:, 0], clipped[:, 1], marker="x", s=16, color=color, alpha=0.6, label=None)

    if show_zscore_hist:
        fig, (ax, ax_hist) = plt.subplots(
            1, 2, figsize=(8, 4.2), gridspec_kw={"width_ratios": [2, 1], "wspace": 0.3}
        )
    else:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        ax_hist = None

    _scatter(ax, z_a, labels[0], color=_color_for_label(labels[0]))
    _scatter(ax, z_b, labels[1], color=_color_for_label(labels[1]))
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel("z[0]")
    ax.set_ylabel("z[1]")
    ax.legend(frameon=False)
    ax.grid(True, which="both", alpha=0.25, linestyle=":")
    ax.set_title(f"{model_type} Latent space distribution")

    if ax_hist is not None:
        # z-score magnitude relative to combined mean/std (first two dims)
        all_z = np.vstack([z_a[:, :2], z_b[:, :2]])
        mu = np.mean(all_z, axis=0)
        sigma = np.std(all_z, axis=0) + 1e-8
        za_z = np.abs((z_a[:, :2] - mu) / sigma).mean(axis=1)
        zb_z = np.abs((z_b[:, :2] - mu) / sigma).mean(axis=1)
        zb_z = np.maximum(zb_z, 1e-6)
        za_z = np.maximum(za_z, 1e-6)
        zmin = float(min(za_z.min(), zb_z.min()))
        zmax = float(max(za_z.max(), zb_z.max(), 3.0))
        bins = np.geomspace(max(zmin, 1e-3), zmax, 25)
        ax_hist.hist(
            za_z,
            bins=bins,
            alpha=0.35,
            color=_color_for_label(labels[0]),
            label=f"{labels[0]} |z|-zscore",
            density=True,
            histtype="stepfilled",
            edgecolor=_color_for_label(labels[0]),
        )
        ax_hist.hist(
            zb_z,
            bins=bins,
            alpha=0.35,
            color=_color_for_label(labels[1]),
            label=f"{labels[1]} |z|-zscore",
            density=True,
            histtype="stepfilled",
            edgecolor=_color_for_label(labels[1]),
        )
        ax_hist.set_xscale("log")
        ax_hist.set_xlabel("|z|-zscore (log)")
        ax_hist.set_ylabel("Density")
        ax_hist.legend(frameon=False, fontsize=9)
        ax_hist.set_title("Outlier score")

    fname.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def reconstruction_quantiles(
    sample: np.ndarray,
    model_data: ModelData,
    title: str,
    fname: Path,
    n_samples: int = 50,
    rng: PRNGKey = None,
):
    """Sample multiple reconstructions and plot quantile bands."""
    rng = rng if rng is not None else PRNGKey(0)
    model = VAE(latents=model_data.latent_dim, data_dim=_model_data_dim(model_data))
    recon_samples = reconstruct(sample, model_data, rng=rng, n_reps=n_samples, model=model)
    recon_samples = np.asarray(recon_samples)
    q_low, q_med, q_hi = np.quantile(recon_samples, [0.1, 0.5, 0.9], axis=0)

    fig, ax = plt.subplots(figsize=(6, 3))
    t = np.arange(sample.shape[-1])
    ax.plot(t, sample, color=COLOR_ORIG, lw=1.2, label="Original")
    ax.fill_between(t, q_low, q_hi, color=COLOR_RECON, alpha=0.18, label="10-90%")
    ax.plot(t, q_med, color=COLOR_RECON, lw=1.0, label="Median recon")
    y_min, y_max = _compute_ylim(np.vstack([sample, q_low, q_hi]))
    ax.set_ylim(y_min, y_max)
    ax.set_title(title)
    ax.set_xlabel("Time (arb)")
    ax.set_ylabel("Amplitude (norm.)")
    ax.legend()
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
    model = VAE(latents=model_data.latent_dim, data_dim=_model_data_dim(model_data))
    outdir.mkdir(parents=True, exist_ok=True)
    rng = jax.random.PRNGKey(seed)
    dims = tuple(dims)
    phases = jax.random.uniform(rng, (len(dims),), minval=0.0, maxval=2 * jnp.pi)

    xs: list[np.ndarray] = []
    zs: list[np.ndarray] = []

    # First generate all frames to derive stable y-limits.
    for i in range(steps):
        t = 2 * jnp.pi * (i / steps)
        z = jnp.zeros((1, model_data.latent_dim))
        for idx, d in enumerate(dims):
            z = z.at[0, d].set(amplitude * jnp.sin(t + phases[idx]))
        x = model.apply({"params": model_data.params}, z, method=model.generate)
        xs.append(np.asarray(x[0]))
        zs.append(np.asarray(z[0]))

    frame_ylim = y_lim if y_lim is not None else _compute_ylim(np.vstack(xs))

    for i, (x_arr, z_arr) in enumerate(zip(xs, zs)):
        frame_path = outdir / f"frame_{i:04d}.png"
        _plot_waveform_with_latent(
            x_arr,
            z_arr,
            dims=dims,
            y_lim=frame_ylim,
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
    show_endpoints: bool = True,
):
    """Encode two samples, interpolate their latents, decode along the path."""
    model = VAE(latents=model_data.latent_dim, data_dim=_model_data_dim(model_data))
    outdir.mkdir(parents=True, exist_ok=True)
    z_a = encode(x_a[None, :], model_data, model=model)
    z_b = encode(x_b[None, :], model_data, model=model)

    xs: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    alphas = np.linspace(0.0, 1.0, steps)

    for alpha in alphas:
        z = (1 - alpha) * z_a + alpha * z_b
        x = model.apply({"params": model_data.params}, z, method=model.generate)
        xs.append(np.asarray(x[0]))
        zs.append(np.asarray(z[0]))

    frame_ylim = y_lim if y_lim is not None else _compute_ylim(np.vstack(xs))
    latent_lim = float(np.max(np.abs(np.vstack(zs)))) + 1.0

    for i, (alpha, x_arr, z_arr) in enumerate(zip(alphas, xs, zs)):
        frame_path = outdir / f"frame_{i:04d}.png"
        _plot_waveform_with_latent(
            x_arr,
            z_arr,
            dims=(0, 1),
            y_lim=frame_ylim,
            fname=frame_path,
            title=f"Interpolation α={float(alpha):.2f}",
            amplitude=latent_lim,
        )

    # Optionally overlay start/end references as a side panel GIF
    if show_endpoints:
        ref_path = outdir / "frames_with_refs"
        ref_path.mkdir(parents=True, exist_ok=True)
        for i, (alpha, x_arr) in enumerate(zip(alphas, xs)):
            frame_path = ref_path / f"frame_{i:04d}.png"
            fig, ax = plt.subplots(figsize=(6, 3))
            t = np.arange(x_arr.shape[-1])
            cmap = mpl.cm.get_cmap("cividis")
            color = cmap(float(alpha))
            ax.plot(t, x_a, color=SIGNAL_COLOR, alpha=0.6, label="Start (Signal)")
            ax.plot(t, x_b, color=GLITCH_COLOR, alpha=0.6, label="End (Glitch)")
            ax.plot(t, x_arr, color=color, lw=1.2, label=f"Interp α={alpha:.2f}")
            ax.set_ylim(frame_ylim)
            ax.set_xlabel("Time (arb)")
            ax.set_ylabel("Amplitude (norm.)")
            ax.legend(frameon=False, fontsize=9)
            fig.tight_layout(pad=1.1)
            fig.savefig(frame_path)
            plt.close(fig)
        generate_gif(
            str(ref_path / "frame_*.png"),
            str(outdir / "latent_interp_refs.gif"),
            duration=100,
            final_pause=800,
        )
        for frame in ref_path.glob("frame_*.png"):
            frame.unlink(missing_ok=True)

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
        show_reconstruction=True,
    )
    cross_encode_outlier_plot(
        model_data=glitch_model,
        ref_data=glitch_data.train,
        other_sample=signal_data.train[0],
        title="Signal encoded by Glitch VAE",
        fname=ARTIFACT_DIR / "signal_in_glitch_vae.png",
        show_reconstruction=True,
    )
    cross_reconstruction_plot(
        sample=glitch_data.train[100],
        model_data=signal_model,
        title="Glitch reconstructed by Signal VAE",
        fname=ARTIFACT_DIR / "glitch_recon_by_signal_vae.png",
    )
    cross_reconstruction_plot(
        sample=signal_data.train[0],
        model_data=glitch_model,
        title="Signal reconstructed by Glitch VAE",
        fname=ARTIFACT_DIR / "signal_recon_by_glitch_vae.png",
    )
    # Residual overlays
    signal_recon = reconstruct(signal_data.train[0], signal_model, model=None)
    glitch_recon = reconstruct(glitch_data.train[0], glitch_model, model=None)
    reconstruction_with_residuals(
        signal_data.train[0],
        np.asarray(signal_recon),
        title="Signal recon & residual (Signal VAE)",
        fname=ARTIFACT_DIR / "signal_residual.png",
    )
    reconstruction_with_residuals(
        glitch_data.train[0],
        np.asarray(glitch_recon),
        title="Glitch recon & residual (Glitch VAE)",
        fname=ARTIFACT_DIR / "glitch_residual.png",
    )
    # Latent distribution overlap
    latent_distribution_plot(
        model_data=signal_model,
        data_a=np.asarray(signal_data.train[:1000]),
        data_b=np.asarray(glitch_data.train[:1000]),
        labels=("Signal", "Glitch"),
        fname=ARTIFACT_DIR / "latent_overlap_signal_model.png",
        show_zscore_hist=True,
    )
    latent_distribution_plot(
        model_data=glitch_model,
        data_a=np.asarray(signal_data.train[:1000]),
        data_b=np.asarray(glitch_data.train[:1000]),
        labels=("Signal", "Glitch"),
        fname=ARTIFACT_DIR / "latent_overlap_glitch_model.png",
        show_zscore_hist=True,
    )
    # Uncertainty bands on reconstructions
    reconstruction_quantiles(
        sample=signal_data.train[0],
        model_data=signal_model,
        title="Signal recon quantiles (Signal VAE)",
        fname=ARTIFACT_DIR / "signal_recon_quantiles.png",
    )
    reconstruction_quantiles(
        sample=glitch_data.train[0],
        model_data=glitch_model,
        title="Glitch recon quantiles (Glitch VAE)",
        fname=ARTIFACT_DIR / "glitch_recon_quantiles.png",
    )

    # Animations
    anim_dir = ARTIFACT_DIR / "animations"
    latent_walk(signal_model, anim_dir / "signal_latent_walk", steps=64, y_lim=signal_ylim)
    latent_walk(glitch_model, anim_dir / "glitch_latent_walk", steps=64, y_lim=glitch_ylim)
    interpolate_samples(
        model_data=signal_model,
        x_a=np.asarray(signal_data.train[20]),
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
