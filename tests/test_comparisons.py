import os

import h5py
from utils import BRANCH

from starccato_jax import StarccatoVAE
from starccato_jax.plotting import plot_distributions


def _save_signals(outdir, signals):
    with h5py.File(os.path.join(outdir, "vae_signals.h5"), "w") as f:
        f.create_dataset("signals", data=signals)


def test_gan_comparison(
    outdir, gan_signals, richers_signals, cached_vae_signals
):
    vae = StarccatoVAE()
    vae_signal = vae.generate(n=len(gan_signals))
    _save_signals(outdir, vae_signal)

    # rescale all signals to have the same scale
    gan_signals = (gan_signals - gan_signals.mean()) / gan_signals.std()
    vae_signal = (vae_signal - vae_signal.mean()) / vae_signal.std()
    richers_signals = (
        richers_signals - richers_signals.mean()
    ) / richers_signals.std()
    cached_vae_signals = (
        cached_vae_signals - cached_vae_signals.mean()
    ) / cached_vae_signals.std()

    plot_distributions(
        richers_signals,
        gan_signals[: len(richers_signals)],
        labels=["Richers", "GAN"],
        fname=os.path.join(outdir, "richers_vs_gan.png"),
        title="Richers vs GAN",
    )
    plot_distributions(
        richers_signals,
        vae_signal[: len(richers_signals)],
        labels=["Richers", "VAE"],
        fname=os.path.join(outdir, "richers_vs_vae.png"),
        title="Richers vs VAE",
    )
    plot_distributions(
        gan_signals,
        vae_signal,
        labels=["GAN", "VAE"],
        fname=os.path.join(outdir, "gan_vs_vae.png"),
        title="GAN vs VAE",
    )
    plot_distributions(
        cached_vae_signals,
        vae_signal,
        labels=["cached VAE[main]", f"VAE[{BRANCH}]"],
        fname=os.path.join(outdir, "vae_vs_cached.png"),
        title=f"Cached vs {BRANCH}",
    )
