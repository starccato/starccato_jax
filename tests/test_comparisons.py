import os

import h5py
import numpy as np
from utils import BRANCH

from starccato_jax import StarccatoVAE
from starccato_jax.plotting import plot_distributions


def _save_signals(fname, signals):
    with h5py.File(fname, "w") as f:
        f.create_dataset("signals", data=signals)


def standardize(signals):
    return (signals - signals.mean()) / signals.std()


def test_comparisons(outdir, gan_signals, richers_signals, cached_vae_signals):
    outdir = os.path.join(outdir, "comparisons")
    os.makedirs(outdir, exist_ok=True)

    vae = StarccatoVAE()
    vae_signal = vae.generate(n=len(gan_signals))
    _save_signals(f"{outdir}/vae_signals[{BRANCH}].h5", vae_signal)

    # rescale all signals to have the same scale
    gan_signals = standardize(gan_signals)
    # roll the GAN signals forward a bit..
    gan_signals = np.roll(gan_signals, 5, axis=1)
    vae_signal = standardize(vae_signal)
    richers_signals = standardize(richers_signals)
    cached_vae_signals = standardize(cached_vae_signals)

    # save
    with h5py.File(f"{outdir}/signal_comparisons.h5", "w") as f:
        f.create_dataset("richers_signals", data=richers_signals)
        f.create_dataset("gan_signals", data=gan_signals)
        f.create_dataset("vae_signals", data=vae_signal)

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
