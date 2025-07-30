import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

from starccato_jax import StarccatoVAE
from starccato_jax.plotting import plot_distributions
from starccato_jax.plotting.plot_distributions import _plot_quantiles
from utils import BRANCH

FS = 4096


def _save_signals(fname, signals):
    with h5py.File(fname, "w") as f:
        f.create_dataset("signals", data=signals)


def standardize(signals):
    signals = (signals - signals.mean()) / signals.std()
    # roll all signals so peak is at center
    peak_idx = np.median(signals, axis=0).argmax()
    signals = np.roll(signals, -peak_idx + len(signals[0]) // 2, axis=1)
    n = len(signals[0])
    t = np.arange(n) / FS
    t = t - t[n // 2]
    return t, signals


def test_comparisons(outdir, gan_signals, richers_signals, cached_vae_signals):
    outdir = os.path.join(outdir, "comparisons")
    os.makedirs(outdir, exist_ok=True)

    vae = StarccatoVAE()
    vae_signals = vae.generate(n=len(gan_signals))
    _save_signals(f"{outdir}/vae_signals[{BRANCH}].h5", vae_signals)

    datasets = dict(
        richers=standardize(richers_signals),
        gan=standardize(gan_signals),
        vae=standardize(vae_signals),
        cached_vae=standardize(cached_vae_signals)
    )

    with h5py.File(f"{outdir}/signal_comparisons.h5", "w") as f:
        f.create_dataset("richers_signals", data=datasets["richers"][1])
        f.create_dataset("gan_signals", data=datasets["gan"][1])
        f.create_dataset("vae_signals", data=datasets["vae"][1])

    fig, axes = plt.subplots(4, 1, figsize=(5, 6), sharex=True)
    for i, (name, (t, signals)) in enumerate(datasets.items()):
        _plot_quantiles(t, signals, axes[i], color=f"C{i}")
        axes[i].set_ylabel(name)
        axes[i].set_yticks([])
    plt.subplots_adjust(hspace=0.)
    plt.savefig(f"{outdir}/signal_comparisons.png", dpi=300, bbox_inches='tight')
