"""plot_vae_model_comparisons.py


This script is used to compare:
    1. GAN generated signals
    2. Richers signals
    3. Cached VAE signals (from the main branch)

"""
from dataclasses import dataclass

import h5py
import numpy as np

from starccato_jax import Config, StarccatoVAE
from starccato_jax.plotting.plot_distributions import plot_distributions


@dataclass
class CachedSignals:
    richers_signals: np.ndarray
    gan_signals: np.ndarray
    vae_signals: np.ndarray

    @classmethod
    def load(cls):
        with h5py.File("cached_signals.h5", "r") as f:
            richers_signals = f["richers_signals"][:]
            gan_signals = f["gan_signals"][:]
            vae_signals = f["vae_signals"][:]
        return cls(richers_signals, gan_signals, vae_signals)

    @property
    def n(self):
        return len(self.richers_signals)


def make_plots(cache: CachedSignals, new_signals: np.ndarray):
    plot_distributions(
        cache.richers_signals,
        new_signals,
        labels=["Richers", "New"],
        fname="plts/richers_vs_new.png",
        title="Richers vs New",
    )
    plot_distributions(
        cache.vae_signals,
        new_signals,
        labels=["VAE", "New"],
        fname="plts/vae_vs_new.png",
        title="VAE vs New",
    )
    plot_distributions(
        cache.gan_signals,
        new_signals,
        labels=["GAN", "New"],
        fname="plts/gan_vs_new.png",
        title="GAN vs New",
    )
    plot_distributions(
        cache.richers_signals,
        cache.gan_signals,
        labels=["Richers", "GAN"],
        fname="plts/richers_vs_gan.png",
        title="Richers vs GAN",
    )
    plot_distributions(
        cache.richers_signals,
        cache.vae_signals,
        labels=["GAN", "VAE"],
        fname="plts/gan_vs_vae.png",
        title="GAN vs VAE",
    )


def main():
    config = Config(latent_dim=8, epochs=10, cyclical_annealing_cycles=0)
    vae = StarccatoVAE.train(
        model_dir="model_out", config=config, track_gradients=False
    )
    cached_sigals = CachedSignals.load()
    vae_signals = vae.generate(n=cached_sigals.n)
    make_plots(cached_sigals, vae_signals)


if __name__ == "__main__":
    main()
