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
from starccato_jax.plotting.plot_reconstructions import plot_reconstructions

import os 

HERE = os.path.dirname(__file__)


@dataclass
class CachedSignals:
    richers_signals: np.ndarray
    gan_signals: np.ndarray
    vae_signals: np.ndarray

    @classmethod
    def load(cls):
        with h5py.File(f"{HERE}/signal_comparisons.h5", "r") as f:
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
        fname=f"{HERE}/plts/richers_vs_new.png",
        title="Richers vs New",
    )
    plot_distributions(
        cache.vae_signals,
        new_signals,
        labels=["VAE", "New"],
        fname=f"{HERE}/plts/vae_vs_new.png",
        title="VAE vs New",
    )
    plot_distributions(
        cache.gan_signals,
        new_signals,
        labels=["GAN", "New"],
        fname=f"{HERE}/plts/gan_vs_new.png",
        title="GAN vs New",
    )
    
    ## delete later?
    plot_distributions(
        cache.richers_signals,
        cache.gan_signals,
        labels=["Richers", "GAN"],
        fname=f"{HERE}/plts/richers_vs_gan.png",
        title="Richers vs GAN",
    )
    plot_distributions(
        cache.richers_signals,
        cache.vae_signals,
        labels=["Richers", "VAE"],
        fname=f"{HERE}/plts/richers_vs_vae.png",
        title="Richers vs VAE",
    )
    
#    plot_reconstructions(
#        model_data = ModelData,
#        val_data = np.ndarray,
#        nrows = 3,
#        rng: jax.random.PRNGKey = None
#    )

#n = len(val_data)
#zs = jax.random.normal(jax.random.PRNGKey(0), (n, starccato_vae.latent_dim))

#generated_signal = starccato_vae.generate(z=zs)
#for i in range(n):
#    kwgs = dict(lw=0.1, alpha=0.1)
#    plt.plot(generated_signal[i], color="tab:orange", **kwgs)
#    plt.plot(val_data[i], color="k", **kwgs)



def main():
    config = Config(latent_dim=64, 
                    epochs=5000, 
                    cyclical_annealing_cycles=0,
                    learning_rate=1e-3,
                    batch_size=32)
    vae = StarccatoVAE.train(
        model_dir=f"{HERE}/model_out", config=config, track_gradients=True
    )
    cached_sigals = CachedSignals.load()
    vae_signals = vae.generate(n=cached_sigals.n)
    make_plots(cached_sigals, vae_signals)


if __name__ == "__main__":
    main()


# Notes:
# Need to use training set (not full Richers) to compare
# Lowered batch size
# Batchnorm


# Try convolutional layers
# Try recurrent layers (GRU)
# Try transformer