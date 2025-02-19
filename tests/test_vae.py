import os

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from starccato_jax import Config, StarccatoVAE
from starccato_jax.data import get_default_weights, load_training_data


def test_version():
    from starccato_jax import __version__

    assert isinstance(__version__, str)


def test_training_data():
    train_data, val_data = load_training_data(train_fraction=0.8, clean=True)
    assert train_data.shape == (1411, 256)
    assert isinstance(train_data, jnp.ndarray)


def test_train_vae(outdir):
    # Train the VAE model
    vae = StarccatoVAE.train(
        model_dir=outdir,
        train_fraction=0.8,
        config=Config(latent_dim=8, epochs=10, cyclical_annealing_cycles=0),
    )
    assert os.path.exists(outdir)

    # load and use VAE
    signal = vae.generate()[0]
    assert signal.shape == (256,)


def test_default(outdir):
    get_default_weights(clean=True)
    vae = StarccatoVAE()
    signal = vae.generate()[0]
    assert signal.shape == (256,)
    reconstructed = vae.reconstruct(signal)
    assert reconstructed.shape == (256,)


def test_plotting(outdir):
    vae = StarccatoVAE()
    signal = vae.generate()[0]

    fig, axes = plt.subplots(2, 1, figsize=(4, 6), sharex=True)
    ax = axes[0]
    vae.plot(ax, n=3)
    vae.plot(ax, n=100, ci=0.9)
    vae.plot(ax, n=100, ci=0.5)
    fig.savefig(os.path.join(outdir, "plot_generated_sample.png"))

    ax = axes[1]
    coverage = vae.reconstruction_coverage(signal, n=100)
    vae.plot(ax, n=100, x=signal, ci=0.9, uniform_ci=True)
    vae.plot(ax, n=100, x=signal, ci=0.9, uniform_ci=False, color="tab:blue")
    ax.set_title(f"Reconstruction coverage: {coverage:.2f}")
    fig.savefig(os.path.join(outdir, "plot_reconstruction_cis.png"))
    plt.close(fig)
