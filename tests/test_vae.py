import os

import jax.numpy as jnp

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


def test_default():
    get_default_weights(clean=True)
    vae = StarccatoVAE()
    signal = vae.generate()[0]
    assert signal.shape == (256,)
    reconstructed = vae.reconstruct(signal)
    assert reconstructed.shape == (256,)
