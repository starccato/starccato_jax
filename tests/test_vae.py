import os

import jax.numpy as jnp

from starccato_jax.config import Config
from starccato_jax.data import load_data
from starccato_jax.io import load_model
from starccato_jax.model import generate
from starccato_jax.sampler import sample_latent_vars_given_data
from starccato_jax.trainer import train_vae


def test_version():
    from starccato_jax import __version__

    assert isinstance(__version__, str)


def test_train_vae(outdir):
    train_data, val_data = load_data(train_fraction=0.8, clean=True)
    assert train_data.shape == (1411, 256)
    assert isinstance(train_data, jnp.ndarray)

    # Train the VAE model
    train_vae(
        train_data,
        val_data,
        config=Config(latent_dim=8, epochs=10, cyclical_annealing_cycles=0),
        save_dir=outdir,
    )
    assert os.path.exists(outdir)

    # load and use VAE
    signal = generate(load_model(outdir))[0]
    assert signal.shape == (256,)

    sample_latent_vars_given_data(
        data=val_data[10],
        model_path=outdir,
        outdir=outdir,
        num_chains=1,
        num_warmup=20,
        num_samples=100,
        num_temps=8,
    )
    assert os.path.exists(outdir)
