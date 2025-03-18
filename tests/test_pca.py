import jax
import jax.numpy as jnp
import pytest

from starccato_jax import StarccatoPCA
from starccato_jax.data import load_training_data
from starccato_jax.pca import pca
from starccato_jax.plotting import plot_distributions


def test_pca(outdir):
    starccato_pca = StarccatoPCA()
    train_data, valid_data = load_training_data()
    recon = starccato_pca.reconstruct(train_data)
    plot_distributions(
        train_data,
        recon,
        labels=["Raw Data", "PCA Data"],
        fname=f"{outdir}/training_pca.png",
    )
    recon = starccato_pca.reconstruct(valid_data)
    plot_distributions(
        valid_data,
        recon,
        labels=["Raw Data", "PCA Data"],
        fname=f"{outdir}/validation_pca.png",
    )

    with pytest.raises(ValueError):
        starccato_pca.encode(jnp.zeros((4, 4)))


def test_basic():
    z_dim = 32
    n_samples = 200
    RNG = jax.random.PRNGKey(42)
    x = jax.random.normal(RNG, shape=(n_samples, z_dim * 2))
    state = pca.fit(x, n_components=z_dim)
    z = pca.transform(state, x)
    assert z.shape == (n_samples, z_dim)
    x_recovered = pca.recover(state, z)
    assert x_recovered.shape == x.shape
