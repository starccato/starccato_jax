import jax
import jax.numpy as jnp
import pytest

from starccato_jax.credible_intervals import pointwise_ci, uniform_ci


@pytest.fixture
def sample_post_pred():
    # Use a fixed key for reproducibility
    key = jax.random.PRNGKey(42)
    n_samples, n_xpoints = 1000, 50
    # Generate standard normal samples with JAX
    return jax.random.normal(key, shape=(n_samples, n_xpoints))


def test_pointwise_ci(sample_post_pred):
    qtiles = pointwise_ci(sample_post_pred, ci=0.9)
    # Check that the output has the correct shape (n_xpoints,)
    assert qtiles.shape == (3, sample_post_pred.shape[1])
    # Check that for each x, lower is less than upper
    assert jnp.all(qtiles[0] < qtiles[-1])


def test_uniform_ci_shape(sample_post_pred):
    qtiles = uniform_ci(sample_post_pred, ci=0.9)
    # Check that the output has the correct shape (n_xpoints,)
    assert qtiles.shape == (3, sample_post_pred.shape[1])
    # Check that for each x, lower is less than upper
    assert jnp.all(qtiles[0] < qtiles[-1])
