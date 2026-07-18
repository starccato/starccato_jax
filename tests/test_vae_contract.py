from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from starccato_jax.vae.core.loss import vae_loss
from starccato_jax.vae.core.model import VAE


class _FixedPosteriorModel:
    reconstruct_from_mean = object()

    def apply(self, variables, x, *args, **kwargs):
        mean = jnp.ones((x.shape[0], 3))
        logvar = jnp.zeros_like(mean)
        return jnp.zeros_like(x), mean, logvar


def test_kl_is_total_nats_per_example():
    result = vae_loss(
        {},
        jnp.zeros((4, 8)),
        jax.random.PRNGKey(0),
        _FixedPosteriorModel(),
        beta=1.0,
    )
    # Each of the three N(1, 1) latent dimensions contributes 0.5 nat.
    assert np.isclose(result.kl_divergence, 1.5)


def test_normalized_decoder_has_fixed_mean_and_rms():
    model = VAE(latents=3, data_dim=32, normalize_decoder_output=True)
    rng = jax.random.PRNGKey(1)
    params = model.init(rng, jnp.ones((2, 32)), rng, True)["params"]
    z = jax.random.normal(jax.random.PRNGKey(2), (16, 3))
    generated = model.apply({"params": params}, z, method=model.generate)

    np.testing.assert_allclose(np.mean(generated, axis=-1), 0.0, atol=1e-6)
    np.testing.assert_allclose(np.std(generated, axis=-1), 1.0, atol=1e-5)


def test_encode_mean_is_deterministic():
    model = VAE(latents=3, data_dim=32, normalize_decoder_output=True)
    rng = jax.random.PRNGKey(3)
    params = model.init(rng, jnp.ones((2, 32)), rng, True)["params"]
    x = jax.random.normal(jax.random.PRNGKey(4), (2, 32))

    first = model.apply({"params": params}, x, method=model.encode_mean)
    second = model.apply({"params": params}, x, method=model.encode_mean)
    np.testing.assert_array_equal(first, second)


def test_deterministic_loss_uses_encoder_mean():
    model = VAE(latents=3, data_dim=32, normalize_decoder_output=True)
    rng = jax.random.PRNGKey(5)
    params = model.init(rng, jnp.ones((2, 32)), rng, True)["params"]
    x = jax.random.normal(jax.random.PRNGKey(6), (4, 32))

    first = vae_loss(
        params,
        x,
        jax.random.PRNGKey(7),
        model,
        beta=1.0,
        deterministic=True,
    )
    second = vae_loss(
        params,
        x,
        jax.random.PRNGKey(8),
        model,
        beta=1.0,
        deterministic=True,
    )
    np.testing.assert_array_equal(
        first.reconstruction_loss, second.reconstruction_loss
    )
    np.testing.assert_array_equal(first.loss, second.loss)


def test_stochastic_loss_still_samples_latent():
    model = VAE(latents=3, data_dim=32, normalize_decoder_output=True)
    rng = jax.random.PRNGKey(9)
    params = model.init(rng, jnp.ones((2, 32)), rng, True)["params"]
    x = jax.random.normal(jax.random.PRNGKey(10), (4, 32))

    first = vae_loss(
        params,
        x,
        jax.random.PRNGKey(11),
        model,
        beta=1.0,
        deterministic=False,
    )
    second = vae_loss(
        params,
        x,
        jax.random.PRNGKey(12),
        model,
        beta=1.0,
        deterministic=False,
    )
    assert not np.isclose(
        first.reconstruction_loss, second.reconstruction_loss
    )
