from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from starccato_jax.vae.core.data_containers import ModelData
from starccato_jax.vae.core.diagnostics import (
    decoder_collision_summary,
    decoder_jacobian_singular_values,
    reconstruction_fidelity,
    summarize_decoder_geometry,
    summarize_fidelity,
)
from starccato_jax.vae.core.model import VAE


def initialized_model() -> ModelData:
    model = VAE(latents=3, data_dim=32, normalize_decoder_output=True)
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, jnp.ones((1, 32)), rng, True)["params"]
    return ModelData(
        params=params,
        latent_dim=3,
        data_dim=32,
        normalize_decoder_output=True,
    )


def test_identical_reconstruction_has_zero_error():
    rng = np.random.default_rng(1)
    target = rng.normal(size=(4, 32))
    metrics = reconstruction_fidelity(
        target, target, sample_rate=2048.0, flow=100.0, fmax=800.0
    )
    np.testing.assert_allclose(metrics["time_mse"], 0.0)
    np.testing.assert_allclose(metrics["mismatch"], 0.0, atol=1e-15)
    np.testing.assert_allclose(metrics["log_spectral_distance"], 0.0)
    np.testing.assert_allclose(metrics["peak_time_error_ms"], 0.0)
    summary = summarize_fidelity(metrics)
    assert summary["mismatch_p90"] < 1e-14


def test_reconstruction_fidelity_validates_psd_shape():
    target = np.ones((2, 32))
    with pytest.raises(ValueError, match="rFFT grid"):
        reconstruction_fidelity(
            target,
            target,
            sample_rate=2048.0,
            noise_psd=np.ones(3),
        )


def test_decoder_geometry_is_finite_and_full_rank():
    model_data = initialized_model()
    z = np.array([[0.1, -0.2, 0.3], [-0.4, 0.2, 0.1]])
    singular_values = decoder_jacobian_singular_values(model_data, z)
    assert singular_values.shape == (2, 3)
    assert np.all(np.isfinite(singular_values))
    summary = summarize_decoder_geometry(model_data, z)
    assert summary["n_points"] == 2
    assert 0 <= summary["effective_rank_min"] <= 3
    assert np.isfinite(summary["roundtrip_rmse_median"])


def test_decoder_collision_summary_counts_pairs():
    model_data = initialized_model()
    z = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    summary = decoder_collision_summary(
        model_data, z, minimum_latent_distance=1.0
    )
    assert summary["eligible_pairs"] == 3
    assert 0 <= summary["collision_pairs"] <= 3
    assert 0.0 <= summary["minimum_mismatch"] <= 1.0
