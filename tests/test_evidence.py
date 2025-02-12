import os

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import pytest
from jax import jit
from jax.scipy.stats import norm

from starccato_jax.sampler.stepping_stone_evidence import stepping_stone_evidence

import matplotlib.pyplot as plt


### Simple "Lartillot" LnL model from https://arxiv.org/pdf/1810.04488

@jit
def lnl_const(p, v):
    return -(p / 2) * jnp.log(2 * jnp.pi * v)


@jit
def true_lnZ(p, v):
    return (p / 2) * jnp.log(v / (1 + v)) + lnl_const(p, v)


@jit
def log_like(theta, p, v):
    return -jnp.sum(theta ** 2 / (2 * v)) + lnl_const(p, v)


@jit
def log_prior(theta):
    return jnp.sum(norm.logpdf(theta, loc=0, scale=1))


def simulate_posterior_samples(p, v, nsamples, key):
    mean = jnp.zeros(p)
    cov_matrix = v / (v + 1) * jnp.eye(p)
    return random.multivariate_normal(key, mean=mean, cov=cov_matrix, shape=(nsamples,))


@pytest.fixture
def lartillot_data(tmp_path):
    """Generate synthetic posterior samples and temperatures for testing."""
    p, v = 5, 2.0
    nsamples = 1000
    ntemps = 10  # Number of inverse temperatures
    betas = jnp.linspace(0, 1, ntemps)  # Annealing schedule

    rng = jax.random.PRNGKey(42)
    samples = simulate_posterior_samples(p, v, nsamples, rng)

    # THIS ISNT QUITE CORRECT...
    # Compute log-likelihoods for each sample at each beta
    ln_likes = jnp.array([[log_like(theta, p, v)  for theta in samples] for beta in betas]).T

    return {
        "ln_likes": ln_likes,
        "betas": betas,
        "true_lnZ": true_lnZ(p, v),
        "rng": rng,
    }


def plot_estimate(lnz, lnz_err, true_lnZ, outdir):
    plt.errorbar(0, lnz, yerr=lnz_err, fmt='o', color='tab:orange', label='Stepping Stone')
    plt.axhline(true_lnZ, color='r', label='True LnZ')
    plt.legend(frameon=False)
    plt.ylabel("LnZ")
    plt.xticks([])
    plt.savefig(os.path.join(outdir, "lnz_stepping_stone_comparison.png"))


def test_stepping_stone_evidence(lartillot_data, outdir):
    """Ensure stepping stone evidence runs without errors."""
    ln_z, ln_z_err = stepping_stone_evidence(
        lartillot_data["ln_likes"], lartillot_data["betas"], outdir, lartillot_data["rng"]
    )
    plot_estimate(ln_z, ln_z_err, lartillot_data["true_lnZ"], outdir)




    assert np.isfinite(ln_z), "Log evidence should be finite"
    assert np.isfinite(ln_z_err), "Log evidence uncertainty should be finite"

    assert np.isclose(
        float(ln_z), float(lartillot_data["true_lnZ"]),
        atol=0.5
    ), "Log evidence should match the true value within tolerance"

    assert np.isfinite(ln_z_err), "Log evidence uncertainty should be finite"
    assert ln_z_err >= 0, "Log evidence uncertainty should be non-negative"
    plot_path = os.path.join(outdir, "lnz_stepping_stone.png")
    assert os.path.exists(plot_path), "Plot file should be created"
