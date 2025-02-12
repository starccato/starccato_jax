import os

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import pytest
from jax import jit
from jax.scipy.stats import multivariate_normal, norm
from lartillot_model import LartillotModel

from starccato_jax.sampler.stepping_stone_evidence import (
    stepping_stone_evidence,
)
from starccato_jax.sampler.utils import beta_spaced_samples


def plot_estimate(lnz, lnz_err, true_lnZ, outdir):
    plt.figure(figsize=(2, 3))
    plt.errorbar(
        0,
        lnz,
        yerr=lnz_err,
        fmt="o",
        color="tab:orange",
        label="Stepping Stone",
    )
    plt.axhline(true_lnZ, color="r", label="True LnZ")
    plt.legend(frameon=False)
    plt.ylabel("LnZ")
    plt.xticks([])
    plt.savefig(os.path.join(outdir, "lnz_stepping_stone_comparison.png"))


def test_lartillot_model():
    assert np.isclose(
        LartillotModel(p=20, v=0.01).lnZ, -46.15, atol=0.01
    ), "True log evidence should be close to -46.15"


def test_stepping_stone_evidence(outdir):
    lartillot_model = LartillotModel(p=20, v=0.01)
    rng = random.PRNGKey(0)
    ntemps = 32
    betas = beta_spaced_samples(ntemps, 0.3, 1)
    lnl_chains = lartillot_model.generate_lnl_chains(1000, betas, rng).T

    """Ensure stepping stone evidence runs without errors."""
    ln_z, ln_z_err = stepping_stone_evidence(lnl_chains, betas, outdir, rng)
    plot_estimate(ln_z, ln_z_err, lartillot_model.lnZ, outdir)

    assert np.isfinite(ln_z), "Log evidence should be finite"
    assert np.isfinite(ln_z_err), "Log evidence uncertainty should be finite"

    assert np.isclose(
        float(ln_z), float(lartillot_model.lnZ), atol=0.1
    ), "Log evidence should match the true value within tolerance"

    assert np.isfinite(ln_z_err), "Log evidence uncertainty should be finite"
    assert ln_z_err >= 0, "Log evidence uncertainty should be non-negative"
    plot_path = os.path.join(outdir, "lnz_stepping_stone.png")
    assert os.path.exists(plot_path), "Plot file should be created"
