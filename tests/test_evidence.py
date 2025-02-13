import os

import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
from lartillot_gaussian import LartillotGaussianModel

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
    plt.ylim(true_lnZ - 2, true_lnZ + 2)
    plt.axhline(true_lnZ, color="r", label="True LnZ")
    plt.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    plt.ylabel("LnZ")
    plt.xticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "lnz_stepping_stone_comparison.png"))


def test_stepping_stone_evidence(outdir):
    lartillot_model = LartillotGaussianModel(d=20, v=0.01)
    rng = random.PRNGKey(0)
    ntemps = 32
    betas = beta_spaced_samples(ntemps, 0.3, 1)
    lnl_chains = lartillot_model.generate_lnl_chains(1000, betas).T

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
