import dataclasses
import os
from typing import List

import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
from lartillot_gaussian import LartillotGaussianModel

from starccato_jax.sampler.harmonic_mean_evidence import (
    harmonic_mean_evidence,
    hyper_sphere_evidence,
)
from starccato_jax.sampler.stepping_stone_evidence import (
    stepping_stone_evidence,
)
from starccato_jax.sampler.utils import beta_spaced_samples


@dataclasses.dataclass
class LnZEstimate:
    ln_z: float
    err: float
    label: str = "NA"


def test_stepping_stone_evidence(outdir):
    lartillot_model = LartillotGaussianModel(d=5, v=1)
    rng = random.PRNGKey(0)
    ntemps = 32
    betas = beta_spaced_samples(ntemps, 0.3, 1)
    tempd_chains = lartillot_model.generate_chains(1000, betas)
    chains = lartillot_model.generate_chains(500, nchains=200)

    ss_lnz = LnZEstimate(
        *stepping_stone_evidence(tempd_chains.lnl.T, betas, outdir, rng),
        "Stepping Stone",
    )

    harmonic_lnz = LnZEstimate(
        *harmonic_mean_evidence(chains.samples, chains.lnl), "Harmonic Mean"
    )

    plot_estimate(lartillot_model.lnZ, [ss_lnz, harmonic_lnz], outdir)

    plot_path = os.path.join(outdir, "lnz_comparison.png")

    np.testing.assert_allclose(
        ss_lnz.ln_z,
        lartillot_model.lnZ,
        atol=0.3,
        err_msg="SS Log evidence should match the true value within tolerance",
    )
    np.testing.assert_allclose(
        harmonic_lnz.ln_z,
        lartillot_model.lnZ,
        atol=0.3,
        err_msg="Harmonic Log evidence should match the true value within tolerance",
    )

    assert os.path.exists(plot_path), "Plot file should be created"


def plot_estimate(
    true_lnZ: float,
    lnzs: List[LnZEstimate],
    outdir: str = "out",
):
    plt.figure(figsize=(2 * len(lnzs), 3))

    for i, lnz in enumerate(lnzs):
        plt.errorbar(
            i,
            lnz.ln_z,
            # yerr=lnz.err,
            fmt="o",
            color="tab:orange",
            label=lnz.label,
        )

    ylims = plt.ylim()

    # ensure ylims at least +- 2 from true_lnZ
    if true_lnZ - 2 < ylims[0]:
        plt.ylim(true_lnZ - 2, true_lnZ + 2)

    plt.ylim(true_lnZ - 2, true_lnZ + 2)
    # set tick labels
    plt.xticks(range(len(lnzs)), [lnz.label for lnz in lnzs])
    plt.axhline(true_lnZ, color="r", label="True LnZ")
    plt.ylabel("LnZ")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "lnz_comparison.png"))
