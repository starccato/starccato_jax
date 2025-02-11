import jax
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from jax.random import PRNGKey
from jax.scipy.special import logsumexp
from typing import Tuple


def stepping_stone_evidence(ln_likes: jnp.ndarray, betas: jnp.ndarray, outdir: str, rng: PRNGKey)-> Tuple[float, float]:
    """
    Compute the evidence using the stepping stone approximation.

    See Patricio's paper https://arxiv.org/abs/1810.04488 and
    https://pubmed.ncbi.nlm.nih.gov/21187451/ for details.

    The uncertainty calculation is hopefully combining the evidence in each
    of the steps.

    Returns
    -------
    ln_z: float
        Estimate of the natural log evidence
    ln_z_err: float
        Estimate of the uncertainty in the evidence
    """

    steps, ntemps = ln_likes.shape
    if ntemps > steps:
        raise ValueError("Sus.. ntemps > steps.")

    ln_z, ln_ratio = _calculate_stepping_stone(ln_likes, betas)

    # Patricio's bootstrap method to estimate the evidence uncertainty.
    ll = 50  # Block length
    repeats = 100  # Repeats
    ln_z_realisations = []
    try:
        for _ in range(repeats):
            idxs = jax.random.randint(rng, (steps - ll,), 0, steps - ll)
            ln_z_realisations.append(
                _calculate_stepping_stone(ln_likes[idxs, :], betas)[0]
            )
        ln_z_err = jnp.std(jnp.array(ln_z_realisations))
    except ValueError as e:
        print("Failed to estimate stepping stone uncertainty: ", e)
        ln_z_err = jnp.nan

    _create_stepping_stone_plot(means=ln_ratio, outdir=outdir)

    return ln_z, ln_z_err


def _calculate_stepping_stone(ln_likes: jnp.ndarray, betas: jnp.ndarray)-> Tuple[float, jnp.ndarray]:
    n_samples = ln_likes.shape[0]
    d_betas = betas[1:] - betas[:-1]
    ln_ratio = logsumexp(d_betas * ln_likes[:,:-1], axis=0) - jnp.log(n_samples)
    return jnp.sum(ln_ratio), ln_ratio


def _create_stepping_stone_plot(means:jnp.ndarray, outdir:str):
    n_steps = len(means)

    fig, axes = plt.subplots(nrows=2, figsize=(8, 10))

    ax = axes[0]
    ax.plot(np.arange(1, n_steps + 1), means)
    ax.set_xlabel("$k$")
    ax.set_ylabel("$r_{k}$")

    ax = axes[1]
    ax.plot(np.arange(1, n_steps + 1), np.cumsum(means[::1])[::1])
    ax.set_xlabel("$k$")
    ax.set_ylabel("Cumulative $\\ln Z$")

    plt.tight_layout()
    fig.savefig(f"{outdir}/lnz_stepping_stone.png")
    plt.close()
