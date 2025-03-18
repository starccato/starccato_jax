import os
from typing import List, Tuple

import arviz as az
import jax
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from starccato_sampler.sampler import sample

from starccato_jax import StarccatoPCA, StarccatoVAE
from starccato_jax.data import load_training_data
from starccato_jax.plotting.utils import TIME

RNG = jax.random.PRNGKey(0)


def plot_pe_comparison(
    data_1d: jax.numpy.ndarray,
    true_1d: jax.numpy.ndarray,
    result_fnames: List[str],
    labels: List[str],
    colors: List[str],
    fname: str = "comparison.pdf",
):
    results = [az.from_netcdf(f) for f in result_fnames]
    quantiles = [r.sample_stats["quantiles"].values for r in results]
    err = [q - true_1d for q in quantiles]

    fig = plt.figure(figsize=(4, 3))
    gs = GridSpec(
        2,
        1,
        height_ratios=[4, 1],
    )
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax0.set_xlim(TIME[0], TIME[-1])
    ax1.set_xlim(TIME[0], TIME[-1])
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Error")
    plt.subplots_adjust(hspace=0)

    ax0.plot(TIME, data_1d, label="Data", color="gray", lw=2, alpha=0.5)
    ax0.plot(TIME, true_1d, label="True", color="black", lw=1)
    for e, q, lbl, c in zip(err, quantiles, labels, colors):
        _add_qtl(ax0, q, c, lbl)
        _add_qtl(ax1, e, c, lbl)
    ax0.legend(frameon=False)
    plt.savefig(fname, bbox_inches="tight")
    plt.close(fig)


def _add_qtl(ax, qtl, color, lbl):
    ax.fill_between(TIME, qtl[0], qtl[-1], color=color, alpha=0.5, lw=0)
    ax.plot(TIME, qtl[1], color=color, lw=1, ls="-", alpha=0.75, label=lbl)


if __name__ == "__main__":
    train_data, validation_data = load_training_data()
    true = validation_data[0].ravel()
    noise_sigma = 0.1
    data = true + jax.random.normal(RNG, true.shape) * noise_sigma
    data = data.ravel()
    kwgs = dict(data=data, rng_int=0, truth=true)

    # # run MCMC
    for label, model in zip(["pca", "vae"], [StarccatoVAE(), StarccatoPCA()]):
        sample(starccato_model=model, outdir=f"out_{label}", **kwgs)
