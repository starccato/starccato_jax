import os

import arviz as az
import jax
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from starccato_pca import StarccatoPCA
from starccato_sampler.sampler import sample

from starccato_jax import StarccatoVAE
from starccato_jax.data import load_training_data
from starccato_jax.plotting.utils import TIME

RNG = jax.random.PRNGKey(0)


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

    vae_data = az.from_netcdf("out_vae/inference.nc")
    pca_data = az.from_netcdf("out_pca/inference.nc")
    vae_qtls = vae_data.sample_stats["quantiles"].values
    pca_qtls = pca_data.sample_stats["quantiles"].values

    # gredspec of 80 20
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

    ax0.plot(TIME, data, label="Data", color="gray", lw=2, alpha=0.5)
    ax0.plot(TIME, true, label="True", color="black", lw=1)
    _add_qtl(ax0, vae_qtls, "tab:blue", "VAE (90% CI)")
    _add_qtl(ax0, pca_qtls, "tab:orange", "PCA (90% CI)")
    ax0.legend(frameon=False)

    # error qtls
    vae_err = vae_qtls - true
    pca_err = pca_qtls - true
    _add_qtl(ax1, vae_err, "tab:blue", "VAE error")
    _add_qtl(ax1, pca_err, "tab:orange", "PCA error")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Error")

    # remove space between subplots
    plt.subplots_adjust(hspace=0)

    # add some padding for the label before saving
    plt.savefig("comparison.pdf", bbox_inches="tight")
