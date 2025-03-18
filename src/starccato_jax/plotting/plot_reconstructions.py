import os
from typing import Tuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from .. import credible_intervals
from ..vae.core import ModelData, reconstruct
from .utils import MODEL_COL, TIME, add_quantiles


def plot_reconstructions(
    model_data: ModelData,
    val_data: np.ndarray,
    nrows: int = 3,
    fname: str = None,
    title: str = None,
    rng: jax.random.PRNGKey = None,
    uniform_ci: bool = False,
):
    ncols = nrows
    nsamples = nrows * ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
    axes = axes.flatten()
    for i in range(nsamples):
        recon = reconstruct(val_data[i], model_data, rng, n_reps=100)
        if uniform_ci:
            qtls = credible_intervals.uniform_ci(recon, ci=0.9)
        else:
            qtls = credible_intervals.pointwise_ci(recon, ci=0.9)
        add_quantiles(
            axes[i], qtls, "Reconstruction", MODEL_COL, y_obs=val_data[i]
        )
        axes[i].set_axis_off()
    axes[-1].legend(frameon=False, loc="lower right")
    plt.subplots_adjust(hspace=0, wspace=0)

    if title is not None:
        plt.suptitle(title)

    if fname is not None:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        plt.savefig(fname)
        plt.close(fig)


def plot_model(
    ax=None,
    n: int = 1,
    x: jnp.ndarray = None,
    xstar: jnp.ndarray = None,
    ci: float = None,
    uniform_ci: bool = False,
    color: str = "tab:orange",
) -> Tuple[plt.Figure, plt.Axes]:
    """Makes plots with the Starccato VAE model.

    If only n is provided, n Z samples will be randomly generated.

    Z and X cant be provided at the same time.
    If Z is provided, the Z will be used to generate X*
    If X is provided, the X will be used to reconstruct X*
    If X and n are provided, X* are reconstructed n times (different RNG)

    If CI is provided, the confidence interval will be plotted.
    If uniform_ci is True, the uniform CI will be plotted (otherswise pointwise CI).

    """
    if ax is None:
        fig, ax = plt.subplots()
    fig = ax.get_figure()
    if x is not None:
        ax.plot(TIME, x.ravel(), lw=1, color="black")

    if ci is not None:
        if uniform_ci:
            qtls = credible_intervals.uniform_ci(xstar, ci=ci)
        else:
            qtls = credible_intervals.pointwise_ci(xstar, ci=ci)
        add_quantiles(ax, qtls, color=color, x=TIME)
    else:
        lw = 0.05 if n > 50 else 1
        alpha = 0.25 if n > 50 else 1
        for i in range(n):
            ax.plot(TIME, xstar[i], lw=lw, alpha=alpha, color=color)

    return fig, ax
