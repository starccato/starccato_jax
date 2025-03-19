import os

import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from matplotlib.gridspec import GridSpec
from scipy.spatial.distance import jensenshannon

from .utils import TIME


def _plot_quantiles(x, y, ax, color, label=None):
    # qtils 90, 95, 99 (upper and lower)
    # qtls_vals = [0.75, 0.90, 0.99]
    qtls_vals = [0.75, 0.99]
    for q in qtls_vals:
        qtl = np.quantile(y, [1 - q, q], axis=0)
        ax.fill_between(x, qtl[0], qtl[1], alpha=0.2, color=color, lw=0)
    ax.plot(x, np.quantile(y, 0.5, axis=0), color=color, lw=2, label=label)
    ax.set_ylabel("Strain [1/Hz]")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-11, 7)
    ax.set_yticks([-10, -5, 0, 5])


def _plot_jsd(x, ax, dataset, vae_dataset):
    # compute JSD between the distributions of the two datasets
    n = len(x)
    jsd = np.zeros(n)
    for i in range(n):
        d1, d2 = dataset[:, i], vae_dataset[:, i]
        bins = np.linspace(
            min(d1.min(), d2.min()), max(d1.max(), d2.max()), 50
        )
        hist_1, _ = np.histogram(d1, bins=bins, density=True)
        hist_2, _ = np.histogram(d2, bins=bins, density=True)
        jsd[i] = jensenshannon(hist_1, hist_2)
    # net JSD
    mean_jsd = np.mean(jsd)
    ax.text(
        0.95,
        0.95,
        f"Mean JSD = {mean_jsd:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    ax.plot(x, jsd, color="tab:red", lw=2)
    ax.set_ylabel("JSD [nats]")
    ax.set_xlim(x[0], x[-1])


def _plot_mse(x, ax, dataset, vae_dataset):
    n = len(x)
    err = np.array(
        [(dataset[:, i] - vae_dataset[:, i]) ** 2 for i in range(n)]
    )
    mse = np.mean(err)
    err_qtl = np.quantile(err, [0.25, 0.5, 0.75], axis=1)

    # annotate the MSE in top right corner
    ax.text(
        0.95,
        0.95,
        f"MSE = {mse:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
    )
    ax.plot(x, err_qtl[1], color="tab:red", lw=2)
    ax.fill_between(
        x, err_qtl[0], err_qtl[2], alpha=0.2, color="tab:red", lw=0
    )
    ax.set_ylabel("Error$^2$")
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(bottom=0)


def plot_distributions(
    dataset: jnp.ndarray,
    vae_dataset: jnp.ndarray,
    fname=None,
    title=None,
    labels=["Raw Data", "VAE Data"],
):
    x = TIME.copy()

    # make a grid spec (3 rows, 40%, 40%, 20%)
    fig = plt.figure(figsize=(3.5, 4))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[4, 2, 2])
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)

    _plot_quantiles(x, dataset, ax0, color="tab:gray", label=labels[0])
    _plot_quantiles(x, vae_dataset, ax0, color="tab:orange", label=labels[1])
    _plot_jsd(x, ax1, dataset, vae_dataset)
    _plot_mse(x, ax2, dataset, vae_dataset)

    # ensure that the ax0 doesnt have xtick labels but ax2 does (shared x axis)
    ax0.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    ax1.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    ax0.legend(frameon=False, loc="upper right")

    ax2.set_xlabel("Time (s)")
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

    if title is not None:
        ax0.set_title(title, pad=10)

    if fname is not None:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        plt.savefig(fname, bbox_inches="tight")
        plt.close(fig)

    return fig, [ax0, ax1, ax2]
