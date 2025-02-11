import os
from typing import List

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ..io import ModelData
from ..loss import Losses, TrainValMetrics, aggregate_metrics
from ..model import reconstruct

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List

import warnings


def plot_training_metrics(
        training_metrics: List[TrainValMetrics], fname: str = None
):
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(3, 1, height_ratios=[1, 3, 0], hspace=0)  # Small top, large bottom, hidden extra row

    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)
    ax_twin = ax_bottom.twinx()

    n = len(training_metrics)
    metrics = aggregate_metrics(training_metrics)

    # Top plot (Beta values)
    ax_top.plot(metrics.train_metrics.beta, label="Beta", color="tab:red", alpha=1, lw=2)
    ax_top.set_ylabel(r"$\beta$")
    ax_top.set_ylim(-.05, 1.05)
    ax_top.set_yticks([0, 1])

    # Bottom plot (main loss)
    _plot_loss(ax_bottom, ax_twin, metrics.train_metrics, "Train", "tab:blue")
    _plot_loss(ax_bottom, ax_twin, metrics.val_metrics, "Val", "tab:orange")
    ax_bottom.set_xlabel("Epoch")
    ax_bottom.set_ylabel(r"$\mathcal{L}_{\rm Recon} + \beta \mathcal{L}_{\rm KL}$")
    ax_twin.set_ylabel(r"$\mathcal{L}_{\rm Recon}, \mathcal{L}_{\rm KL}$")
    ax_bottom.plot([], [], label=r"Reconstruction $\mathcal{L}_{\rm Recon}$", color="tab:gray", ls=":", alpha=0.5)
    ax_bottom.plot([], [], label=r"KLdiv $ \mathcal{L}_{\rm KL}$", color="tab:gray", ls="--", alpha=0.5)
    ax_bottom.legend(frameon=False, loc="upper right")
    ax_bottom.set_ylim(bottom=0)
    ax_bottom.set_xlim(0, n)



    # Remove x ticks from the top plot
    ax_top.tick_params(labelbottom=False)

    if fname is not None:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        plt.savefig(fname, bbox_inches="tight")

    plt.show()

def _plot_loss(ax: plt.Axes, ax2:plt.Axes, losses: Losses, label: str, color: str):
    ax.plot(losses.loss, label=label, color=color, lw=2)
    ax2.plot(losses.reconstruction_loss, color=color, ls=":", alpha=0.5)
    ax2.plot(losses.kl_divergence, color=color, ls="--", alpha=0.5)


def plot_reconstructions(
    model_data: ModelData,
    val_data: np.ndarray,
    nrows: int = 3,
    fname: str = None,
    title: str = None,
    rng: jax.random.PRNGKey = None,
):
    ncols = nrows
    nsamples = nrows * ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
    axes = axes.flatten()
    for i in range(nsamples):
        recon = reconstruct(val_data[i], model_data, rng, n_reps=100)
        qtls = jnp.quantile(recon, jnp.array([0.025, 0.5, 0.975]), axis=0)
        _add_quantiles(
            axes[i], qtls, "Reconstruction", "tab:orange", y_obs=val_data[i]
        )
        axes[i].set_axis_off()
    axes[-1].legend(frameon=False, loc="lower right")
    plt.subplots_adjust(hspace=0, wspace=0)

    if title is not None:
        plt.suptitle(title)

    if fname is not None:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        plt.savefig(fname)


def _add_quantiles(
    ax: plt.Axes,
    y_ci: np.ndarray,
    label: str,
    color: str,
    alpha: float = 0.5,
    y_obs: np.ndarray = None,
):
    # assert that the y_ci are differnt values (no bug in reconstruction)
    if np.allclose(y_ci[0], y_ci[1]):
        warnings.warn("Quantiles are the same, no uncertainty in reconstruction... SUSPICIOUS")

    _, xlen = y_ci.shape
    x = np.arange(xlen)
    ax.fill_between(
        x, y_ci[0], y_ci[2], color=color, alpha=alpha, label=label, lw=0
    )
    ax.plot(y_ci[1], color=color, lw=1, ls="--")
    if y_obs is not None:
        ax.plot(y_obs, color="black", lw=2, zorder=-1, label="Observed")
        # set ylim _slightly_ above and below the y_obs
        ax.set_ylim(np.min(y_obs) - 0.1, np.max(y_obs) + 0.1)
