import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from uniplot import plot as uplot

from ..vae.core.data_containers import Losses, TrainValMetrics


def plot_training_metrics(
    training_metrics: TrainValMetrics, fname: str = None
):
    fig, (ax_top, ax_recon, ax_kl) = plt.subplots(
        3,
        1,
        figsize=(8, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 2, 2], "hspace": 0},
    )

    metrics = training_metrics
    # Truncate to the actual trained length (e.g., early stopping).
    if hasattr(metrics, "_i"):
        n = metrics._i + 1
    else:
        n = metrics.n

    # Top plot (Beta or Capacity)
    beta = metrics.train_metrics.beta
    has_cyclical_beta = beta.max() - beta.min() > 1e-6
    if metrics.use_capacity:
        ax_top.plot(
            metrics.train_metrics.capacity[:n],
            label="Capacity",
            color="tab:red",
            alpha=1,
            lw=2,
        )
        ax_top.set_ylabel(r"$\mathcal{C}$")
        ax_top.set_ylim(bottom=0)
    elif has_cyclical_beta:
        ax_top.plot(
            beta[:n],
            label="Beta",
            color="tab:red",
            alpha=1,
            lw=2,
        )
        ax_top.set_ylabel(r"$\beta$")
        ax_top.set_ylim(-0.05, 1.05)
        ax_top.set_yticks([0, 1])
    else:
        ax_top.axis("off")

    # Reconstruction loss per epoch
    _plot_series(
        ax_recon,
        metrics.train_metrics.reconstruction_loss[:n],
        metrics.val_metrics.reconstruction_loss[:n],
        ylabel=r"$\mathcal{L}_{\rm Recon}$",
        title="Reconstruction loss",
    )
    ax_recon.set_ylim(bottom=0)
    ax_recon.set_xlim(0, n)

    # KL loss per epoch
    _plot_series(
        ax_kl,
        metrics.train_metrics.kl_divergence[:n],
        metrics.val_metrics.kl_divergence[:n],
        ylabel=r"$\mathcal{L}_{\rm KL}$",
        title="KL divergence",
    )
    ax_kl.set_ylim(bottom=0)
    ax_kl.set_xlabel("Epoch")
    ax_kl.set_xlim(0, n)

    # Remove x ticks from the top and middle plots
    ax_top.tick_params(labelbottom=False)
    ax_recon.tick_params(labelbottom=False)

    if fname is not None:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        plt.savefig(fname, bbox_inches="tight")
        plt.close(fig)


def _plot_series(ax: plt.Axes, train, val, ylabel: str, title: str):
    ax.plot(train, label="Train", color="tab:blue", lw=2)
    ax.plot(val, label="Val", color="tab:orange", lw=2)
    ax.set_ylabel(ylabel)
    # Put the title inside the axes for compactness
    ax.text(
        0.01,
        0.95,
        title,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )
    ax.legend(frameon=False, loc="upper right")


def plot_loss_in_terminal(training_metrics: TrainValMetrics):
    uplot(
        [
            training_metrics.train_metrics.kl_divergence,
            training_metrics.train_metrics.reconstruction_loss,
        ],
        legend_labels=["KL", "MSE"],
        title="Training Loss vs Epoch",
        lines=False,
    )
