import os
from typing import List

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from uniplot import plot as uplot

from ..vae.core.data_containers import Losses, TrainValMetrics


def plot_training_metrics(
    training_metrics: TrainValMetrics, fname: str = None
):
    fig = plt.figure(figsize=(8, 4))
    gs = gridspec.GridSpec(
        3, 1, height_ratios=[1, 3, 0], hspace=0
    )  # Small top, large bottom, hidden extra row

    ax_top = fig.add_subplot(gs[0])
    ax_bottom = fig.add_subplot(gs[1], sharex=ax_top)
    ax_twin = ax_bottom.twinx()

    metrics = training_metrics

    # Top plot (Beta values)
    ax_top.plot(
        metrics.train_metrics.beta,
        label="Beta",
        color="tab:red",
        alpha=1,
        lw=2,
    )
    ax_top.set_ylabel(r"$\beta$")
    ax_top.set_ylim(-0.05, 1.05)
    ax_top.set_yticks([0, 1])

    # Bottom plot (main loss)
    _plot_loss(ax_bottom, ax_twin, metrics.train_metrics, "Train", "tab:blue")
    _plot_loss(ax_bottom, ax_twin, metrics.val_metrics, "Val", "tab:orange")
    ax_bottom.set_xlabel("Epoch")
    ax_bottom.set_ylabel(
        r"$\mathcal{L}_{\rm Recon} + \beta \mathcal{L}_{\rm KL}$"
    )
    ax_twin.set_ylabel(r"$\mathcal{L}_{\rm Recon}, \mathcal{L}_{\rm KL}$")
    ax_bottom.plot(
        [],
        [],
        label=r"Reconstruction $\mathcal{L}_{\rm Recon}$",
        color="tab:gray",
        ls=":",
        alpha=0.5,
    )
    ax_bottom.plot(
        [],
        [],
        label=r"KLdiv $ \mathcal{L}_{\rm KL}$",
        color="tab:gray",
        ls="--",
        alpha=0.5,
    )
    ax_bottom.legend(frameon=False, loc="upper right")
    ax_bottom.set_ylim(bottom=0)
    ax_bottom.set_xlim(0, metrics.n)

    # Remove x ticks from the top plot
    ax_top.tick_params(labelbottom=False)

    if fname is not None:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        plt.savefig(fname, bbox_inches="tight")
        plt.close(fig)


def _plot_loss(
    ax: plt.Axes, ax2: plt.Axes, losses: Losses, label: str, color: str
):
    ax.plot(losses.loss, label=label, color=color, lw=2)
    ax2.plot(losses.reconstruction_loss, color=color, ls=":", alpha=0.5)
    ax2.plot(losses.kl_divergence, color=color, ls="--", alpha=0.5)


def plot_loss_in_terminal(training_metrics: List[TrainValMetrics]):
    ys = [
        training_metrics.train_metrics.kl_divergence,
        training_metrics.train_metrics.reconstruction_loss,
    ]
    uplot(
        ys,
        legend_labels=["KL", "MSE"],
        title="Training Loss vs Epoch",
        lines=False,
    )
