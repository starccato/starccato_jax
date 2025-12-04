import os
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np

__all__ = ["plot_latent_kl"]


def plot_latent_kl(
    kl_values: Iterable[float],
    threshold: float = 0.1,
    fname: str | None = None,
    title: str | None = None,
    save_sorted: bool = False,
    sorted_fname: str | None = None,
):
    """
    Plot KL divergence per latent dimension.

    Args:
        kl_values: KL values per dimension (array-like).
        threshold: Threshold above which a latent is considered "active".
        fname: Optional path to save the plot.
        title: Optional title.
        save_sorted: If True, also save a sorted KL + cumulative fraction plot.
        sorted_fname: Optional path for the sorted plot (required if save_sorted).
    """
    kl_values = np.array(kl_values, dtype=float)
    dims = np.arange(len(kl_values))
    colors = np.where(kl_values >= threshold, "tab:green", "tab:gray")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(dims, kl_values, color=colors, edgecolor="k", alpha=0.9)
    ax.axhline(threshold, color="tab:red", ls="--", lw=1, alpha=0.8)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("KL (nats)")
    if title:
        ax.set_title(title)
    ax.set_xlim(-0.5, len(kl_values) - 0.5)
    ax.set_ylim(bottom=0)

    if fname is not None:
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        plt.savefig(fname, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, ax

    if save_sorted and sorted_fname is not None:
        _plot_sorted_kl(kl_values, threshold, sorted_fname, title=title)


def _plot_sorted_kl(
    kl_values: np.ndarray,
    threshold: float,
    fname: str,
    title: str | None = None,
):
    kl_sorted = np.sort(kl_values)[::-1]
    kl_cumsum = np.cumsum(kl_sorted)
    kl_frac = kl_cumsum / (kl_cumsum[-1] + 1e-8)

    fig, (ax_top, ax_frac) = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
    ax_top.plot(np.arange(len(kl_sorted)), kl_sorted, marker="o")
    ax_top.axhline(threshold, color="tab:red", ls="--", alpha=0.6, label="Active threshold")
    ax_top.set_ylabel("KL per dim (sorted)")
    ax_top.legend(frameon=False)

    ax_frac.plot(np.arange(len(kl_frac)), kl_frac, marker="o", color="tab:green")
    ax_frac.axhline(0.8, color="tab:gray", ls="--", alpha=0.6, label="80%")
    ax_frac.axhline(0.9, color="tab:gray", ls=":", alpha=0.6, label="90%")
    ax_frac.set_xlabel("Latent dim (sorted)")
    ax_frac.set_ylabel("Cumulative KL fraction")
    ax_frac.legend(frameon=False, loc="lower right")

    if title:
        fig.suptitle(title)
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    plt.savefig(fname, bbox_inches="tight")
    plt.close(fig)
