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
):
    """
    Plot KL divergence per latent dimension.

    Args:
        kl_values: KL values per dimension (array-like).
        threshold: Threshold above which a latent is considered "active".
        fname: Optional path to save the plot.
        title: Optional title.
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
