import warnings

import matplotlib.pyplot as plt
import numpy as np

from ..credible_intervals import coverage_probability

MODEL_COL = "tab:orange"
OBS_COL = "tab:gray"
TRUE_COL = "black"

FS = 4096
ND = 256
TIME = np.arange(0, ND) / FS
TIME = TIME - TIME[58]  # index of the peak


def add_quantiles(
    ax: plt.Axes,
    y_ci: np.ndarray,
    label: str = None,
    color: str = MODEL_COL,
    alpha: float = 0.5,
    y_obs: np.ndarray = None,
    x=TIME.copy(),
):
    # assert that the y_ci are differnt values (no bug in reconstruction)
    if np.allclose(y_ci[0], y_ci[1]):
        warnings.warn(
            "Quantiles are the same, no uncertainty in reconstruction... SUSPICIOUS"
        )

    _, xlen = y_ci.shape

    if y_obs is not None:
        ax.plot(x, y_obs, color=TRUE_COL, lw=2, zorder=-1, label="Observed")
        # set ylim _slightly_ above and below the y_obs
        ax.set_ylim(np.min(y_obs) - 0.1, np.max(y_obs) + 0.1)
        coverage = coverage_probability(y_ci, y_obs)
        label = f"{label} ({coverage:.0%})"

    ax.fill_between(
        x, y_ci[0], y_ci[2], color=color, alpha=alpha, label=label, lw=0
    )
    ax.plot(x, y_ci[1], color=color, lw=1, ls="--")
