import jax
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from scipy.stats import alpha

from .model import reconstruct
from .io import ModelData
from .loss import TrainValMetrics, Losses, aggregate_metrics
from typing import List


def plot_training_metrics(training_metrics: List[TrainValMetrics], fname: str = None):
    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    n = len(training_metrics)
    metrics = aggregate_metrics(training_metrics)
    _plot_loss(ax, metrics.train_metrics, 'Train', 'tab:blue')
    _plot_loss(ax, metrics.val_metrics, 'Val', 'tab:orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.plot([], [], label='Reconstruction', color='tab:gray', ls=':', alpha=0.5)
    plt.plot([], [], label='KL Divergence', color='tab:gray', ls='--', alpha=0.5)
    plt.legend(frameon=False, loc='upper right')

    ax_metrics = plt.twinx()
    ax_metrics.plot(metrics.train_metrics.beta, label='Beta', color='k', alpha=0.5, lw=2)
    ax.set_ylim(bottom=0)
    ax_metrics.set_ylim(0,1)
    ax.set_xlim(0, n)
    ax_metrics.set_xlim(0, n)
    ax_metrics.set_ylabel('Beta')
    if fname is not None:
        plt.savefig(fname)


def _plot_loss(ax: plt.Axes, losses: Losses, label: str, color: str):
    ax.plot(losses.loss, label=label, color=color, lw=2)
    ax.plot(losses.reconstruction_loss, color=color, ls=':', alpha=0.5)
    ax.plot(losses.kl_divergence, color=color, ls='--', alpha=0.5)


def plot_reconstructions(model_data: ModelData, val_data: np.ndarray, nrows: int = 3, fname: str = None):
    ncols = nrows
    nsamples = nrows * ncols
    rng = jax.random.PRNGKey(0)
    idx = np.random.choice(val_data.shape[0], nsamples, replace=False)
    orig = val_data[idx]

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.5 * ncols, 2.5 * nrows))
    axes = axes.flatten()
    for i in range(nsamples):
        recon = reconstruct(orig[i], model_data, rng, n_reps=100)
        qtls = jnp.quantile(recon, jnp.array([0.025, 0.5, 0.975]), axis=0)
        _add_quantiles(axes[i], qtls, 'Reconstruction', 'tab:gray', y_obs=orig[i])
        axes[i].set_axis_off()
    axes[-1].legend(frameon=False, loc='lower right')
    plt.subplots_adjust(hspace=0, wspace=0)
    if fname is not None:
        plt.savefig(fname)


def _add_quantiles(
        ax: plt.Axes, y_ci: np.ndarray, label: str, color: str, alpha: float = 0.3,
        y_obs: np.ndarray = None):
    _, xlen = y_ci.shape
    x = np.arange(xlen)
    ax.fill_between(x, y_ci[0], y_ci[2], color=color, alpha=alpha, label=label, lw=0)
    ax.plot(y_ci[1], color=color, lw=1, ls='--')
    if y_obs is not None:
        ax.plot(y_obs, color='black', lw=2, zorder=-1, label='Observed')
