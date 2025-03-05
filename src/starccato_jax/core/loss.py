from dataclasses import dataclass
from typing import List, Union

import jax.numpy as jnp
import numpy as np

from .model import call_vae


@dataclass
class Losses:
    reconstruction_loss: Union[float, List[float]]
    kl_divergence: Union[float, List[float]]
    loss: Union[float, List[float]]
    beta: Union[float, List[float]]


@dataclass
class TrainValMetrics:
    train_metrics: Losses
    val_metrics: Losses

    def __str__(self) -> str:
        tl, vl = self.train_metrics.loss, self.val_metrics.loss
        return f"Train Loss: {tl:.3e}, Val Loss: {vl:.3e}"


def cyclical_annealing_beta(
    n_epoch, start=0.0, stop=1.0, n_cycle=4, ratio=0.5
):
    """
    Computes a cyclical annealing schedule for the beta parameter in a VAE.

    Parameters:
        start (float): Initial beta value (e.g., 0.0).
        stop (float): Maximum beta value (e.g., 1.0).
        n_epoch (int): Total number of epochs.
        n_cycle (int): Number of cycles for annealing. Set to 0 for no annealing.
        ratio (float): Ratio of the increasing phase within each cycle.

    Returns:
        np.ndarray: A list of beta values for each epoch.
    """
    beta_schedule = np.ones(n_epoch) * stop  # Default to max beta

    if n_cycle > 0:
        period = n_epoch / n_cycle  # Length of each cycle
        step = (stop - start) / (period * ratio)

        for c in range(n_cycle):
            v = start
            for i in range(int(period * ratio)):  # Annealing phase
                idx = int(i + c * period)
                if idx < n_epoch:
                    beta_schedule[idx] = 1.0 / (
                        1.0 + np.exp(-(v * 12.0 - 6.0))
                    )
                v += step

    return beta_schedule


def aggregate_metrics(metrics_list: List[TrainValMetrics]) -> TrainValMetrics:
    """Convert a list of TrainValMetrics (one per epoch) into a single object with lists of values."""
    return TrainValMetrics(
        train_metrics=Losses(
            reconstruction_loss=[
                m.train_metrics.reconstruction_loss for m in metrics_list
            ],
            kl_divergence=[
                m.train_metrics.kl_divergence for m in metrics_list
            ],
            loss=[m.train_metrics.loss for m in metrics_list],
            beta=[m.train_metrics.beta for m in metrics_list],
        ),
        val_metrics=Losses(
            reconstruction_loss=[
                m.val_metrics.reconstruction_loss for m in metrics_list
            ],
            kl_divergence=[m.val_metrics.kl_divergence for m in metrics_list],
            loss=[m.val_metrics.loss for m in metrics_list],
            beta=[m.train_metrics.beta for m in metrics_list],
        ),
    )


def vae_loss(recon_x, x, mean, logvar, beta):
    # Reconstruction loss (MSE)
    reconstruction_loss = jnp.mean((x - recon_x[..., 0]) ** 2)
    # KL divergence loss
    kl_divergence = -0.5 * jnp.mean(
        1 + logvar - jnp.square(mean) - jnp.exp(logvar)
    )
    net_loss = reconstruction_loss + beta * kl_divergence
    return Losses(reconstruction_loss, kl_divergence, net_loss, beta)


def compute_metrics(model_data, x, rng, validation_x, beta) -> TrainValMetrics:
    reconstructed_x, mean, logvar = call_vae(x, model_data, rng)
    reconstructed_xval, mean_val, logvar_val = call_vae(
        validation_x, model_data, rng
    )
    return TrainValMetrics(
        vae_loss(reconstructed_x, x, mean, logvar, beta),
        vae_loss(reconstructed_xval, validation_x, mean_val, logvar_val, beta),
    )
