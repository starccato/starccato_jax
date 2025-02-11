from dataclasses import dataclass
from typing import List

import jax.numpy as jnp
import numpy as np


@dataclass
class Losses:
    reconstruction_loss: float
    kl_divergence: float
    loss: float
    beta: float


@dataclass
class TrainValMetrics:
    train_metrics: Losses
    val_metrics: Losses

    def __str__(self) -> str:
        tl, vl = self.train_metrics.loss, self.val_metrics.loss
        return f"Train Loss: {tl:.3e}, Val Loss: {vl:.3e}"


# def compute_beta(step: int, cycle_length: int=0) -> float:
#     """Cyclical linear annealing from 0 to 1 over `cycle_length` steps.
#
#     - Setting `cycle_length` to the number of steps in an epoch will result in a linear annealing schedule within each epoch.
#     - Setting `cycle_length` to the number of steps in a training loop will result in a linear annealing schedule across the entire training loop.
#     - Setting `cycle_length` to a small number (0) will result in a constant value of 1 (no annealing).
#     """
#     cycle_step = step % cycle_length
#     return min(cycle_step / cycle_length, 1.0)


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


def vae_loss(params, x, rng, model, beta: float) -> Losses:
    """
    Computes the VAE loss.

    Parameters:
        params: The model parameters.
        x: The input batch.
        rng: Random number generator key.
        model: The VAE model.
        beta: Weighting factor for the KL divergence.

    Returns:
        Losses: A container holding the reconstruction loss, KL divergence,
                total loss, and beta.
    """
    # Forward pass: get reconstruction, mean, and log variance from the model.
    reconstructed, mean, logvar = model.apply({"params": params}, x, rng)

    # Reconstruction loss: mean squared error over all pixels (or features)
    reconstruction_loss = jnp.mean((x - reconstructed) ** 2)

    # Compute KL divergence

    # PER SAMPLE KL DIVERGENCE
    # per sample by summing over the latent dimensions.
    # Then average over the batch.
    # kl_per_sample = -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar), axis=1)
    # kl_divergence = jnp.mean(kl_per_sample)

    # BATCH AVERAGE KL DIVERGENCE
    # which would compute the mean KL divergence over the batch.
    kl_divergence = -0.5 * jnp.mean(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

    # Total loss: reconstruction loss plus beta-scaled KL divergence.
    net_loss = reconstruction_loss + beta * kl_divergence

    return Losses(reconstruction_loss, kl_divergence, net_loss, beta)

def compute_metrics(
    state, x, rng, model, validation_x, beta
) -> TrainValMetrics:
    return TrainValMetrics(
        vae_loss(state.params, x, rng, model, beta),
        vae_loss(state.params, validation_x, rng, model, beta),
    )
