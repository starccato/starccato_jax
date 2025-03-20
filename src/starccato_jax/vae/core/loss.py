from dataclasses import dataclass
from typing import List, Union

import jax.numpy as jnp
import numpy as np

from .data_containers import Losses


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
    kl_divergence = -0.5 * jnp.mean(
        1 + logvar - jnp.square(mean) - jnp.exp(logvar)
    )

    # Total loss: reconstruction loss plus beta-scaled KL divergence.
    net_loss = reconstruction_loss + beta * kl_divergence

    return Losses(reconstruction_loss, kl_divergence, net_loss, beta)
