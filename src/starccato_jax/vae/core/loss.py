import jax.numpy as jnp

from .data_containers import Losses


def vae_loss(
    params,
    x,
    rng,
    model,
    beta: float,
    kl_free_bits: float = 0.0,
) -> Losses:
    """
    Computes the VAE loss.

    Parameters:
        params: The model parameters.
        x: The input batch.
        rng: Random number generator key.
        model: The VAE model.
        beta: Weighting factor for the KL divergence.
        kl_free_bits: Minimum KL divergence contribution (per batch). Set to 0
            to disable.

    Returns:
        Losses: A container holding the reconstruction loss, KL divergence,
                total loss, and beta.
    """
    # Forward pass: get reconstruction, mean, and log variance from the model.
    reconstructed, mean, logvar = model.apply({"params": params}, x, rng)

    # Reconstruction loss: mean squared error over all pixels (or features)
    reconstruction_loss = jnp.mean((x - reconstructed) ** 2)

    # BATCH AVERAGE KL DIVERGENCE
    # which would compute the mean KL divergence over the batch.
    kl_divergence = -0.5 * jnp.mean(
        1 + logvar - jnp.square(mean) - jnp.exp(logvar)
    )
    # Apply free-bits as a floor; when kl_free_bits=0 this is a no-op.
    kl_divergence = jnp.maximum(kl_divergence, kl_free_bits)

    # Total loss: reconstruction loss plus beta-scaled KL divergence.
    net_loss = reconstruction_loss + beta * kl_divergence

    return Losses(reconstruction_loss, kl_divergence, net_loss, beta)
