import jax
import jax.numpy as jnp

from .data_containers import Losses


def vae_loss(
    params,
    x,
    rng,
    model,
    beta: float,
    kl_free_bits: float = 0.0,
    use_capacity: bool = False,
    capacity: float = 0.0,
    beta_capacity: float = 1.0,
    deterministic: bool = True,
) -> Losses:
    """
    Computes the VAE loss.

    Parameters:
        params: The model parameters.
        x: The input batch.
        rng: Random number generator key.
        model: The VAE model.
        beta: Weighting factor for the KL divergence.
        kl_free_bits: Minimum total KL contribution in nats. Set to 0 to
            disable.
        use_capacity: If True, use capacity-controlled KL objective.
        capacity: Upper total-KL budget (nats) for capacity objective.
        beta_capacity: Weight for capacity objective.

    Returns:
        Losses: A container holding the reconstruction loss, KL divergence,
                total loss, and beta.
    """
    # Validation/checkpoint metrics use the encoder mean.  Disabling dropout
    # alone is not deterministic because the ordinary VAE forward pass still
    # samples from q(z | x).
    if deterministic:
        reconstructed, mean, logvar = model.apply(
            {"params": params}, x, method=model.reconstruct_from_mean
        )
    else:
        reconstructed, mean, logvar = model.apply(
            {"params": params},
            x,
            rng,
            False,
            rngs={"dropout": rng},
        )

    # Reconstruction loss: mean squared error over all pixels (or features)
    reconstruction_loss = jnp.mean((x - reconstructed) ** 2)

    # Mean over the batch of the *total* latent KL per example. Summing the
    # latent dimensions is essential: capacity and free-bit values are in
    # total nats, independent of the chosen latent dimensionality.
    kl_per_example = -0.5 * jnp.sum(
        1 + logvar - jnp.square(mean) - jnp.exp(logvar), axis=-1
    )
    kl_divergence = jnp.mean(kl_per_example)
    # Apply free-bits as a floor; when kl_free_bits=0 this is a no-op.
    kl_divergence = jnp.maximum(kl_divergence, kl_free_bits)

    # Total loss: reconstruction loss plus beta-scaled KL divergence.
    if use_capacity:
        # Capacity is an upper information budget, not an equality target.
        cap = jnp.asarray(capacity)
        # Hinge version penalizes only KL above capacity.
        net_loss = reconstruction_loss + beta_capacity * jnp.maximum(
            kl_divergence - jax.lax.stop_gradient(cap), 0.0
        )
    else:
        net_loss = reconstruction_loss + beta * kl_divergence

    return Losses(
        reconstruction_loss=reconstruction_loss,
        kl_divergence=kl_divergence,
        loss=net_loss,
        beta=beta,
        capacity=capacity,
    )
