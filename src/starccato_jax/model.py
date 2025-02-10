from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random
from jax.random import PRNGKey

__all__ = ["VAE", "ModelData", "generate", "reconstruct"]


class Encoder(nn.Module):
    """VAE Encoder."""

    latents: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[float, float]:
        x = nn.Dense(1024, name="fc1")(x)
        x = nn.relu(x)
        x = nn.Dense(64, name="fc2")(x)
        mean_x = nn.Dense(self.latents, name="fc2_mean")(x)
        logvar_x = nn.Dense(self.latents, name="fc2_logvar")(x)
        return mean_x, logvar_x


class Decoder(nn.Module):
    """VAE Decoder."""

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        z = nn.Dense(64, name="fc1")(z)
        z = nn.relu(z)
        z = nn.Dense(256, name="fc2")(z)
        return z


class VAE(nn.Module):
    """Full VAE model."""

    latents: int = 20

    def setup(self):
        self.encoder = Encoder(self.latents)
        self.decoder = Decoder()

    def __call__(self, x: jnp.ndarray, z_rng: jnp.ndarray):
        mean, logvar = self.encoder(x)
        z = _reparameterize(z_rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

    def generate(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.decoder(z)


def _reparameterize(
    rng: PRNGKey, mean: jnp.ndarray, logvar: jnp.ndarray
) -> jnp.ndarray:
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std


@dataclass
class ModelData:
    params: Dict
    latent_dim: int


def generate(
    model_data: ModelData, z: jnp.ndarray = None, rng: PRNGKey = None
) -> jnp.ndarray:
    rng = rng if rng is not None else PRNGKey(0)
    z = (
        jax.random.normal(rng, shape=(1, model_data.latent_dim))
        if z is None
        else z
    )
    model = VAE(model_data.latent_dim)
    return model.apply({"params": model_data.params}, z, method=model.generate)


def reconstruct(
    x: jnp.ndarray, model_data: ModelData, rng: PRNGKey = None, n_reps: int = 1
) -> jnp.ndarray:
    rng = rng if rng is not None else PRNGKey(0)
    # duplicate the same x for n_reps times so we have a batch of size n_reps
    x = x if n_reps == 1 else jnp.repeat(x[None, :], n_reps, axis=0)
    model = VAE(model_data.latent_dim)
    return model.apply({"params": model_data.params}, x, rng)[0]
