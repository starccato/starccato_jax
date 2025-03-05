from dataclasses import dataclass
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random
from jax.random import PRNGKey

__all__ = ["VAE", "ModelData", "generate", "reconstruct", "encode"]


def _reparameterize(
    rng: PRNGKey, mean: jnp.ndarray, logvar: jnp.ndarray
) -> jnp.ndarray:
    std = jnp.exp(0.5 * logvar)
    eps = jax.random.normal(rng, std.shape)
    return mean + eps * std


class Encoder(nn.Module):
    """Convolutional Encoder for VAE with normalization and dropout."""

    latents: int
    dropout_rate: float = 0.1
    train: bool = True  # controls BatchNorm and Dropout

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Ensure x has a channel dimension: (batch, 256) -> (batch, 256, 1)
        if x.ndim == 2:
            x = x[..., None]

        # Expecting x shape: (batch, 256, 1)
        x = nn.Conv(
            features=32,
            kernel_size=(3,),
            strides=(2,),
            padding="SAME",
            name="conv1",
        )(x)
        x = nn.BatchNorm(use_running_average=not self.train, name="bn1")(x)
        x = nn.leaky_relu(x, negative_slope=0.1)
        x = nn.Dropout(
            rate=self.dropout_rate,
            deterministic=not self.train,
            name="drop1",
        )(x)

        x = nn.Conv(
            features=64,
            kernel_size=(3,),
            strides=(2,),
            padding="SAME",
            name="conv2",
        )(x)
        x = nn.BatchNorm(use_running_average=not self.train, name="bn2")(x)
        x = nn.leaky_relu(x, negative_slope=0.1)
        x = nn.Dropout(
            rate=self.dropout_rate, deterministic=not self.train, name="drop2"
        )(x)

        # Flatten the convolutional features.
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(128, name="fc")(x)
        x = nn.BatchNorm(use_running_average=not self.train, name="bn_fc")(x)
        x = nn.leaky_relu(x, negative_slope=0.1)

        # Generate latent parameters.
        mean_x = nn.Dense(self.latents, name="fc_mean")(x)
        logvar_x = nn.Dense(self.latents, name="fc_logvar")(x)
        return mean_x, logvar_x


class Decoder(nn.Module):
    """Convolutional Decoder for VAE with normalization and dropout."""

    output_shape: Tuple[int, int]  # e.g. (256, 1)
    dropout_rate: float = 0.1
    train: bool = True

    @nn.compact
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        x = nn.Dense(128, name="fc")(z)
        x = nn.BatchNorm(use_running_average=not self.train, name="bn_fc")(x)
        x = nn.leaky_relu(x, negative_slope=0.1)
        x = nn.Dropout(
            rate=self.dropout_rate,
            deterministic=not self.train,
            name="drop_fc",
        )(x)

        # For the reshape, assume the flattened conv feature map size from encoder is 64x64.
        flat_dim = 64 * 64
        x = nn.Dense(flat_dim, name="fc_reshape")(x)
        x = nn.BatchNorm(
            use_running_average=not self.train, name="bn_reshape"
        )(x)
        x = nn.leaky_relu(x, negative_slope=0.1)
        x = x.reshape((x.shape[0], 64, 64))

        x = nn.ConvTranspose(
            features=32,
            kernel_size=(3,),
            strides=(2,),
            padding="SAME",
            name="deconv1",
        )(x)
        x = nn.BatchNorm(
            use_running_average=not self.train, name="bn_deconv1"
        )(x)
        x = nn.leaky_relu(x, negative_slope=0.1)
        x = nn.Dropout(
            rate=self.dropout_rate,
            deterministic=not self.train,
            name="drop_deconv1",
        )(x)

        x = nn.ConvTranspose(
            features=self.output_shape[-1],
            kernel_size=(3,),
            strides=(2,),
            padding="SAME",
            name="deconv2",
        )(x)
        # For the final layer, you might not apply normalization or activation depending on your data range.
        return x


class VAE(nn.Module):
    """Full Convolutional VAE model with normalization, dropout and LeakyReLU activations."""

    latents: int
    output_shape: Tuple[int, int] = (256, 1)  # e.g. (256, 1)
    dropout_rate: float = 0.1
    train: bool = True

    def setup(self):
        self.encoder = Encoder(
            self.latents, dropout_rate=self.dropout_rate, train=self.train
        )
        self.decoder = Decoder(
            self.output_shape, dropout_rate=self.dropout_rate, train=self.train
        )

    def __call__(self, x: jnp.ndarray, rng: PRNGKey):
        mean, logvar = self.encoder(x)
        z = _reparameterize(rng, mean, logvar)
        recon_x = self.decoder(z)
        return recon_x, mean, logvar

    def generate(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.decoder(z)

    def encode(self, x: jnp.ndarray, rng: PRNGKey) -> jnp.ndarray:
        mean, logvar = self.encoder(x)
        z = _reparameterize(rng, mean, logvar)
        return z


def _reparameterize(
    rng: PRNGKey, mean: jnp.ndarray, logvar: jnp.ndarray
) -> jnp.ndarray:
    """
    Reparameterization trick for VAE

    This takes in the mean and logvar of the latent distribution and returns a sample
    from the distribution.
    """
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std


@dataclass
class ModelData:
    params: Dict
    latent_dim: int
    batch_stats: Dict


def generate(
    model_data: ModelData,
    z: jnp.ndarray = None,
    rng: PRNGKey = None,
    n: int = 1,
) -> jnp.ndarray:
    rng = rng if rng is not None else PRNGKey(0)
    z = (
        jax.random.normal(rng, shape=(n, model_data.latent_dim))
        if z is None
        else z
    )
    model = VAE(model_data.latent_dim, train=False)
    return model.apply(
        {"params": model_data.params, "batch_stats": model_data.batch_stats},
        z,
        method=model.generate,
    )


def reconstruct(
    x: jnp.ndarray, model_data: ModelData, rng: PRNGKey = None, n_reps: int = 1
) -> jnp.ndarray:
    recons = call_vae(x, model_data, rng=rng, n_reps=n_reps)[0]
    # remove channel dimension if it exists (batch, 256, 1) -> (batch, 256)
    return recons[..., 0] if recons.ndim == 3 else recons


def call_vae(
    x: jnp.ndarray,
    model_data: ModelData,
    model=None,
    rng: PRNGKey = None,
    n_reps: int = 1,
) -> jnp.ndarray:
    rng = rng if rng is not None else PRNGKey(0)
    # duplicate the same x for n_reps times so we have a batch of size n_reps
    x = x if n_reps == 1 else jnp.repeat(x[None, :], n_reps, axis=0)
    if model is None:
        model = VAE(model_data.latent_dim, train=False)
    return model.apply(
        {"params": model_data.params, "batch_stats": model_data.batch_stats},
        x,
        rng,
    )


def encode(
    x: jnp.ndarray,
    model_data: ModelData,
    rng: PRNGKey = None,
) -> jnp.ndarray:
    """
    Encodes the input `x` into the latent vector `z` using the trained encoder.

    Args:
        x (jnp.ndarray): Input data.
        model_data (ModelData): Model parameters and latent dimension.

    Returns:
        jnp.ndarray: Encoded latent vector `z`.
    """
    rng = rng if rng is not None else PRNGKey(0)
    model = VAE(model_data.latent_dim, train=False)
    z = model.apply(
        {"params": model_data.params, "batch_stats": model_data.batch_stats},
        x,
        rng,
        method=model.encode,
    )
    return z
