from typing import Tuple

import jax.numpy as jnp
from flax import linen as nn
from flax.core.frozen_dict import FrozenDict
from jax import random
from jax.random import PRNGKey

from .data_containers import ModelData

__all__ = [
    "VAE",
    "ModelData",
    "generate",
    "reconstruct",
    "encode",
    "encode_mean",
    "normalize_waveform",
]


def normalize_waveform(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    """Return a zero-mean, unit-RMS waveform along the final axis."""
    centered = x - jnp.mean(x, axis=-1, keepdims=True)
    rms = jnp.sqrt(jnp.mean(jnp.square(centered), axis=-1, keepdims=True))
    return centered / jnp.maximum(rms, eps)


class Encoder(nn.Module):
    """VAE Encoder."""

    latents: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> Tuple[float, float]:
        x = nn.Dense(256, name="fc1")(x)
        x = nn.leaky_relu(x, negative_slope=0.01)
        x = nn.Dense(128, name="fc2")(x)
        x = nn.leaky_relu(x, negative_slope=0.01)
        x = nn.Dense(64, name="fc3")(x)
        x = nn.leaky_relu(x, negative_slope=0.01)
        mean_x = nn.Dense(self.latents, name="fc3_mean")(x)
        logvar_x = nn.Dense(self.latents, name="fc3_logvar")(x)
        return mean_x, logvar_x


def _has_legacy_decoder(params: dict) -> bool:
    """Detect older decoder params (fc4 present, no layer norms)."""
    if isinstance(params, FrozenDict):
        params = params.unfreeze()
    dec = params.get("decoder", {})
    return "fc4" in dec


class Decoder(nn.Module):
    """VAE Decoder."""

    output_dim: int

    @nn.compact
    def __call__(
        self, z: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        z = nn.Dense(64, name="fc1")(z)
        # use_scale/use_bias=False to keep compatibility with older checkpoints
        z = nn.LayerNorm(name="ln1", use_scale=False, use_bias=False)(z)
        z = nn.leaky_relu(z, negative_slope=0.01)
        z = nn.Dropout(rate=0.1, name="drop1")(z, deterministic=deterministic)

        z = nn.Dense(128, name="fc2")(z)
        z = nn.LayerNorm(name="ln2", use_scale=False, use_bias=False)(z)
        z = nn.leaky_relu(z, negative_slope=0.01)
        z = nn.Dropout(rate=0.1, name="drop2")(z, deterministic=deterministic)

        z = nn.Dense(self.output_dim, name="fc3")(z)
        return z


class LegacyDecoder(nn.Module):
    """Legacy decoder (pre-LayerNorm/Dropout) for backward compatibility."""

    output_dim: int

    @nn.compact
    def __call__(
        self, z: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        z = nn.Dense(64, name="fc1")(z)
        z = nn.leaky_relu(z, negative_slope=0.01)
        z = nn.Dense(128, name="fc2")(z)
        z = nn.leaky_relu(z, negative_slope=0.01)
        z = nn.Dense(256, name="fc3")(z)
        z = nn.leaky_relu(z, negative_slope=0.01)
        z = nn.Dense(self.output_dim, name="fc4")(z)
        return z


class VAE(nn.Module):
    """Full VAE model."""

    latents: int = 32
    data_dim: int | None = None
    use_legacy_decoder: bool = False
    normalize_decoder_output: bool = True

    def setup(self):
        if self.data_dim is None:
            raise ValueError("VAE requires data_dim to be set.")
        self.encoder = Encoder(self.latents)
        self.decoder = (
            LegacyDecoder(self.data_dim)
            if self.use_legacy_decoder
            else Decoder(self.data_dim)
        )

    def __call__(
        self, x: jnp.ndarray, rng: PRNGKey, deterministic: bool = True
    ):
        if x.shape[-1] != self.data_dim:
            raise ValueError(
                f"Input dimension {x.shape[-1]} does not match configured "
                f"data_dim={self.data_dim}."
            )
        mean, logvar = self.encoder(x)
        z = _reparameterize(rng, mean, logvar)
        recon_x = self.decode(z, deterministic=deterministic)
        return recon_x, mean, logvar

    def reconstruct_from_mean(self, x: jnp.ndarray):
        """Reconstruct ``x`` from the encoder mean without sampling.

        This path is intended for validation and checkpoint selection.  The
        ordinary forward path remains stochastic so training and public
        posterior-reconstruction APIs retain their existing behaviour.
        """
        if x.shape[-1] != self.data_dim:
            raise ValueError(
                f"Input dimension {x.shape[-1]} does not match configured "
                f"data_dim={self.data_dim}."
            )
        mean, logvar = self.encoder(x)
        recon_x = self.decode(mean, deterministic=True)
        return recon_x, mean, logvar

    def decode(
        self, z: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        waveform = self.decoder(z, deterministic=deterministic)
        if self.normalize_decoder_output:
            waveform = normalize_waveform(waveform)
        return waveform

    def generate(self, z: jnp.ndarray) -> jnp.ndarray:
        return self.decode(z, deterministic=True)

    def encode(self, x: jnp.ndarray, rng: PRNGKey) -> jnp.ndarray:
        mean, logvar = self.encoder(x)
        z = _reparameterize(rng, mean, logvar)
        return z

    def encode_mean(self, x: jnp.ndarray) -> jnp.ndarray:
        """Return the deterministic approximate-posterior mean."""
        mean, _ = self.encoder(x)
        return mean


def _reparameterize(
    rng: PRNGKey, mean: jnp.ndarray, logvar: jnp.ndarray
) -> jnp.ndarray:
    """
    Reparameterization trick for VAE

    This takes the mean and log variance of the latent distribution and returns
    a sample from the distribution.
    """
    std = jnp.exp(0.5 * logvar)
    eps = random.normal(rng, logvar.shape)
    return mean + eps * std


def generate(
    model_data: ModelData,
    z: jnp.ndarray,
    model: VAE = None,
) -> jnp.ndarray:
    if model is None:
        model = VAE(
            model_data.latent_dim,
            data_dim=model_data.data_dim,
            use_legacy_decoder=_has_legacy_decoder(model_data.params),
            normalize_decoder_output=model_data.normalize_decoder_output,
        )

    return model.apply({"params": model_data.params}, z, method=model.generate)


def encode(
    x: jnp.ndarray,
    model_data: ModelData,
    rng: PRNGKey = None,
    model: VAE = None,
) -> jnp.ndarray:
    """
    Encodes the input `x` into the latent vector `z` using the trained encoder.

    Args:
        x (jnp.ndarray): Input data.
        model_data (ModelData): Model parameters and latent dimension.
        rng (PRNGKey, optional): Random number generator. Defaults to None.
        model (VAE, optional): VAE model. Defaults to None.

    Returns:
        jnp.ndarray: Encoded latent vector `z`.
    """
    rng = rng if rng is not None else PRNGKey(0)
    if model is None:
        model = VAE(
            model_data.latent_dim,
            data_dim=model_data.data_dim,
            use_legacy_decoder=_has_legacy_decoder(model_data.params),
            normalize_decoder_output=model_data.normalize_decoder_output,
        )

    z = model.apply({"params": model_data.params}, x, rng, method=model.encode)
    return z


def encode_mean(
    x: jnp.ndarray,
    model_data: ModelData,
    model: VAE = None,
) -> jnp.ndarray:
    """Encode ``x`` using the approximate-posterior mean (no RNG required)."""
    if model is None:
        model = VAE(
            model_data.latent_dim,
            data_dim=model_data.data_dim,
            use_legacy_decoder=_has_legacy_decoder(model_data.params),
            normalize_decoder_output=model_data.normalize_decoder_output,
        )
    return model.apply(
        {"params": model_data.params}, x, method=model.encode_mean
    )


def reconstruct(
    x: jnp.ndarray,
    model_data: ModelData,
    rng: PRNGKey = None,
    n_reps: int = 1,
    model: VAE = None,
) -> jnp.ndarray:
    rng = rng if rng is not None else PRNGKey(0)
    # duplicate the same x for n_reps times so we have a batch of size n_reps
    if n_reps > 1:
        if x.ndim != 1:
            x = x[0]
        x = jnp.repeat(x[None, :], n_reps, axis=0)

    if model is None:
        model = VAE(
            model_data.latent_dim,
            data_dim=model_data.data_dim,
            use_legacy_decoder=_has_legacy_decoder(model_data.params),
            normalize_decoder_output=model_data.normalize_decoder_output,
        )

    return model.apply({"params": model_data.params}, x, rng)[0]
