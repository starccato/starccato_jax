import jax
import jax.numpy as jnp
from flax import linen as nn
from jax import random

__all__ = ['VAE', 'generate']



class Encoder(nn.Module):
  """VAE Encoder."""

  latents: int

  @nn.compact
  def __call__(self, x):
    x = nn.Dense(512, name='fc1')(x)
    x = nn.relu(x)
    mean_x = nn.Dense(self.latents, name='fc2_mean')(x)
    logvar_x = nn.Dense(self.latents, name='fc2_logvar')(x)
    return mean_x, logvar_x


class Decoder(nn.Module):
  """VAE Decoder."""

  @nn.compact
  def __call__(self, z):
    z = nn.Dense(512, name='fc1')(z)
    z = nn.relu(z)
    z = nn.Dense(256, name='fc2')(z)
    return z


class VAE(nn.Module):
  """Full VAE model."""

  latents: int = 20

  def setup(self):
    self.encoder = Encoder(self.latents)
    self.decoder = Decoder()

  def __call__(self, x, z_rng):
    mean, logvar = self.encoder(x)
    z = _reparameterize(z_rng, mean, logvar)
    recon_x = self.decoder(z)
    return recon_x, mean, logvar

  def generate(self, z):
    return self.decoder(z)



def _reparameterize(rng, mean, logvar):
  std = jnp.exp(0.5 * logvar)
  eps = random.normal(rng, logvar.shape)
  return mean + eps * std

def generate(state_params,  latent_dim:int=None, z=None, rng_seed=42, rng=None):
  if rng is None:
    rng = jax.random.PRNGKey(rng_seed)
  if z is None:
    # Generate a random latent vector (batch size 1).
    z = jax.random.normal(rng, shape=(1, latent_dim))
  p = {'params': state_params['params']}
  return VAE(latent_dim).apply(p, z, method=VAE.generate)


def reconstruct(x, state_params, latent_dim:int, rng_seed:int=42, rng:jax.random.PRNGKey=None, n_reps=1):
  if rng is None:
    rng = jax.random.PRNGKey(rng_seed)
  p = {'params': state_params['params']}

  if n_reps > 1:
    # duplicate the same x for n_reps times so we have a batch of size n_reps
    x = jnp.repeat(x[None, :], n_reps, axis=0)

  reconstructed, _, _ = VAE(latent_dim).apply(p, x, rng)
  return reconstructed