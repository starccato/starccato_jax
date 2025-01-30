# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Variational Autoencoder example on binarized MNIST dataset.

See "Auto-encoding variational Bayes" (Kingma & Welling, 2014) [0].

[0]https://arxiv.org/abs/1312.6114
"""

from collections.abc import Iterator, Sequence
import dataclasses
from typing import NamedTuple

from absl import flags
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax



@dataclasses.dataclass
class Config:
  batch_size: int = 128
  learning_rate: float = 1e-3
  training_steps: int = 5000
  eval_every: int = 100
  seed: int = 0


class Batch(NamedTuple):
  x: jax.Array


@dataclasses.dataclass
class Encoder(hk.Module):
  """Encoder model."""

  latent_size: int
  hidden_size: int = 128

  def __call__(self, x: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Encodes an INUPUT as an isotropic Guassian latent code."""
    x = hk.Flatten()(x)
    x = hk.Linear(self.hidden_size)(x)
    x = jax.nn.relu(x)

    mean = hk.Linear(self.latent_size)(x)
    log_stddev = hk.Linear(self.latent_size)(x)
    stddev = jnp.exp(log_stddev)

    return mean, stddev


@dataclasses.dataclass
class Decoder(hk.Module):
  """Decoder model."""

  output_shape: Sequence[int]
  hidden_size: int = 128

  def __call__(self, z: jax.Array) -> jax.Array:
    z = hk.Linear(self.hidden_size)(z)
    z = jax.nn.relu(z)
    x_hat = hk.Linear(output_size=self.output_shape[0])(z)
    return x_hat


class VAEOutput(NamedTuple):
  input: jax.Array
  mean: jax.Array
  variance: jax.Array
  output: jax.Array


@dataclasses.dataclass
class VariationalAutoEncoder(hk.Module):
  """Main VAE model class."""

  encoder: Encoder
  decoder: Decoder

  def __call__(self, x: jax.Array) -> VAEOutput:
    """Forward pass of the variational autoencoder."""
    x = x.astype(jnp.float32)
    mean, stddev = self.encoder(x)
    z = mean + stddev * jax.random.normal(hk.next_rng_key(), mean.shape)
    decoded_x = self.decoder(z)
    return VAEOutput(x, mean, jnp.square(stddev), decoded_x)


class TrainingState(NamedTuple):
  params: hk.Params
  opt_state: optax.OptState
  rng_key: jax.Array

def compute_mse_loss(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  """Computes mean squared error loss."""
  if x.shape != y.shape:
    raise ValueError("Loss requires x and y to have the same shape.")
  return jnp.mean(jnp.square(x - y))


def run_training(train_dataset: Iterator[Batch], eval_dataset: Iterator[Batch], config: Config, outdir:str):



  @hk.transform
  def model(x):
    vae = VariationalAutoEncoder(
        encoder=Encoder(latent_size=10),
        decoder=Decoder(output_shape=x.shape[1:]),
    )
    return vae(x)

  @jax.jit
  def loss_fn(params, rng_key, batch: Batch) -> jax.Array:
    """ELBO loss: E_p[log(x)] - KL(d||q), where p ~ Be(0.5) and q ~ N(0,1)."""

    # Run the model on the inputs.
    _, mean, var, x_hat = model.apply(params, rng_key, batch)

    # Mean squared error for reproduction loss.
    log_likelihood = -compute_mse_loss(batch.x, x_hat)

    # KL divergence between Gaussians N(mean, std) and N(0, 1).
    kl = 0.5 * jnp.sum(-jnp.log(var) - 1. + var + jnp.square(mean), axis=-1)

    # Loss is the negative evidence lower-bound.
    return -jnp.mean(log_likelihood - kl)

  optimizer = optax.adam(config.learning_rate)

  @jax.jit
  def update(state: TrainingState, batch: Batch) -> TrainingState:
    """Performs a single SGD step."""
    rng_key, next_rng_key = jax.random.split(state.rng_key)
    gradients = jax.grad(loss_fn)(state.params, rng_key, batch)
    updates, new_opt_state = optimizer.update(gradients, state.opt_state)
    new_params = optax.apply_updates(state.params, updates)
    return TrainingState(new_params, new_opt_state, next_rng_key)


  # Initialise the training state.
  initial_rng_key = jax.random.PRNGKey(config.seed)
  initial_params = model.init(initial_rng_key, next(train_dataset))
  initial_opt_state = optimizer.init(initial_params)
  state = TrainingState(initial_params, initial_opt_state, initial_rng_key)


  metrics = []
  recon_data = next(eval_dataset)

  # Run training and evaluation.
  for step in range(config.training_steps):
    train_batch = next(train_dataset)
    state = update(state, train_batch)

    if step % config.eval_every == 0:
      train_loss = loss_fn(state.params, state.rng_key, train_batch)
      valid_loss = loss_fn(state.params, state.rng_key, next(eval_dataset))
      print(
            f"Step: {step}, Train ELBO: {-train_loss:.2f}, Valid ELBO: {-valid_loss:.2f}"
      )
      metrics.append((step, -train_loss, -valid_loss))
      model_out:VAEOutput = model.apply(state.params, state.rng_key, recon_data)

      plot_reconstruction(model_out.input, model_out.output, f"{outdir}/step_{step}.png")

  metrics = list(zip(*metrics))
  plot_metrics(metrics[0], metrics[1], metrics[2], outdir)



import matplotlib.pyplot as plt

def plot_metrics(steps, train_elbo, valid_elbo, outdir):
  plt.figure(figsize=(5, 3.5))
  plt.plot(steps, train_elbo, label="Train ELBO")
  plt.plot(steps, valid_elbo, label="Valid ELBO")
  plt.xlabel("Steps")
  plt.ylabel("ELBO")
  plt.legend()
  plt.savefig(f"{outdir}/elbo.png")

def plot_reconstruction(xs, x_hats, fname):
    fig, axs = plt.subplots(3, 3, figsize=(5, 5))

    for i,ax in enumerate(axs.flatten()):
        ax.plot(xs[i], label='True', lw=1, alpha=1, color='black')
        ax.plot(x_hats[i], label='Reconstructed', lw=0.5, alpha=0.3, color='tab:red')
        ax.axis("off")
    # remove all space between subplots
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(fname)