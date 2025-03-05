import os
import time
from functools import partial
from typing import List

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from matplotlib import pyplot as plt

from ..config import Config
from ..plotting import (
    generate_gif,
    plot_distributions,
    plot_reconstructions,
    plot_training_metrics,
)
from .io import save_model
from .loss import (
    TrainValMetrics,
    compute_metrics,
    cyclical_annealing_beta,
    vae_loss,
)
from .model import VAE, ModelData, reconstruct

__all__ = ["train_vae"]


# ------------------------------
# Loss function and training step.
# ------------------------------


@partial(jax.jit, static_argnames=("model",))
def _train_step(state, x, rng, model, beta):
    loss, grads = jax.value_and_grad(
        lambda params: vae_loss(params, x, rng, model, beta).loss
    )(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss


# ------------------------------
# Create training state with configurable latent_dim.
# ------------------------------
def _create_train_state(
    rng: jax.random.PRNGKey,
    latent_dim: int,
    data_len: int,
    learning_rate: float,
):
    model = VAE(latent_dim)
    # Initialize the model with dummy data of shape (1, DATA_LEN)
    params = model.init(rng, jnp.ones((1, data_len)), rng)["params"]
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx
    )
    return state, model


# ------------------------------
# Training loop.
# ------------------------------
def train_vae(
    train_data,
    val_data,
    config: Config,
    save_dir="vae_outdir",
    print_every=100,
    plot_every=np.inf,
):
    os.makedirs(save_dir, exist_ok=True)
    data_len = train_data.shape[1]
    t0 = time.time()

    rng = jax.random.PRNGKey(0)
    state, model = _create_train_state(
        rng, config.latent_dim, data_len, config.learning_rate
    )
    metrics: List[TrainValMetrics] = []
    n_train = train_data.shape[0]
    step = 0
    beta = cyclical_annealing_beta(
        n_epoch=config.epochs,
        n_cycle=config.cyclical_annealing_cycles,
        start=config.beta_start,
        stop=config.beta_end,
    )

    for epoch in range(config.epochs):
        # Shuffle training data indices.
        perm = np.random.permutation(n_train)
        rng, subkey = jax.random.split(rng)
        for i in range(0, n_train, config.batch_size):
            # beta = compute_beta(step, config.cycle_length)
            batch = train_data[perm[i : i + config.batch_size]]
            state, batch_loss = _train_step(
                state, batch, subkey, model, beta[epoch]
            )
            step += 1

        metrics.append(
            compute_metrics(
                state, train_data, subkey, model, val_data, beta[epoch]
            )
        )

        if print_every != np.inf and epoch % print_every == 0:
            print(f"Epoch {epoch}: {metrics[epoch]}")

        if plot_every != np.inf and epoch % plot_every == 0:
            model_data = ModelData(
                params=state.params, latent_dim=config.latent_dim
            )
            _save_training_plots(
                model_data,
                metrics,
                save_dir,
                val_data,
                rng=rng,
                epoch=config.epochs,
            )

    model_data = ModelData(params=state.params, latent_dim=config.latent_dim)
    _save_training_plots(
        model_data, metrics, save_dir, val_data, rng=rng, epoch=config.epochs
    )

    save_model(state, config, metrics, savedir=save_dir)

    print(f"Training complete. (time: {time.time() - t0:.2f}s)")
    if plot_every < np.inf:
        _save_gifs(save_dir)


def _save_training_plots(
    model_data: ModelData, metrics, save_dir, val_data, rng=None, epoch=None
):
    plot_training_metrics(metrics, fname=f"{save_dir}/loss.png")
    plot_reconstructions(
        model_data,
        val_data,
        fname=f"{save_dir}/plots/reconstruction_E{epoch}.png",
        title=f"Epoch {epoch}",
        rng=rng,
    )
    plot_distributions(
        val_data,
        reconstruct(val_data, model_data, rng),
        fname=f"{save_dir}/plots/distributions_E{epoch}.png",
        title=f"Epoch {epoch}",
    )
    plt.close("all")


def _save_gifs(save_dir):
    generate_gif(
        image_pattern=f"{save_dir}/plots/reconstruction_E*.png",
        output_gif=f"{save_dir}/training_reconstructions.gif",
    )
    generate_gif(
        image_pattern=f"{save_dir}/plots/reconstruction_E*.png",
        output_gif=f"{save_dir}/training_reconstructions.gif",
    )
