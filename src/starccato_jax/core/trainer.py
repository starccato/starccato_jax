import os
import time
from functools import partial
from typing import Any, List

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state

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
    # Define a loss function that accepts the current parameters.
    def loss_fn(params):
        # Include the mutable batch statistics.
        variables = {"params": params, "batch_stats": state.batch_stats}
        rngs = {"dropout": jax.random.split(rng)[1]}

        # Apply the model with dropout and BatchNorm in training mode.
        # mutable=["batch_stats"] lets Flax know to update batch stats.
        (recon_x, mean, logvar), new_model_state = model.apply(
            variables, x, rng, mutable=["batch_stats"], rngs=rngs
        )
        loss = vae_loss(recon_x, x, mean, logvar, beta).loss
        return loss, new_model_state

    # Calculate loss and gradients, also get updated batch stats.
    (loss, new_model_state), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params
    )
    # Update the state with the new gradients and batch statistics.
    new_state = state.apply_gradients(
        grads=grads, batch_stats=new_model_state["batch_stats"]
    )
    return new_state, loss


class TrainState(train_state.TrainState):
    batch_stats: Any


def _create_train_state(
    rng: jax.random.PRNGKey,
    latent_dim: int,
    data_len: int,
    learning_rate: float,
):
    # When initializing, we need to create a dummy input that matches the expected shape.
    # For our convolutional model with normalization, we assume the input shape is (batch, data_len, channels).
    dummy_input = jnp.ones((1, data_len, 1))
    # Initialize the model (set train flag to True for training mode).
    model = VAE(latent_dim, output_shape=(data_len, 1), train=True)
    variables = model.init(rng, dummy_input, rng)
    params = variables["params"]
    batch_stats = variables.get("batch_stats")
    tx = optax.adam(learning_rate)
    # Include batch_stats in the train state.
    state = TrainState.create(
        apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats
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
    model_data = None

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

        model_data = ModelData(
            params=state.params,
            latent_dim=config.latent_dim,
            batch_stats=state.batch_stats,
        )
        metrics.append(
            compute_metrics(
                model_data, train_data, subkey, val_data, beta[epoch]
            )
        )

        if print_every != np.inf and epoch % print_every == 0:
            print(f"Epoch {epoch}: {metrics[epoch]}")

        if plot_every != np.inf and epoch % plot_every == 0:
            _save_training_plots(
                model_data,
                metrics,
                save_dir,
                val_data,
                rng=rng,
                epoch=config.epochs,
            )

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


def _save_gifs(save_dir):
    generate_gif(
        image_pattern=f"{save_dir}/plots/reconstruction_E*.png",
        output_gif=f"{save_dir}/training_reconstructions.gif",
    )
    generate_gif(
        image_pattern=f"{save_dir}/plots/reconstruction_E*.png",
        output_gif=f"{save_dir}/training_reconstructions.gif",
    )
