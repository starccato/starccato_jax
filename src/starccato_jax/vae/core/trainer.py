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
from tqdm.auto import tqdm

from ...logging import logger
from ...plotting import (
    generate_gif,
    plot_distributions,
    plot_gradients,
    plot_loss_in_terminal,
    plot_reconstructions,
    plot_training_metrics,
)
from ..config import Config
from .data_containers import Losses, TrainValMetrics
from .io import save_model
from .loss import cyclical_annealing_beta, vae_loss
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
    return state, loss, grads


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
    track_gradients=False,
):
    os.makedirs(save_dir, exist_ok=True)
    data_len = train_data.shape[1]
    t0 = time.time()

    rng = jax.random.PRNGKey(0)
    state, model = _create_train_state(
        rng, config.latent_dim, data_len, config.learning_rate
    )
    metrics = TrainValMetrics.for_epochs(config.epochs)
    n_train = train_data.shape[0]
    step = 0
    beta = cyclical_annealing_beta(
        n_epoch=config.epochs,
        n_cycle=config.cyclical_annealing_cycles,
        start=config.beta_start,
        stop=config.beta_end,
    )

    model_data = None
    progress_bar = tqdm(range(config.epochs), desc="Training")
    for epoch in progress_bar:
        # Shuffle training data indices.
        perm = np.random.permutation(n_train)
        rng, subkey = jax.random.split(rng)

        avg_grad_norm = None
        epoch_grads = []  # Store the norm of the gradients for each epoch
        for i in range(0, n_train, config.batch_size):
            # beta = compute_beta(step, config.cycle_length)
            batch = train_data[perm[i : i + config.batch_size]]
            state, batch_loss, batch_grad = _train_step(
                state, batch, subkey, model, beta[epoch]
            )
            if track_gradients:
                # Compute the norm of the gradients
                grad_norm = jax.tree_util.tree_map(
                    lambda x: jnp.linalg.norm(x), batch_grad
                )
                epoch_grads.append(grad_norm)

            step += 1

        model_data = ModelData(
            params=state.params, latent_dim=config.latent_dim
        )

        metrics.append(
            i=epoch,
            train_loss=vae_loss(
                model_data.params, train_data, rng, model, beta[epoch]
            ),
            val_loss=vae_loss(
                model_data.params, val_data, rng, model, beta[epoch]
            ),
            gradient_norms=epoch_grads,
        )
        progress_bar.set_postfix(dict(Metrics=f"{metrics}"))

        if plot_every != np.inf and epoch % plot_every == 0 and epoch > 0:
            progress_bar.set_description("Plotting")
            _save_training_plots(
                model_data,
                metrics,
                save_dir,
                val_data,
                rng=rng,
                epoch=config.epochs,
            )
            progress_bar.set_description("Training")

    model_data = ModelData(params=state.params, latent_dim=config.latent_dim)
    _save_training_plots(
        model_data, metrics, save_dir, val_data, rng=rng, epoch=config.epochs
    )
    plot_loss_in_terminal(metrics)
    if track_gradients:
        plot_gradients(
            metrics.gradient_norms.data, fname=f"{save_dir}/gradients.png"
        )

    save_model(state, config, metrics, savedir=save_dir)

    logger.info(f"Training complete. (time: {time.time() - t0:.2f}s)")
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
