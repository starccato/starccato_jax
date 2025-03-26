import os
import time
from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from jax.random import PRNGKey
from tqdm.auto import tqdm

from ....data import TrainValData
from ....logging import logger
from ...config import Config
from ..data_containers import TrainValMetrics
from ..io import save_model
from ..loss import vae_loss
from ..model import VAE, ModelData
from .plot_utils import save_training_plots

__all__ = ["train_vae"]


# ------------------------------
# Loss function and training step.
# ------------------------------


@partial(jax.jit, static_argnames=("model",))
def _train_step(
    state: train_state.TrainState,
    x: jnp.ndarray,
    rng: PRNGKey,
    model: VAE,
    beta: float,
):
    loss, grads = jax.value_and_grad(
        lambda params: vae_loss(params, x, rng, model, beta).loss
    )(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss, grads


def _create_train_state(
    rng: PRNGKey,
    latent_dim: int,
    data_len: int,
    learning_rate: float,
) -> Tuple[train_state.TrainState, VAE]:
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
    data: TrainValData,
    config: Config,
    save_dir="vae_outdir",
    plot_every=np.inf,
    track_gradients=False,
):
    os.makedirs(save_dir, exist_ok=True)
    n_train, data_len = data.train.shape
    rng = jax.random.PRNGKey(0)
    state, model = _create_train_state(
        rng, config.latent_dim, data_len, config.learning_rate
    )
    metrics = TrainValMetrics.for_epochs(config.epochs)

    t0 = time.time()
    progress_bar = tqdm(range(config.epochs), desc="Training")
    for epoch in progress_bar:
        rng, subkey = jax.random.split(rng)
        batches = data.generate_training_batches(config.batch_size, rng)
        train_args = (subkey, model, config.beta_schedule[epoch])

        epoch_grads = []  # Store the norm of the gradients for each epoch
        for batch in batches:
            state, _, batch_grad = _train_step(state, batch, *train_args)

            if track_gradients:  # TODO: do this in post (along with vae_loss)
                epoch_grads.append(_compute_norms_for_tree(batch_grad))

        model_data = ModelData(
            params=state.params, latent_dim=config.latent_dim
        )

        metrics.append(
            i=epoch,
            train_loss=vae_loss(
                model_data.params,
                data.train,
                rng,
                model,
                config.beta_schedule[epoch],
            ),
            val_loss=vae_loss(
                model_data.params,
                data.val,
                rng,
                model,
                config.beta_schedule[epoch],
            ),
            gradient_norms=epoch_grads,
        )
        progress_bar.set_postfix(dict(Metrics=f"{metrics}"))

        if plot_every != np.inf and epoch % plot_every == 0 and epoch > 0:
            progress_bar.set_description("Plotting")
            save_training_plots(
                model_data,
                metrics,
                save_dir,
                data,
                rng=rng,
                epoch=epoch,
            )
            progress_bar.set_description("Training")

    model_data = ModelData(params=state.params, latent_dim=config.latent_dim)
    save_training_plots(
        model_data, metrics, save_dir, data, rng=rng, epoch=config.epochs
    )
    save_model(state, config, metrics, savedir=save_dir)

    logger.info(f"Training complete. (time: {time.time() - t0:.2f}s)")
    save_training_plots(
        model_data, metrics, save_dir, data, rng=rng, final=True
    )


def _compute_norms_for_tree(data_tree):
    return jax.tree_util.tree_map(lambda x: jnp.linalg.norm(x), data_tree)
