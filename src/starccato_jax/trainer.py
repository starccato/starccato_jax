import os

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import train_state
from functools import partial
from flax.training import checkpoints
import optax

from .model import VAE
from .utils import plot_loss, plot_reconstructions


__all__ = ['train_vae', 'load_model']



# ------------------------------
# Loss function and training step.
# ------------------------------
def vae_loss(params, x, rng, model):
    reconstructed, mean, logvar = model.apply({'params': params}, x, rng)
    reconstruction_loss = jnp.mean((x - reconstructed) ** 2)
    kl_divergence = -0.5 * jnp.mean(1 + logvar - jnp.square(mean) - jnp.exp(logvar))
    return reconstruction_loss + kl_divergence

@partial(jax.jit, static_argnames=("model",))
def train_step(state, x, rng, model):
    loss, grads = jax.value_and_grad(lambda params: vae_loss(params, x, rng, model))(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

# ------------------------------
# Create training state with configurable latent_dim.
# ------------------------------
def create_train_state(rng:jax.random.PRNGKey, latent_dim:int, data_len:int, learning_rate:float ):
    model = VAE(latent_dim)
    # Initialize the model with dummy data of shape (1, DATA_LEN)
    params = model.init(rng, jnp.ones((1, data_len)), rng)['params']
    tx = optax.adam(learning_rate)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state, model

# ------------------------------
# Training loop.
# ------------------------------
def train_vae(train_data, val_data, latent_dim, n_epochs=1000, batch_size=32, learning_rate:float=1e-3, save_dir='vae_outdir'):
    os.makedirs(save_dir, exist_ok=True)
    data_len = train_data.shape[1]

    rng = jax.random.PRNGKey(0)
    state, model = create_train_state(rng, latent_dim, data_len, learning_rate)
    train_losses = []
    val_losses = []
    n_train = train_data.shape[0]
    n_val = val_data.shape[0]

    for epoch in range(n_epochs):
        # Shuffle training data indices.
        perm = np.random.permutation(n_train)
        epoch_train_loss = 0.
        n_batches = 0
        rng, subkey = jax.random.split(rng)
        for i in range(0, n_train, batch_size):
            batch_idx = perm[i:i + batch_size]
            # Since the train_data is already a JAX device array, slicing is efficient.
            batch = train_data[batch_idx]
            state, loss = train_step(state, batch, subkey, model)
            epoch_train_loss += loss
            n_batches += 1
        epoch_train_loss /= n_batches
        train_losses.append(epoch_train_loss)

        # Validation loss
        epoch_val_loss = 0.
        n_batches_val = 0
        for i in range(0, n_val, batch_size):
            batch = val_data[i:i + batch_size]
            loss = vae_loss(state.params, batch, subkey, model)
            epoch_val_loss += loss
            n_batches_val += 1
        epoch_val_loss /= n_batches_val
        val_losses.append(epoch_val_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Train Loss = {epoch_train_loss:.4f}, Val Loss = {epoch_val_loss:.4f}")

    plot_loss(train_losses, val_losses, fname=f"{save_dir}/loss.png")
    plot_reconstructions(state, model, val_data, fname=f"{save_dir}/reconstructions.png")
    save_model(state, dict(latent_dim=latent_dim), filename=f"{save_dir}/MODEL")
    _save_losses(train_losses, val_losses, fname=f"{save_dir}/losses.txt")
    print("Training complete.")



# --------------------------------------------
# Save and Load Functions
# --------------------------------------------

def save_model(state, config, filename="/tmp/flax_ckpt/MODEL"):
    """ Saves model parameters and config using Orbax CheckpointManager """
    checkpt = {'state': state, 'config': config,}
    checkpoints.save_checkpoint(
        ckpt_dir=filename,
        target=checkpt,
        step=0,
        overwrite=True,
        keep=2
        )
    print(f"Model saved to {filename}")


def load_model(savedir="/tmp/flax_ckpt/"):
    """ Loads model parameters and config using Orbax CheckpointManager """
    raw_restored = checkpoints.restore_checkpoint(ckpt_dir=f"{savedir}/MODEL", target=None)
    return raw_restored["state"], raw_restored["config"]['latent_dim']


def _save_losses(train_loss, valid_loss, fname):
    np.savetxt(fname, np.array([train_loss, valid_loss]))
    print(f"Losses saved to {fname}")
