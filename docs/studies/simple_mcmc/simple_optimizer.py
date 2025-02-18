import os
import shutil

import jax
import jax.numpy as jnp
import jax.random as random
import matplotlib.pyplot as plt
import numpy as np
import optax
from matplotlib import gridspec

from starccato_jax.data import load_data
from starccato_jax.io import load_model
from starccato_jax.model import generate
from starccato_jax.plotting.gif_generator import generate_gif

MODEL_DIR = "vae_outdir"
PLOTDIR = "out_optimization"
os.makedirs(PLOTDIR, exist_ok=True)

train_data, val_data = load_data()
true = val_data[42]
model = load_model(MODEL_DIR)


def plot(signal, z: np.array, fname, i=None):
    # 1 col 2 row subplot ( 1 tall row for signal, 1 short row for z -- 1/4 height of signal)

    fig, axs = plt.subplots(
        2, 1, figsize=(4, 4), gridspec_kw={"height_ratios": [4, 1]}
    )

    for ax in axs:
        ax.set_axis_off()

    ax = axs[0]
    ax.plot(true, "k", lw=2)
    ax.plot(signal, "tab:orange", lw=1)
    ax.set_ylim(min(true) - 0.1, max(true) + 0.1)
    ax.set_xlim(0, len(true))
    if i is not None:
        ax.text(0.1, 0.9, f"Step {i}", transform=ax.transAxes)

    # z is a 1d array. Plot it as a pcolor plot -- 1 row, z.size columns
    ax = axs[1]
    ax.pcolor(z.reshape(1, -1), cmap="viridis", edgecolors="k", linewidths=0.0)

    plt.tight_layout(pad=0)
    plt.savefig(fname)
    plt.close("all")


@jax.jit
def compute_loss(params, rng):
    pred = generate(model, params, rng)
    return jnp.mean((pred - true) ** 2, axis=-1)


start_learning_rate = 1e-2
optimizer = optax.adam(start_learning_rate)

# Initialize parameters of the model + optimizer.
rng = random.PRNGKey(0)
params = jnp.zeros(model.latent_dim)
opt_state = optimizer.init(params)
target = generate(model, params, rng)

# A simple update loop.
for i in range(5000):
    rng, subkey = jax.random.split(rng)  # update the key for each iteration
    grads = jax.grad(compute_loss)(params, subkey)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

    if i % 50 == 0:
        print(f"Step {i}, loss: {compute_loss(params, rng)}")
        target = generate(model, params, rng)
        plot(target, os.path.join(PLOTDIR, f"signal_{i}.png"), i)

generate_gif(f"{PLOTDIR}/signal_*.png", f"signal.gif")

# remove the generated images folder

shutil.rmtree(PLOTDIR)
