from tqdm.auto import trange

from starccato_jax.data import load_data
from starccato_jax.sampler import sample_latent_vars_given_data
from starccato_jax.trainer import Config, train_vae
from starccato_jax.model import generate
from starccato_jax.io import load_model
import os

import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS



import optax
import jax

MODEL_DIR = 'vae_outdir'

# train model

train_data, val_data = load_data()
true = val_data[42]


model = load_model(MODEL_DIR)


@jax.jit
def compute_loss(target):
    return jnp.mean((target - true) ** 2)


start_learning_rate = 1e-1
optimizer = optax.adam(start_learning_rate)

# Initialize parameters of the model + optimizer.
rng = random.PRNGKey(0)
params = jnp.zeros(model.latent_dim)
opt_state = optimizer.init(params)
target = generate(model, params, rng)

# A simple update loop.
for _ in range(1000):
    grads = jax.grad(compute_loss)(target)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

assert jnp.allclose(params, target_params), \
'Optimization should retrieve the target params used to generate the data.'



def fit_data_with_vae_and_optimizer(model, x, y):
    model.eval()
    target = torch.FloatTensor(y).unsqueeze(0)

    # Initialize latent vector
    z = torch.randn(1, latent_dim, requires_grad=True)
    optimizer = optim.Adam([z], lr=0.1)

    for _ in range(1000):
        optimizer.zero_grad()
        generated = model.decode(z)
        loss = nn.functional.mse_loss(generated, target)
        loss.backward()
        optimizer.step()

    return generated.detach().numpy()[0]


# Fit the data using our VAE
optimized_fit = fit_data_with_vae_and_optimizer(model, x, y_obs)

# Plot the results
plt.figure(figsize=(4, 3))
plt.scatter(x, y_obs, label='Original Data')
plt.plot(x, optimized_fit, 'r-', label='VAE Fit')
plt.title("Fitting New Data with VAE using optimizer")
plt.xlabel("x")
plt.ylabel("y")
plt.xlim(0,1)
plt.legend()
plt.show()
