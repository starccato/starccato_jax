import os

import arviz as az
import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from .io import load_model
from .model import ModelData, generate
from .plotting import sampler_diagnostic_plots, plot_ci


def _bayesian_model(y_obs: jnp.ndarray, vae_data: ModelData):
    # Define priors for the latent variables
    theta = numpyro.sample(
        "z", dist.Uniform(0, 1).expand([vae_data.latent_dim])
    )
    # Generate the signal
    y_model = generate(vae_data, z=theta, rng=random.PRNGKey(0))

    # Likelihood (Assuming Gaussian noise)
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))  # Noise level
    numpyro.sample("obs", dist.Normal(y_model, sigma), obs=y_obs)


def sample_latent_vars_given_data(
    data: jnp.ndarray,
    model_path: str,
    rng_int: int = 0,
    outdir="out_mcmc",
    num_chains=2,
    num_warmup=500,
    num_samples=2000,
) -> MCMC:
    """
    Sample latent variables given the data.

    :param data: Data to condition the latent variables on.
    :param rng: Random number generator.
    :param latent_dim: Dimension of the latent space.
    :return: Sampled latent variables.
    """
    rng_key = random.PRNGKey(rng_int)
    vae_data = load_model(model_path)
    nuts_kernel = NUTS(lambda y_obs: _bayesian_model(y_obs, vae_data))
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )
    mcmc.run(rng_key, y_obs=data)

    os.makedirs(outdir, exist_ok=True)

    inf_object = az.from_numpyro(mcmc)
    print(az.summary(inf_object, var_names=["z"]))
    sampler_diagnostic_plots(inf_object, outdir)
    z_samples = inf_object.posterior["z"].values.reshape(
        -1, vae_data.latent_dim
    )
    plot_ci(
        data, z_samples, vae_data, fname=os.path.join(outdir, "ci_plot.png")
    )

    inf_object.to_netcdf(os.path.join(outdir, "inference.nc"))

    return mcmc

