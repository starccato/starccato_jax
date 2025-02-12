import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from jax.random import PRNGKey
from numpyro.infer import MCMC, NUTS

from ..model import ModelData, generate


def _bayesian_model(
    y_obs: jnp.ndarray, vae_data: ModelData, beta: float = 1.0
):
    """
    Bayesian model with tempering.

    Parameters:
      y_obs   : observed data.
      vae_data: model data (e.g. containing latent_dim).
      beta    : tempering parameter; beta=1 corresponds to full posterior.
    """

    # Define priors for the latent variables
    theta = numpyro.sample(
        "z", dist.Uniform(0, 1).expand([vae_data.latent_dim])
    )
    # Generate the signal
    y_model = generate(vae_data, z=theta, rng=random.PRNGKey(0))

    sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))  # Noise level

    # Compute the untempered log–likelihood (Assuming Gaussian noise)
    lnl = dist.Normal(y_model, sigma).log_prob(y_obs).sum()

    # Temper the likelihood (the power likelihood)
    numpyro.factor("likelihood", beta * lnl)

    # Save the untempered log–likelihood (for LnZ computation).
    numpyro.deterministic("untempered_loglike", lnl)


def _run_mcmc(
    y_obs: jnp.ndarray,
    vae_data: ModelData,
    num_samples: int,
    num_warmup: int,
    num_chains: int,
    rng: PRNGKey,
    beta: float = 1.0,
    progress_bar: bool = False,
) -> MCMC:
    """
    Run MCMC sampling.

    Parameters:
      y_obs       : observed data.
      vae_data    : model data (e.g. containing latent_dim).
      num_samples : number of samples to draw.
      num_warmup  : number of warmup steps.
      num_chains  : number of chains.
      rng         : random number generator.
      beta        : tempering parameter; beta=1 corresponds to full posterior.
    """
    nuts_kernel = NUTS(
        lambda y_obs: _bayesian_model(y_obs, vae_data, beta=beta)
    )
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(rng, y_obs=y_obs)
    return mcmc
