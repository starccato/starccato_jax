import jax.numpy as jnp
import jax.random as random
from jax.random import PRNGKey
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from ..model import ModelData, generate


def _bayesian_model(y_obs: jnp.ndarray, vae_data: ModelData, beta: float = 1.0):
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

    # Temper the likelihood.
    numpyro.factor("likelihood", beta * lnl)

    # Save the untempered log–likelihood (for LnZ computation).
    numpyro.deterministic("untempered_loglike", lnl)


def stepping_stone_estimate_with_uncertainty(tempered_loglikes: jnp.ndarray, beta_schedule: jnp.ndarray):
    """

    Adapted from bilby.core.sampler.stepping_stone.stepping_stone_estimator


    Compute the stepping–stone (GSS) log marginal likelihood estimate and its uncertainty.

    For each adjacent pair in the beta_schedule, compute:

        R_i = Δβ · ν + log(mean(exp(Δβ · (L - ν))))

    where L are the untempered log–likelihood samples at the lower beta,
    and ν = max(L) (for numerical stability).

    The uncertainty in each R_i is approximated by:

        σ_R_i ≈ sqrt(var(exp(Δβ · (L - ν))) / (n · m^2))

    and the overall variance is the sum of the variances.
    """
    K = len(beta_schedule)
    pseudo_ratios = []
    variances = []
    for i in range(K - 1):
        diff = beta_schedule[i + 1] - beta_schedule[i]
        L = jnp.array(tempered_loglikes_list[i])  # shape (num_samples,)
        n_samples = L.size
        nu = jnp.max(L)
        # Compute x = exp(diff * (L - ν))
        x = jnp.exp(diff * (L - nu))
        m = jnp.mean(x)
        R_i = diff * nu + jnp.log(m)
        pseudo_ratios.append(R_i)
        # Delta–method uncertainty approximation:
        var_x = jnp.var(x, ddof=1)
        sigma_R_i = jnp.sqrt(var_x / (n_samples * (m ** 2)))
        variances.append(sigma_R_i ** 2)
    log_z = jnp.sum(pseudo_ratios)
    log_z_uncertainty = jnp.sqrt(jnp.sum(variances))
    return log_z, log_z_uncertainty


def _run_mcmc(
        y_obs: jnp.ndarray,
        vae_data: ModelData,
        num_samples: int,
        num_warmup: int,
        num_chains: int,
        rng: PRNGKey,
        beta: float = 1.0,
        progress_bar: bool = False
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
    nuts_kernel = NUTS(lambda y_obs: _bayesian_model(y_obs, vae_data, beta=beta))
    mcmc = MCMC(
        nuts_kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )
    mcmc.run(rng, y_obs=y_obs)
    return mcmc
