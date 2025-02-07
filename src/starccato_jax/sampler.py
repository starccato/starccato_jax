import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import os
import arviz as az
import matplotlib.pyplot as plt
import numpy as np

import jax
import corner

from .trainer import load_model
from .model import generate, reconstruct


def _bayesian_model(y_obs: jnp.ndarray, vae_state, vae_latent_dims):
    # Define priors for the latent variables
    theta = numpyro.sample("z", dist.Uniform(0, 1).expand([vae_latent_dims]))
    # Generate the signal
    y_model = generate(vae_state, z=theta, rng=random.PRNGKey(0))

    # Likelihood (Assuming Gaussian noise)
    sigma = numpyro.sample("sigma", dist.HalfNormal(0.1))  # Noise level
    numpyro.sample("obs", dist.Normal(y_model, sigma), obs=y_obs)


def sample_latent_vars_given_data(data: jnp.ndarray, model_path: str, rng_int: int = 0, outdir='out_mcmc', nchains=2) -> MCMC:
    """
    Sample latent variables given the data.

    :param data: Data to condition the latent variables on.
    :param rng: Random number generator.
    :param latent_dim: Dimension of the latent space.
    :return: Sampled latent variables.
    """
    rng_key = random.PRNGKey(rng_int)
    vae_state, vae_latent_dims = load_model(model_path)
    nuts_kernel = NUTS(lambda y_obs: _bayesian_model(y_obs, vae_state, vae_latent_dims))
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=2000, num_chains=nchains, progress_bar=True)
    mcmc.run(rng_key, y_obs=data)

    os.makedirs(outdir, exist_ok=True)

    inf_object = az.from_numpyro(mcmc)
    print(az.summary(inf_object, var_names=['z']))
    _diagnostic_plots(inf_object, outdir)
    z_samples = inf_object.posterior['z'].values.reshape(-1, vae_latent_dims)
    plot_ci(data, z_samples, vae_state, vae_latent_dims, fname=os.path.join(outdir, 'ci_plot.png'))

    return mcmc


def _diagnostic_plots(inf_object, outdir):
    # Plot the trace plot
    az.plot_trace(inf_object, var_names=['z'])
    plt.savefig(os.path.join(outdir, 'trace_plot.png'))
    plt.close()

    # Plot the corner plot
    corner.corner(inf_object, var_names=['z'])
    plt.savefig(os.path.join(outdir, 'corner_plot.png'))
    plt.close()


def plot_ci(y_obs, z_posterior, state_params, latent_dim, nsamps=100, fname=None):
    z_samples = z_posterior[np.random.choice(z_posterior.shape[0], nsamps, replace=False)]
    y_preds = generate(state_params, latent_dim, z=z_samples)
    y_recon = reconstruct(y_obs, state_params, latent_dim, n_reps=nsamps)

    ypred_qtls = np.quantile(y_preds, [0.025, 0.5, 0.975], axis=0)
    yrecn_qtls = np.quantile(y_recon, [0.025, 0.5, 0.975], axis=0)

    plt.figure(figsize=(5, 3.5))
    plt.plot(y_obs, label='Observed', color='black', lw=2, zorder=-1)

    # filled region for 95% CI
    _plt_qtl(ypred_qtls, '95% CI (Predictive)', 'tab:orange')
    _plt_qtl(yrecn_qtls, '95% CI (Reconstruction)', 'tab:gray')

    plt.legend()
    if fname is not None:
        plt.savefig(fname)


def _plt_qtl(qtl, label, color):
    x = np.arange(len(qtl[0]))
    plt.fill_between(x, qtl[0], qtl[2], color=color, alpha=0.3, label=label, lw=0)
    plt.plot(qtl[1], color=color, lw=1, ls='--')
