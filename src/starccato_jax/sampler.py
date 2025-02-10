import jax.numpy as jnp
import jax.random as random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import os
import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import corner

from .io import load_model
from .model import generate, reconstruct, ModelData
from .plotting import _add_quantiles


def _bayesian_model(y_obs: jnp.ndarray, vae_data:ModelData):
    # Define priors for the latent variables
    theta = numpyro.sample("z", dist.Uniform(0, 1).expand([vae_data.latent_dim]))
    # Generate the signal
    y_model = generate(vae_data, z=theta, rng=random.PRNGKey(0))

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
    vae_data = load_model(model_path)
    nuts_kernel = NUTS(lambda y_obs: _bayesian_model(y_obs, vae_data))
    mcmc = MCMC(nuts_kernel, num_warmup=500, num_samples=2000, num_chains=nchains, progress_bar=True)
    mcmc.run(rng_key, y_obs=data)

    os.makedirs(outdir, exist_ok=True)

    inf_object = az.from_numpyro(mcmc)
    print(az.summary(inf_object, var_names=['z']))
    _diagnostic_plots(inf_object, outdir)
    z_samples = inf_object.posterior['z'].values.reshape(-1, vae_data.latent_dim)
    plot_ci(data, z_samples, vae_data, fname=os.path.join(outdir, 'ci_plot.png'))

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


def plot_ci(y_obs, z_posterior, vae_data:ModelData, nsamps=100, fname=None):
    z_samples = z_posterior[np.random.choice(z_posterior.shape[0], nsamps, replace=False)]
    y_preds = generate(vae_data, z=z_samples)
    y_recon = reconstruct(y_obs, vae_data, n_reps=nsamps)

    ypred_qtls = np.quantile(y_preds, [0.025, 0.5, 0.975], axis=0)
    yrecn_qtls = np.quantile(y_recon, [0.025, 0.5, 0.975], axis=0)

    plt.figure(figsize=(5, 3.5))
    plt.plot(y_obs, label='Observed', color='black', lw=2, zorder=-1)
    ax = plt.gca()
    _add_quantiles(ax, ypred_qtls,  'Posterior',  'tab:orange')
    _add_quantiles(ax, yrecn_qtls, 'Reconstruction', 'tab:gray')

    plt.legend(frameon=False)
    if fname is not None:
        plt.savefig(fname)

