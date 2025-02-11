import os

import arviz as az
import jax.numpy as jnp
import jax.random as random
from numpyro.infer import MCMC, NUTS
from tqdm.auto import tqdm

from .core import _run_mcmc
from ..io import load_model
from ..plotting import sampler_diagnostic_plots, plot_ci

from .stepping_stone_evidence import stepping_stone_evidence


def sample_latent_vars_given_data(
        data: jnp.ndarray,
        model_path: str,
        rng_int: int = 0,
        outdir="out_mcmc",
        num_warmup=500,
        num_samples=2000,
        num_chains=2,
        num_temps=1,
) -> MCMC:
    """
    Sample latent variables given the data.

    :param data: Data to condition the latent variables on.
    :param rng: Random number generator.
    :param latent_dim: Dimension of the latent space.
    :return: Sampled latent variables.
    """
    rng = random.PRNGKey(rng_int)
    vae_data = load_model(model_path)
    kwgs = dict(
        y_obs=data,
        vae_data=vae_data,
        num_samples=num_samples,
        num_warmup=num_warmup,
        rng=rng,
    )

    mcmc = _run_mcmc(**kwgs, beta=1.0, num_chains=num_chains, progress_bar=True)
    inf_object = az.from_numpyro(mcmc)

    if num_temps > 2:
        beta_schedule = jnp.linspace(1e-8, 1.0, num_temps)
        tempered_lnls = []
        for beta in tqdm(beta_schedule, "Tempering"):
            temp_mcmc = _run_mcmc(**kwgs, beta=beta, num_chains=1, progress_bar=False)
            tempered_lnls.append(temp_mcmc.get_samples()["untempered_loglike"])

        tempered_lnls = jnp.array(tempered_lnls)
        log_z, log_z_uncertainty = stepping_stone_evidence(
            tempered_lnls, beta_schedule, outdir, rng
        )
        print(f"Log Z: {log_z} +/- {log_z_uncertainty}")
        # add these to the inference object
        inf_object.sample_stats["log_z"] = log_z
        inf_object.sample_stats["log_z_uncertainty"] = log_z_uncertainty


    os.makedirs(outdir, exist_ok=True)


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
