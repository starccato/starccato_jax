import os

import arviz as az
import corner
import matplotlib.pyplot as plt
import numpy as np

from ..model import ModelData, generate, reconstruct
from .plotting import _add_quantiles



def sampler_diagnostic_plots(inf_object, outdir):
    # Plot the trace plot
    az.plot_trace(inf_object, var_names=["z"])
    plt.savefig(os.path.join(outdir, "trace_plot.png"))
    plt.close()

    # Plot the corner plot
    corner.corner(inf_object, var_names=["z"])
    plt.savefig(os.path.join(outdir, "corner_plot.png"))
    plt.close()


def plot_ci(y_obs, z_posterior, vae_data: ModelData, nsamps=100, fname=None):
    z_samples = z_posterior[
        np.random.choice(z_posterior.shape[0], nsamps, replace=False)
    ]
    y_preds = generate(vae_data, z=z_samples)
    y_recon = reconstruct(y_obs, vae_data, n_reps=nsamps)

    ypred_qtls = np.quantile(y_preds, [0.025, 0.5, 0.975], axis=0)
    yrecn_qtls = np.quantile(y_recon, [0.025, 0.5, 0.975], axis=0)

    plt.figure(figsize=(5, 3.5))
    plt.plot(y_obs, label="Observed", color="black", lw=2, zorder=-1)
    ax = plt.gca()
    _add_quantiles(ax, ypred_qtls, "Posterior", "tab:orange")
    _add_quantiles(ax, yrecn_qtls, "Reconstruction", "tab:gray")

    plt.legend(frameon=False)
    if fname is not None:
        plt.savefig(fname)
