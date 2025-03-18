import glob

import arviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

REGEX = "out_mcmc/val*/{model}/inference.nc"


def load_all_summary_stats(model="vae"):
    files = glob.glob(REGEX.format(model=model))
    summary_stats = []
    for file in tqdm(files, desc="Loading summary stats"):
        inf = arviz.from_netcdf(file)

        truth = np.array(inf.sample_stats["true_signal"])
        quantiles = inf.sample_stats["quantiles"].values[1]
        err = np.mean((quantiles - truth) ** 2)
        # err = float(inf.sample_stats["mse"].values)
        runtime = float(inf.sample_stats["sampling_runtime"].values)
        summary_stats.append(dict(mse=err, sampling_runtime=runtime))
    return pd.DataFrame(summary_stats)


vae_stats = load_all_summary_stats("vae")
pca_stats = load_all_summary_stats("pca")

# linehistograms of the mse and speedup factor of VAE
fig, axs = plt.subplots(1, 2, figsize=(6, 3))
axs[0].hist(
    vae_stats["mse"], bins=30, alpha=0.75, label="VAE", histtype="step", lw=2
)
axs[0].hist(
    pca_stats["mse"],
    bins=30,
    alpha=0.75,
    label="PCA",
    histtype="step",
    ls="--",
    lw=2,
)
axs[0].set_xlabel("MSE")
axs[0].legend(frameon=False)

speedup = pca_stats["sampling_runtime"] / vae_stats["sampling_runtime"]
axs[1].hist(speedup, bins=30, alpha=0.75, histtype="step", lw=2)
axs[1].set_xlabel("pca/vae time")

# remove yticks from both axes
for ax in axs:
    ax.set_yticks([])

plt.tight_layout()
plt.savefig("pca_vs_vae.png")
