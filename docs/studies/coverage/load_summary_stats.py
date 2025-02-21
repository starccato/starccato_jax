import glob

import arviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

REGEX = "out_mcmc/val*/inference.nc"

KEYS = [
    "reconstruction_coverage",
    "posterior_coverage",
    "sampling_runtime",
    "ss_lnz",
    "ss_lnz_uncertainty",
    "gauss_lnz",
    "gauss_lnz_uncertainty",
]


def load_all_summary_stats():
    files = glob.glob(REGEX)
    summary_stats = []
    for file in tqdm(files, desc="Loading summary stats"):
        inf = arviz.from_netcdf(file)
        stats = {key: float(inf.sample_stats[key].values) for key in KEYS}
        summary_stats.append(stats)
    return pd.DataFrame(summary_stats)


stats = load_all_summary_stats()
stats.to_csv("summary_stats.csv", index=False)

# plot the histograms of the coverage, plot the difference between two LnZ methods, plot histogram of the runtime
nbins = 30
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].hist(
    stats["reconstruction_coverage"],
    bins=nbins,
    alpha=0.5,
    label="Reconstruction Coverage",
)
axs[0].hist(
    stats["posterior_coverage"],
    bins=nbins,
    alpha=0.5,
    label="Posterior Coverage",
)
axs[0].set_title("Coverage")
axs[0].set_xlabel("Coverage")
axs[0].legend()

axs[1].hist(stats["ss_lnz"] - stats["gauss_lnz"], bins=nbins)
axs[1].set_title("SS LnZ - Gauss LnZ")
axs[1].set_xlabel("LnZ Diff")

axs[2].hist(stats["sampling_runtime"], bins=nbins)
axs[2].set_title("Sampling Runtime")
axs[2].set_xlabel("Runtime (s)")

plt.tight_layout()
plt.savefig("summary_stats.png")
