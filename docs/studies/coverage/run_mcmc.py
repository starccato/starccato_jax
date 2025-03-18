import argparse
import os

import arviz as az
import jax
import numpy as np
from docutils.languages.ru import labels
from generate_injections import load_injections
from jax.random import PRNGKey
from starccato_sampler.plotting import plot_pe_comparison
from starccato_sampler.sampler import sample

from starccato_jax import StarccatoPCA, StarccatoVAE
from starccato_jax.data import load_training_data

RNG = PRNGKey(0)
NOISE_SIGMA = 1


def main(
    i: int,
    data: np.ndarray,
    z: np.ndarray = None,
    label: str = None,
    noise: bool = True,
):
    print(
        f"Running sampler on {label} data {i}/{len(data)}. True Z given: {z is not None}"
    )
    idx_lbl = f"{label}_{i}"
    outdir = f"out_mcmc/{idx_lbl}"
    os.makedirs(outdir, exist_ok=True)

    observation = data[i]
    true = data[i].copy()
    if noise:
        observation += np.random.normal(0, NOISE_SIGMA, size=observation.shape)

    kwgs = dict(
        data=observation,
        num_warmup=500,
        num_samples=1000,
        num_chains=1,
        verbose=True,
        noise_sigma=NOISE_SIGMA,
        stepping_stone_lnz=False,
        truth=dict(
            signal=true,
            latent=None if z is None else z[i],
        ),
    )

    lbls = ["vae", "pca"]
    for lbl, model in zip(lbls, [StarccatoVAE(), StarccatoPCA()]):
        sample(starccato_model=model, outdir=f"{outdir}/{lbl}", **kwgs)

    plot_pe_comparison(
        data_1d=observation,
        true_1d=true,
        result_fnames=[f"{outdir}/{lbl}/inference.nc" for lbl in lbls],
        labels=lbls,
        colors=["tab:blue", "tab:red"],
        fname=f"{outdir}/comparison_{idx_lbl}.pdf",
    )


if __name__ == "__main__":
    load_injections()
    parser = argparse.ArgumentParser()
    parser.add_argument("i", type=int)
    parser.add_argument("--dataset", type=str, default="val")
    args = parser.parse_args()

    RUN_ON_VALIDATION = args.dataset == "val"
    if RUN_ON_VALIDATION:
        _, data = load_training_data()
        z = None
        label = "val"
    else:
        data, z = load_injections()
        label = "inj"

    main(args.i, data, z, label)
