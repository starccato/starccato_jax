import argparse
import os

import arviz as az
import jax
import numpy as np
from generate_injections import load_injections
from jax.random import PRNGKey
from starccato_sampler.sampler import sample

from starccato_jax.data import load_training_data
from starccato_jax.starccato_vae import StarccatoVAE

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
    outdir = f"out_mcmc/{label}_{i}"
    os.makedirs(outdir, exist_ok=True)

    observation = data[i]
    true = data[i].copy()
    if noise:
        observation += np.random.normal(0, NOISE_SIGMA, size=observation.shape)

    sample(
        data=observation,
        outdir=outdir,
        num_warmup=500,
        num_samples=1000,
        num_chains=2,
        verbose=True,
        noise_sigma=NOISE_SIGMA,
        stepping_stone_lnz=False,
        truth=dict(
            signal=true,
            latent=None if z is None else z[i],
        ),
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
