import argparse

from jax.random import PRNGKey
from starccato_sampler.sampler import sample

from starccato_jax.data import load_training_data

_, VAL_DATA = load_training_data()
RNG = PRNGKey(0)


def main(i: int):
    print(f"Running sampler on validation data {i}")
    sample(
        VAL_DATA[i],
        outdir="out_mcmc/val{i}",
        num_warmup=500,
        num_samples=1000,
        num_chains=1,
        verbose=True,
        stepping_stone_lnz=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("i", type=int)
    args = parser.parse_args()
    main(args.i)
