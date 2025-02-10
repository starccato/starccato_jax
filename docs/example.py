import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from starccato_jax.data import load_data
from starccato_jax.io import load_model
from starccato_jax.model import reconstruct
from starccato_jax.sampler import sample_latent_vars_given_data
from starccato_jax.trainer import Config, train_vae

HERE = os.path.dirname(__file__)

Z_SIZE = 20


def main():
    train_data, val_data = load_data()
    train_vae(
        train_data,
        val_data,
        config=Config(
            latent_dim=Z_SIZE,
            epochs=30,
            cyclical_annealing_cycles=0,
        ),
        save_dir=f"{HERE}/model_exploration/model_z{Z_SIZE}",
    )
    model_data = load_model(f"{HERE}/model_exploration/model_z{Z_SIZE}")
    reconstructed = reconstruct(train_data[0], model_data)

    sample_latent_vars_given_data(
        train_data[0],
        model_path=f"{HERE}/model_exploration/model_z{Z_SIZE}",
        outdir=f"{HERE}/model_exploration/model_z{Z_SIZE}/mcmc_train",
    )
    sample_latent_vars_given_data(
        val_data[0],
        model_path=f"{HERE}/model_exploration/model_z{Z_SIZE}",
        outdir=f"{HERE}/model_exploration/model_z{Z_SIZE}/mcmc_validation",
    )
    sample_latent_vars_given_data(
        reconstructed,
        model_path=f"{HERE}/model_exploration/model_z{Z_SIZE}",
        outdir=f"{HERE}/model_exploration/model_z{Z_SIZE}/mcmc_reconstructed",
    )


if __name__ == "__main__":
    main()
