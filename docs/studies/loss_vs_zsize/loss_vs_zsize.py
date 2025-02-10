import glob
import os

import matplotlib.pyplot as plt
import numpy as np

from starccato_jax.data import load_data
from starccato_jax.sampler import sample_latent_vars_given_data
from starccato_jax.trainer import Config, train_vae

HERE = os.path.dirname(__file__)

z_sizes = [8, 12, 16, 20, 24, 28, 32]

## TRAIN
train_data, val_data = load_data()
for z_size in z_sizes:
    config = Config(latent_dim=z_size, epochs=600, cyclical_annealing_cycles=4)
    train_vae(
        train_data,
        val_data,
        config,
        save_dir=f"{HERE}/model_exploration/model_z{z_size}",
    )
    sample_latent_vars_given_data(
        val_data[0],
        model_path=f"{HERE}/model_exploration/model_z{z_size}",
        outdir=f"{HERE}/model_exploration/model_z{z_size}/mcmc",
    )

#
# ## GATHER LOSS DATA
#
# train_losses, val_losses  = [], []
# for z_size in z_sizes:
#     # read the losses
#     loss_fpath = f"{HERE}/model_exploration/model_z{z_size}/losses.txt"
#     data = np.loadtxt(loss_fpath)
#     train_losses.append(data[0,-1])
#     val_losses.append(data[1,-1])
#
#
#
# ## PLOT
# plt.figure(figsize=(8, 4))
# plt.plot(z_sizes, train_losses, label="Train Loss")
# plt.plot(z_sizes, val_losses, label="Val Loss")
# plt.xlabel('Latent Dimension')
# plt.ylabel('Loss')
# plt.legend()
# plt.savefig(f"{HERE}/model_exploration/loss_vs_z.png")
