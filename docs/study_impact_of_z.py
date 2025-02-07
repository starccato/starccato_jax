from starccato_jax.trainer import train_vae
from starccato_jax.data import load_data
import glob
import numpy as np
import matplotlib.pyplot as plt

import os
HERE = os.path.dirname(__file__)





z_sizes = [4, 8, 12, 16, 20]

## TRAIN

# train_data, val_data = load_data()
# for z_size in [4, 8, 12, 16, 20]:
#     train_vae(train_data, val_data, latent_dim=z_size, n_epochs=200,
#               save_dir=f"{HERE}/model_exploration/model_z{z_size}"
#               )


## GATHER LOSS DATA

train_losses, val_losses  = [], []
for z_size in [4, 8, 12, 16, 20]:
    # read the losses
    loss_fpath = f"{HERE}/model_exploration/model_z{z_size}/losses.txt"
    data = np.loadtxt(loss_fpath)
    train_losses.append(data[-1, 0])
    val_losses.append(data[-1, 1])



## PLOT
plt.figure(figsize=(8, 4))
plt.plot(z_sizes, train_losses, label="Train Loss")
plt.plot(z_sizes, val_losses, label="Val Loss")
plt.xlabel('Latent Dimension')
plt.ylabel('Loss')
plt.legend()

