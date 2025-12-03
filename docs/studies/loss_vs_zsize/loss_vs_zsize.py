import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm.auto import tqdm

from starccato_jax import Config, StarccatoVAE

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "model_exploration")

z_sizes = np.geomspace(8, 64, num=16, dtype=int)
EPOCHS = 2000


def main():
    mses = []
    for z_size in tqdm(z_sizes, desc="Training VAEs"):
        outdir = os.path.join(OUT, f"model_z{z_size}")
        config = Config(
            latent_dim=z_size, epochs=EPOCHS, cyclical_annealing_cycles=1
        )
        vae = StarccatoVAE.train(
            model_dir=outdir, config=config, plot_every=10000
        )

        # compute MSE with validation set
        val_data = config.data.val
        reconstructions = vae.reconstruct(val_data)
        mse = np.mean((val_data - reconstructions) ** 2)
        mses.append(mse)

    np.savetxt(os.path.join(outdir, "mses.txt"), mses)

    ## GATHER LOSS DATA
    train_losses, val_losses = [], []
    for z_size in z_sizes:
        # read the losses
        loss_fpath = f"{HERE}/model_exploration/model_z{z_size}/losses.txt"
        data = np.loadtxt(loss_fpath)
        train_losses.append(data[0, -1])
        val_losses.append(data[1, -1])



    # cache the losses
    np.savetxt(
        f"{HERE}/model_exploration/losses.txt",
        np.array([train_losses, val_losses]),
    )

    # plot losses
    plt.figure(figsize=(8, 4))
    plt.plot(z_sizes, train_losses, label="Train Loss")
    plt.plot(z_sizes, val_losses, label="Val Loss")
    plt.xlabel("Latent Dimension")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{HERE}/model_exploration/loss_vs_z.png")

    plt.figure(figsize=(8, 4))
    plt.plot(z_sizes, mses, label="Val MSE")
    plt.xlabel("Latent Dimension")
    plt.ylabel("MSE")
    plt.savefig(f"{HERE}/model_exploration/mse_vs_z.png")

if __name__ == "__main__":
    main()
