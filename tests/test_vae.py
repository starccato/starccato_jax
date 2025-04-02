import os
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt

from starccato_jax import Config, StarccatoVAE
from starccato_jax.data import get_default_weights
from starccato_jax.vae.core.io import TrainValMetrics, load_loss_h5


def test_train_vae(outdir):
    # Train the VAE model
    t0 = time.time()
    vae = StarccatoVAE.train(
        model_dir=outdir,
        config=Config(latent_dim=8, epochs=10, cyclical_annealing_cycles=0),
    )
    runtime = round(time.time() - t0, 2)
    expected_runtime = 20
    assert (
        runtime < expected_runtime
    ), f"Training took {runtime} > {expected_runtime}s to complete"
    assert os.path.exists(outdir)

    # load and use VAE
    signal = vae.generate()[0]
    assert signal.shape == (256,)

    losses: TrainValMetrics = load_loss_h5(f"{outdir}/losses.h5")
    assert isinstance(losses, TrainValMetrics)
    assert losses.train_metrics.loss.shape == (10,)
    assert losses.val_metrics.loss.shape == (10,)


def test_model_structure(outdir):
    # save the model structure
    vae = StarccatoVAE()
    struc = vae.model_structure
    with open(f"{outdir}/model_structure.txt", "w") as f:
        f.write(struc)
    print(struc)


def test_default(outdir):
    get_default_weights(clean=True)
    vae = StarccatoVAE()
    z = jnp.zeros(vae.latent_dim)
    signal = vae.generate(z=z)
    encoded_z = vae.encode(signal)
    assert encoded_z.shape == (vae.latent_dim,)
    assert signal.shape == (256,)
    reconstructed = vae.reconstruct(signal)
    assert reconstructed.shape == (256,)


def test_plotting(outdir):
    vae = StarccatoVAE()
    signal = vae.generate()

    fig, axes = plt.subplots(2, 1, figsize=(4, 6), sharex=True)
    ax = axes[0]
    vae.plot(ax, n=3)
    vae.plot(ax, n=100, ci=0.9)
    vae.plot(ax, n=100, ci=0.5)
    fig.savefig(os.path.join(outdir, "plot_generated_sample.png"))

    ax = axes[1]
    coverage = vae.reconstruction_coverage(signal, n=100)
    vae.plot(ax, n=100, x=signal, ci=0.9, uniform_ci=True)
    vae.plot(ax, n=100, x=signal, ci=0.9, uniform_ci=False, color="tab:blue")
    ax.set_title(f"Reconstruction coverage: {coverage:.2f}")
    fig.savefig(os.path.join(outdir, "plot_reconstruction_cis.png"))
    plt.close(fig)
