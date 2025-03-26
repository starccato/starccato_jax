import datetime
import glob
import os

import h5py
import matplotlib.pyplot as plt

from starccato_jax import Config, StarccatoVAE
from starccato_jax.plotting.plot_distributions import plot_distributions
from starccato_jax.vae.core.io import load_loss_h5

# enable 64-bit precision (default is 32-bit) -- will make things slower (but more accurate??)
# jax.config.update("jax_enable_x64", False)

OUTDIR = "model_outputs"


def timestamp() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


def make_comparison_plots(model_dir_1, model_dir_2):
    dirs = [model_dir_1, model_dir_2]
    labels = [os.path.basename(p) for p in dirs]

    # load signal distributions
    signals = []
    for model_dir in dirs:
        with h5py.File(f"{model_dir}/signals.h5", "r") as f:
            signals.append(f["signals"][()])

    # plot signal distributions
    fig, _ = plot_distributions(
        signals[0],
        signals[1],
        title=f"{labels[0]} vs {labels[0]}",
        labels=labels,
    )
    fig.show()

    # load +  plot the losses
    metrics = [load_loss_h5(f"{model_dir}/losses.h5") for model_dir in dirs]
    for i, (label, metric) in enumerate(zip(labels, metrics)):
        plt.plot(
            metric.train_metrics.loss,
            label=f"{labels[i]} Train",
            color=f"C{i}",
        )
        plt.plot(
            metric.val_metrics.loss,
            label=f"{labels[i]} Val",
            color=f"C{i}",
            linestyle="--",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def train_new_model(config):
    model_dir = f"{OUTDIR}/vae_{timestamp()}"
    os.makedirs(model_dir, exist_ok=True)
    model = StarccatoVAE.train(
        config=config,
        model_dir=model_dir,
        track_gradients=False,
    )
    signals = model.generate(n=500)
    with h5py.File(f"{model_dir}/signals.h5", "w") as f:
        f.create_dataset("signals", data=signals)

    return model_dir


if __name__ == "__main__":
    config1 = Config(
        latent_dim=32,
        learning_rate=1e-3,
        epochs=200,
        batch_size=126,
        cyclical_annealing_cycles=2,
    )
    config2 = Config(
        latent_dim=16,
        learning_rate=1e-3,
        epochs=200,
        batch_size=126,
        cyclical_annealing_cycles=2,
    )
    train_new_model(config1)
    train_new_model(config2)
    model_dirs = glob.glob(f"{OUTDIR}/*")
    make_comparison_plots(model_dirs[-2], model_dirs[-1])
