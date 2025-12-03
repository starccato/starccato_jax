import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from ....data import TrainValData
from ....logging import logger
from ....plotting import (
    generate_gif,
    plot_latent_kl,
    plot_distributions,
    plot_gradients,
    plot_loss_in_terminal,
    plot_reconstructions,
    plot_training_metrics,
)
from ..data_containers import ModelData, TrainValMetrics
from ..model import VAE, reconstruct


def save_training_plots(
    model_data: ModelData,
    metrics: TrainValMetrics,
    save_dir: str,
    data: TrainValData,
    rng: jax.random.PRNGKey,
    epoch: int = None,
    final: bool = False,
):
    label, title = "", ""
    if epoch is not None:
        label = f"_E{epoch}"
        title = f"Epoch {epoch}"

    plot_training_metrics(metrics, fname=f"{save_dir}/plots/loss.png")
    for d, name in zip([data.train, data.val], ["train", "val"]):
        plot_reconstructions(
            model_data,
            d,
            fname=f"{save_dir}/plots/{name}_reconstruction{label}.png",
            title=f"{title} {name.capitalize()} Reconstruction",
            rng=rng,
        )
        plot_distributions(
            d,
            reconstruct(d, model_data, rng),
            fname=f"{save_dir}/plots/{name}_distributions{label}.png",
            title=f"{title} {name.capitalize()} Distribution",
        )

    if not metrics.gradient_norms.is_empty:
        plot_gradients(
            metrics.gradient_norms.data,
            fname=f"{save_dir}/plots/gradient_norms{label}.png",
        )

    plt.close("all")

    if final:
        plot_loss_in_terminal(metrics)
        _plot_latent_kl(model_data, data, rng, save_dir)
        _save_gifs(save_dir)


def _save_gifs(save_dir):
    generate_gif(
        image_pattern=f"{save_dir}/plots/train_reconstruction_E*.png",
        output_gif=f"{save_dir}/plots/training_reconstructions.gif",
    )
    generate_gif(
        image_pattern=f"{save_dir}/plots/train_distributions_E*.png",
        output_gif=f"{save_dir}/plots/training_distributions.gif",
    )


def _plot_latent_kl(
    model_data: ModelData, data: TrainValData, rng: jax.random.PRNGKey, save_dir: str
):
    """Compute per-dimension KL on the train set and save a diagnostic plot."""
    model = VAE(model_data.latent_dim, data_dim=model_data.data_dim)
    _, mean, logvar = model.apply(
        {"params": model_data.params}, data.train, rng
    )
    kl_per_dim = np.array(
        jax.device_get(
            jnp.mean(-0.5 * (1 + logvar - jnp.square(mean) - jnp.exp(logvar)), axis=0)
        )
    )
    threshold = 0.1

    active_mask = kl_per_dim >= threshold
    active_dims = np.where(active_mask)[0].tolist()
    dead_dims = np.where(~active_mask)[0].tolist()
    n_active = len(active_dims)
    n_total = kl_per_dim.shape[0]

    logger.info(
        "KL per dim: %d/%d active (>=%.2f)=%s dead=%s",
        n_active,
        n_total,
        threshold,
        active_dims,
        dead_dims,
    )
    plot_latent_kl(
        kl_per_dim,
        threshold=threshold,
        fname=f"{save_dir}/plots/kl_per_dim.png",
        title=(
            f"Latent KL per dimension (train set)\n"
            f"{n_active}/{n_total} active (>= {threshold})"
        ),
    )
