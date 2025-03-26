import jax
import matplotlib.pyplot as plt

from ....data import TrainValData
from ....plotting import (
    generate_gif,
    plot_distributions,
    plot_gradients,
    plot_loss_in_terminal,
    plot_reconstructions,
    plot_training_metrics,
)
from ..data_containers import ModelData, TrainValMetrics
from ..model import reconstruct


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
