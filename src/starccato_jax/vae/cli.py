import click

from .config import Config
from .starccato_vae import StarccatoVAE


@click.command("train_vae")
@click.option("--latent-dim", default=32, help="Latent dimension for the VAE")
@click.option("--epochs", default=1000, help="Number of training epochs")
@click.option("--cycles", default=3, help="Cyclical annealing cycles")
@click.option(
    "--outdir", default="model_out", help="Output directory for the model"
)
@click.option("--dataset", default="ccsne", help="Source of the training data")
def cli_train(
    latent_dim: int, epochs: int, cycles: int, outdir: str, dataset: str
):
    """Train the Starccato VAE model."""
    config = Config(
        latent_dim=latent_dim,
        epochs=epochs,
        cyclical_annealing_cycles=cycles,
        dataset=dataset,
    )
    StarccatoVAE.train(
        model_dir=outdir,
        config=config,
    )
    click.echo("Training complete.")


if __name__ == "__main__":
    cli_train()
