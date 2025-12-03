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
@click.option(
    "--batch-size", default=64, help="Batch size for training"
)
@click.option("--dataset", default="ccsne", help="Source of the training data")
def cli_train(
    latent_dim: int, epochs: int, cycles: int, outdir: str, batch_size:int, dataset: str
):
    """Train the Starccato VAE model."""
    config = Config(
        latent_dim=latent_dim,
        epochs=epochs,
        cyclical_annealing_cycles=cycles,
        dataset=dataset,
        batch_size=batch_size,
    )
    StarccatoVAE.train(
        model_dir=outdir,
        config=config,
    )
    click.echo("Training complete.")


if __name__ == "__main__":
    cli_train()
