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
@click.option(
    "--use-capacity/--no-use-capacity",
    default=True,
    show_default=True,
    help="Enable capacity-controlled KL objective",
)
@click.option(
    "--capacity-start",
    default=0.0,
    show_default=True,
    help="Initial target KL capacity (nats)",
)
@click.option(
    "--capacity-end",
    default=4.0,
    show_default=True,
    help="Final target KL capacity (nats)",
)
@click.option(
    "--capacity-warmup-epochs",
    default=500,
    show_default=True,
    help="Epochs over which to ramp capacity",
)
@click.option(
    "--beta-capacity",
    default=5.0,
    show_default=True,
    help="Weight on |KL - capacity| during capacity training",
)
def cli_train(
    latent_dim: int,
    epochs: int,
    cycles: int,
    outdir: str,
    batch_size: int,
    dataset: str,
    use_capacity: bool,
    capacity_start: float,
    capacity_end: float,
    capacity_warmup_epochs: int,
    beta_capacity: float,
):
    """Train the Starccato VAE model."""
    config = Config(
        latent_dim=latent_dim,
        epochs=epochs,
        cyclical_annealing_cycles=cycles,
        dataset=dataset,
        batch_size=batch_size,
        use_capacity=use_capacity,
        capacity_start=capacity_start,
        capacity_end=capacity_end,
        capacity_warmup_epochs=capacity_warmup_epochs,
        beta_capacity=beta_capacity,
    )
    StarccatoVAE.train(
        model_dir=outdir,
        config=config,
    )
    click.echo("Training complete.")


if __name__ == "__main__":
    cli_train()
