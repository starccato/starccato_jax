import click

from .config import Config
from .starccato_vae import StarccatoVAE


@click.command("train_vae")
@click.option("--latent-dim", default=5, help="Latent dimension for the VAE")
@click.option("--epochs", default=1000, help="Number of training epochs")
@click.option(
    "--cycles",
    default=0,
    show_default=True,
    help="Cyclical beta annealing cycles (used only with --no-use-capacity)",
)
@click.option(
    "--outdir", default="model_out", help="Output directory for the model"
)
@click.option("--batch-size", default=64, help="Batch size for training")
@click.option("--dataset", default="ccsne", help="Source of the training data")
@click.option(
    "--seed",
    default=0,
    show_default=True,
    help="Seed for parameter initialization and training batches.",
)
@click.option(
    "--data-seed",
    default=0,
    show_default=True,
    help="Seed for the fixed train/validation split.",
)
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
    default=12.0,
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
    help="Weight on max(KL - capacity, 0) during capacity training",
)
@click.option(
    "--normalize-decoder-output/--raw-decoder-output",
    default=True,
    show_default=True,
    help="Constrain generated waveforms to zero mean and unit RMS.",
)
def cli_train(
    latent_dim: int,
    epochs: int,
    cycles: int,
    outdir: str,
    batch_size: int,
    dataset: str,
    seed: int,
    data_seed: int,
    use_capacity: bool,
    capacity_start: float,
    capacity_end: float,
    capacity_warmup_epochs: int,
    beta_capacity: float,
    normalize_decoder_output: bool,
):
    """Train the Starccato VAE model."""
    config = Config(
        latent_dim=latent_dim,
        epochs=epochs,
        cyclical_annealing_cycles=cycles,
        dataset=dataset,
        seed=seed,
        data_seed=data_seed,
        batch_size=batch_size,
        use_capacity=use_capacity,
        capacity_start=capacity_start,
        capacity_end=capacity_end,
        capacity_warmup_epochs=capacity_warmup_epochs,
        beta_capacity=beta_capacity,
        normalize_decoder_output=normalize_decoder_output,
    )
    StarccatoVAE.train(
        model_dir=outdir,
        config=config,
    )
    click.echo("Training complete.")


if __name__ == "__main__":
    cli_train()
