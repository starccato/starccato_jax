from click.testing import CliRunner

from starccato_jax.vae.cli import cli_train


def test_cli_train(outdir):
    runner = CliRunner()
    out = f"{outdir}/test_cli_train"

    latent_dim = 8
    epochs = 10

    result = runner.invoke(
        cli_train,
        [
            "--latent-dim",
            latent_dim,
            "--epochs",
            epochs,
            "--cycles",
            "1",
            "--outdir",
            out,
        ],
    )

    assert result.exit_code == 0
    assert "Training complete." in result.output
    assert "Training VAE with config:" in result.output
    assert f"latent_dim={latent_dim}" in result.output
