import numpy as np

from starccato_jax.plotting import (
    plot_gradients,
    plot_latent_kl,
    plot_training_metrics,
)
from starccato_jax.vae.core.data_containers import Gradients, Losses, TrainValMetrics



def test_data_containers_append_and_gradients(tmp_path):
    metrics = TrainValMetrics.for_epochs(3, use_capacity=True)
    grad_norms = Gradients(3)
    fake_grads = [{"layer": {"w": np.array([1.0, 2.0])}}]

    # append once with non-empty gradients to exercise gradient tracking
    metrics.append(
        i=0,
        train_loss=Losses(0.1, 0.2, 0.3, 0.5, 0.0),
        val_loss=Losses(0.2, 0.3, 0.4, 0.5, 0.0),
        gradient_norms=fake_grads,
    )
    assert metrics.train_metrics.loss[0] == 0.3
    assert "layer_w" in metrics.gradient_norms.data


def test_plot_training_metrics_smoke(tmp_path):
    metrics = TrainValMetrics.for_epochs(5, use_capacity=True)
    # Fill with simple ramps so plotting has data
    metrics.train_metrics.reconstruction_loss[:] = np.linspace(0.5, 0.1, 5)
    metrics.val_metrics.reconstruction_loss[:] = np.linspace(0.6, 0.2, 5)
    metrics.train_metrics.kl_divergence[:] = np.linspace(0.05, 0.1, 5)
    metrics.val_metrics.kl_divergence[:] = np.linspace(0.06, 0.11, 5)
    metrics.train_metrics.capacity[:] = np.linspace(0.0, 1.0, 5)
    metrics.val_metrics.capacity[:] = np.linspace(0.0, 1.0, 5)

    out = tmp_path / "loss.png"
    plot_training_metrics(metrics, fname=str(out))
    assert out.exists()


def test_plot_gradients_smoke(tmp_path):
    data = {"layer_w": np.array([0.1, 0.2, 0.3])}
    out = tmp_path / "grads.png"
    plot_gradients(data, fname=str(out))
    assert out.exists()


def test_plot_latent_kl_smoke(tmp_path):
    kl = np.array([0.05, 0.2, 0.0, 0.15])
    out = tmp_path / "kl.png"
    plot_latent_kl(kl, threshold=0.1, fname=str(out))
    assert out.exists()

