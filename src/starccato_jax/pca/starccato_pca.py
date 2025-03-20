import jax.numpy as jnp
from jax.random import PRNGKey

from ..data import load_training_data
from ..logging import logger
from ..starccato_model import StarccatoModel
from . import pca


class StarccatoPCA(StarccatoModel):
    def __init__(self, latent_dim: int = 32):
        """
        Initialize StarccatoPCA from a saved model file.
        If the file does not exist, an empty model is created.
        """
        self._latent_dim = latent_dim
        train_data, valid_data = load_training_data()

        n_training_samples, input_dim = train_data.shape

        self.input_dim = input_dim
        assert (
            latent_dim < input_dim
        ), f"latent_dim ({latent_dim}) must be less than or equal to the number of original dimensions ({n_orig_dim})"
        assert (
            n_training_samples > input_dim
        ), f"Number of training samples ({n_training_samples}) must be greater than the number of original dimensions ({n_orig_dim})"

        self._pca: pca.PCAState = pca.fit(
            train_data, n_components=latent_dim
        )  # PCA model will be loaded or trained
        self.train_mse = float(self.mse(train_data))
        self.valid_mse = float(self.mse(valid_data))
        logger.info(
            f"[{self}] Train MSE: {self.train_mse:.2e}, Valid MSE: {self.valid_mse:.2e}"
        )

    @property
    def latent_dim(self) -> int:
        return self._latent_dim

    def __repr__(self):
        return f"StarccatoPCA(z-dim={self.latent_dim})"

    def generate(
        self, z: jnp.ndarray = None, rng: PRNGKey = None, n: int = 1
    ) -> jnp.ndarray:
        if z is None:
            z = self.sample_latent(rng, n)
        return pca.recover(self._pca, z)

    def encode(self, x: jnp.ndarray, rng: PRNGKey = None) -> jnp.ndarray:
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Input data dimensions {x.shape[1]} != model encoder dim ({self.input_dim})"
            )
        return pca.transform(self._pca, x)

    def reconstruct(
        self, x: jnp.ndarray, rng: PRNGKey = None, n_reps: int = 1
    ) -> jnp.ndarray:
        z = self.encode(x)
        return self.generate(z)

    def mse(self, signals: jnp.ndarray) -> float:
        """
        Compute the mean squared error between the original signals and the reconstructed signals.
        """
        recon = self.reconstruct(signals)
        return float(jnp.mean((signals - recon) ** 2))
