import time
from typing import List, Tuple

import jax.random
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from jax.random import PRNGKey

from .. import credible_intervals
from ..data import get_default_weights, load_training_data
from ..logging import logger
from ..plotting import plot_model
from ..starccato_model import StarccatoModel
from .config import Config
from .core import (
    VAE,
    ModelData,
    encode,
    generate,
    load_model,
    reconstruct,
    train_vae,
)

__all__ = ["StarccatoVAE"]


class StarccatoVAE(StarccatoModel):
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir
        if model_dir is None or model_dir == "default_model":
            self.model_dir = get_default_weights()
        self._data: ModelData = load_model(self.model_dir)
        self._model: VAE = VAE(latents=self.latent_dim)

    def __repr__(self):
        return f"StarccatoVAE(z-dim={self.latent_dim})"

    @property
    def latent_dim(self) -> int:
        return self._data.latent_dim

    @classmethod
    def train(
        cls,
        model_dir: str,
        train_fraction: float = 0.8,
        config: Config = None,
        print_every: int = 1000,
        plot_every: int = np.inf,
        track_gradients: bool = False,
    ):
        train_data, val_data = load_training_data(
            train_fraction=train_fraction
        )
        config = config or Config()
        train_vae(
            train_data,
            val_data,
            config=config,
            save_dir=model_dir,
            print_every=print_every,
            plot_every=plot_every,
            track_gradients=track_gradients,
        )
        model = cls(model_dir)
        print(model.model_structure)
        return model

    def generate(
        self, z: jnp.ndarray = None, rng: PRNGKey = None, n: int = 1
    ) -> jnp.ndarray:
        if z is None:
            z = self.sample_latent(rng, n)
        return generate(self._data, z, model=self._model)

    def reconstruct(
        self, x: jnp.ndarray, rng: PRNGKey = None, n_reps: int = 1
    ) -> jnp.ndarray:
        reconstructed = reconstruct(
            x, self._data, rng=rng, n_reps=n_reps, model=self._model
        )
        return reconstructed

    def encode(self, x: jnp.ndarray, rng: PRNGKey = None) -> jnp.ndarray:
        return encode(x, self._data, rng=rng, model=self._model)

    def plot(
        self,
        ax=None,
        n: int = 1,
        z: jnp.ndarray = None,
        x: jnp.ndarray = None,
        ci: float = None,
        uniform_ci: bool = False,
        rng: PRNGKey = None,
        color: str = "tab:orange",
    ) -> Tuple[plt.Figure, List[plt.Axes]]:
        """Makes plots with the Starccato VAE model.

        If only n is provided, n Z samples will be randomly generated.

        Z and X cant be provided at the same time.
        If Z is provided, the Z will be used to generate X*
        If X is provided, the X will be used to reconstruct X*
        If X and n are provided, X* are reconstructed n times (different RNG)

        If CI is provided, the confidence interval will be plotted.
        If uniform_ci is True, the uniform CI will be plotted (otherswise pointwise CI).

        """
        rng = rng if rng is not None else PRNGKey(0)

        if z is not None and x is not None:
            raise ValueError("Z and X cant be provided at the same time.")

        if z is None and x is None:
            z = jax.random.normal(rng, shape=(n, self.latent_dim))

        xstar = None
        if z is not None:
            xstar = self.generate(z)

        if x is not None:
            xstar = self.reconstruct(x, rng=rng, n_reps=n)

        return plot_model(
            ax,
            n=n,
            x=x,
            xstar=xstar,
            ci=ci,
            uniform_ci=uniform_ci,
            color=color,
        )

    def reconstruction_coverage(
        self, x: jnp.ndarray, n: int = 100, ci: float = 0.9
    ) -> float:
        """Compute the reconstruction coverage of the model.

        The reconstruction coverage is the proportion of the time the true signal is within the
        confidence interval of the reconstructed signal.

        """
        xstar = self.reconstruct(x, n_reps=n)
        qtls = credible_intervals.pointwise_ci(xstar, ci=ci)
        return credible_intervals.coverage_probability(qtls, x)

    @property
    def model_structure(self) -> str:
        rng = jax.random.PRNGKey(0)
        return self._model.tabulate(rng, jnp.zeros(256), rng)
