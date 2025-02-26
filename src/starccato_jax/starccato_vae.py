import jax.random
import matplotlib.pyplot as plt
import numpy as np
from jax import numpy as jnp
from jax.random import PRNGKey

from . import credible_intervals
from .config import Config
from .core import ModelData, generate, load_model, reconstruct, train_vae
from .data import get_default_weights, load_training_data
from .plotting import add_quantiles

__all__ = ["StarccatoVAE"]


class StarccatoVAE:
    def __init__(self, model_dir: str = None):
        self.model_dir = model_dir
        if model_dir is None or model_dir == "default_model":
            self.model_dir = get_default_weights()
        self._data: ModelData = load_model(self.model_dir)

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
        )
        return cls(model_dir)

    def generate(
        self, z: jnp.ndarray = None, rng: PRNGKey = None, n: int = 1
    ) -> jnp.ndarray:
        return generate(self._data, z, rng, n)

    def reconstruct(
        self, x: jnp.ndarray, rng: PRNGKey = None, n_reps: int = 1
    ) -> jnp.ndarray:
        return reconstruct(x, self._data, rng=rng, n_reps=n_reps)

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
    ) -> None:
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

        if ax is None:
            fig, ax = plt.subplots()
        fig = ax.get_figure()

        if z is not None and x is not None:
            raise ValueError("Z and X cant be provided at the same time.")

        if z is None and x is None:
            z = jax.random.normal(rng, shape=(n, self.latent_dim))

        if z is not None:
            xstar = self.generate(z)
        else:
            xstar = self.reconstruct(x, rng=rng, n_reps=n)

        if x is not None:
            ax.plot(x, lw=1, color="black")

        if ci is not None:
            if uniform_ci:
                qtls = credible_intervals.uniform_ci(xstar, ci=ci)
            else:
                qtls = credible_intervals.pointwise_ci(xstar, ci=ci)
            add_quantiles(ax, qtls, color=color)
        else:
            lw = 0.05 if n > 50 else 1
            alpha = 0.25 if n > 50 else 1
            for i in range(n):
                ax.plot(xstar[i], lw=lw, alpha=alpha, color=color)

        return fig, ax

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
