import numpy as np
from jax import numpy as jnp
from jax.random import PRNGKey

from .config import Config
from .core import ModelData, generate, load_model, reconstruct, train_vae
from .data import get_default_weights, load_training_data

__all__ = ["StarccatoVAE"]


class StarccatoVAE:
    def __init__(self, model_dir="default_model"):
        self.model_dir = model_dir
        if model_dir == "default_model":
            self.model_dir = get_default_weights()
        self.model: ModelData = load_model(self.model_dir)

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
        config = config or Config(
            latent_dim=8, epochs=10, cyclical_annealing_cycles=0
        )
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
        self, z: jnp.ndarray = None, rng: PRNGKey = None
    ) -> jnp.ndarray:
        return generate(self.model, z, rng)

    def reconstruct(
        self, x: jnp.ndarray, rng: PRNGKey = None, n_reps: int = 1
    ) -> jnp.ndarray:
        return reconstruct(x, self.model, rng=rng, n_reps=n_reps)
