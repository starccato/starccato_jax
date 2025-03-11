from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp
from jax.random import PRNGKey


class StarccatoModel(ABC):
    @property
    @abstractmethod
    def latent_dim(self) -> int:
        pass

    @abstractmethod
    def generate(
        self, z: jnp.ndarray = None, rng: PRNGKey = None, n: int = 1
    ) -> jnp.ndarray:
        pass

    @abstractmethod
    def reconstruct(
        self, x: jnp.ndarray, rng: PRNGKey = None, n_reps: int = 1
    ) -> jnp.ndarray:
        pass

    @abstractmethod
    def encode(self, x: jnp.ndarray, rng: PRNGKey = None) -> jnp.ndarray:
        pass

    def sample_latent(self, rng: PRNGKey = None, n: int = 1) -> jnp.ndarray:
        if rng is None:
            rng = PRNGKey(0)
        return jax.random.normal(rng, (n, self.latent_dim))
