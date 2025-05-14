import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey, permutation

from .datasets import load_dataset

__all__ = ["TrainValData"]


@dataclass
class TrainValData:
    train: jnp.ndarray
    val: jnp.ndarray

    def __repr__(self):
        return f"TrainValData(train={self.train.shape}, val={self.val.shape})"

    @staticmethod
    @abstractmethod
    def _load_raw(clean: bool = False) -> np.ndarray:
        raise ValueError(
            "The TrainValData class is abstract and cannot be instantiated directly."
        )

    @classmethod
    def load(
        cls,
        train_fraction: float = 0.8,
        clean: bool = False,
        seed: int = 0,
        source: str = "ccsne",
    ):
        data = load_dataset(source, clean=clean)

        # assert that n-rows (different examples) are greater than the len of 1 signal
        assert data.shape[0] > 1, "Data must have more than one example."

        # shuffle data
        np.random.seed(seed)
        np.random.shuffle(data)

        # Standardize each sample (row) to have zero mean and unit variance.
        mus = np.mean(data, axis=1, keepdims=True)
        sigmas = np.std(data, axis=1, keepdims=True)
        data = (data - mus) / sigmas

        # Split data into training (80%) and validation (20%) sets.
        n_total = data.shape[0]
        n_train = int(train_fraction * n_total)
        train_data_np = data[:n_train]
        val_data_np = data[n_train:]

        # Convert preprocessed data to JAX device arrays once.
        train_data = jnp.array(train_data_np)
        val_data = jnp.array(val_data_np)
        return cls(train=train_data, val=val_data)

    @property
    def combined(self) -> jnp.ndarray:
        return jnp.concatenate((self.train, self.val), axis=0)

    def generate_training_batches(
        self, batch_size: int, rng: PRNGKey
    ) -> jnp.ndarray:
        n = self.train.shape[0]
        perm = permutation(rng, n)
        batches = [
            self.train[perm[i : i + batch_size]]
            for i in range(0, n, batch_size)
        ]
        # drop the last batch if it is not full
        if len(batches[-1]) < batch_size:
            batches = batches[:-1]
        return jnp.array(batches)
