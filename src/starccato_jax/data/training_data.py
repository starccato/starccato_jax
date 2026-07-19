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

    @classmethod
    def load(
        cls,
        train_fraction: float = 0.8,
        clean: bool = False,
        seed: int = 0,
        source: str = "ccsne",
    ):
        data = load_dataset(source, clean=clean)

        if not np.all(np.isfinite(data)):
            raise ValueError("Training data contains non-finite values.")

        if data.shape[0] <= 1:
            raise ValueError("Data must have more than one example.")

        # Preserve the historical MT19937 split without mutating global NumPy
        # RNG state or a dataset loader's cached array.
        data = np.array(data, copy=True)
        np.random.RandomState(seed).shuffle(data)

        # Standardize each sample (row) to have zero mean and unit variance.
        mus = np.mean(data, axis=1, keepdims=True)
        sigmas = np.std(data, axis=1, keepdims=True)
        zero_variance = np.flatnonzero(sigmas[:, 0] <= np.finfo(float).eps)
        if zero_variance.size:
            preview = ", ".join(str(i) for i in zero_variance[:5])
            raise ValueError(
                "Cannot standardize zero-variance waveform row(s): " + preview
            )
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
        batches = []
        for i in range(0, n, batch_size):
            stop = i + batch_size
            batches.append(self.train[perm[i:stop]])
        # drop the last batch if it is not full
        if len(batches[-1]) < batch_size:
            batches = batches[:-1]
        return jnp.array(batches)
