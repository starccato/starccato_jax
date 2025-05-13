import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import jax.numpy as jnp
import numpy as np
import pandas as pd
from jax.random import PRNGKey, permutation

from .urls import CCSNE_PARAMETERS_URL, CCSNE_SIGNALS_URL

__all__ = ["CCSNeDataset", "TrainValData"]

HERE = os.path.dirname(__file__)
CCSNE_CACHE = f"{HERE}/ccsne_data.npz"
BLIP_CACHE = f"{HERE}/blip_data.dat"


@dataclass
class TrainValData(ABC):
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
        type: str = "richers_ccsne",
    ):
        data = cls._load_raw(clean=clean)

        # assert that n-rows (different examples) are greater than the len of 1 signal
        # TODO

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


class CCSNeDataset(TrainValData):
    """
    Class to load the CCSNe dataset.
    """

    @staticmethod
    def _load_raw(clean: bool = False) -> np.ndarray:
        if not os.path.exists(CCSNE_CACHE) or clean:
            parameters = pd.read_csv(CCSNE_PARAMETERS_URL)
            data = pd.read_csv(CCSNE_SIGNALS_URL).astype("float32")[
                parameters["beta1_IC_b"] > 0
            ]
            data = data.values.T[:, 140:]  # cut the first few datapoints
            np.savez(CCSNE_CACHE, data=data)
        data = np.load(CCSNE_CACHE)["data"]
        return data


class BlipDataset(TrainValData):
    """
    Class to load the Blip dataset.
    """

    @staticmethod
    def _load_raw(clean: bool = False) -> np.ndarray:
        return np.loadtxt(GLITCH_FNAME)


#
#
# import zipfile
#
# # Open a zip file in read mode ("r")
# with zipfile.ZipFile('your_zip_file.zip', 'r') as zip_ref:
#     # Get a list of all files in the zip archive
#     file_list = zip_ref.namelist()
#     print("Files in the zip archive:", file_list)
#
#     # Extract all files to the current directory
#     # zip_ref.extractall()
#
#     # Extract all files to a specific directory
#     # zip_ref.extractall("destination_folder")
#
#     # Extract a single file
#     # zip_ref.extract('file_to_extract.txt', "destination_folder")
