import os
from typing import Tuple

import jax.numpy as jnp
import numpy as np
import pandas as pd

_ROOT_URL = "https://raw.githubusercontent.com/starccato/data/main/training"
SIGNALS_CSV = f"{_ROOT_URL}/richers_1764.csv"
PARAMETERS_CSV = f"{_ROOT_URL}/richers_1764_parameters.csv"
HERE = os.path.dirname(__file__)
CACHE = f"{HERE}/data.npz"


def load_data(
    train_fraction: float = 0.8, clean: bool = False
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if not os.path.exists(CACHE) or clean:
        _download_and_save_data()
    data = np.load(CACHE)["data"]

    # shuffle data
    np.random.seed(0)
    np.random.shuffle(data)



    # standardise using max value from entire dataset, and zero mean for each row
    # data = data / np.max(np.abs(data))
    # data = data - np.mean(data, axis=1, keepdims=True)

    # # Standardize each sample (row) to have zero mean and unit variance.
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
    return train_data, val_data


def _download_and_save_data():
    parameters = pd.read_csv(PARAMETERS_CSV)
    data = pd.read_csv(SIGNALS_CSV).astype("float32")[
        parameters["beta1_IC_b"] > 0
    ]
    data = data.values.T[:, 140:]  # cut the first few datapoints
    np.savez(CACHE, data=data)
