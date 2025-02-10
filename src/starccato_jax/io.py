import json
from dataclasses import asdict
from typing import List

import h5py
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import freeze
from flax.training.train_state import TrainState

from .config import Config
from .loss import Losses, TrainValMetrics, aggregate_metrics
from .model import ModelData

MODEL_FNAME = "model.h5"
LOSS_FNAME = "losses.h5"


def save_model(
    state: TrainState,
    config: Config,
    train_metrics: List[TrainValMetrics],
    savedir: str,
):
    """Saves model parameters and config using h5py"""
    model_params = state.params
    filename = f"{savedir}/{MODEL_FNAME}"

    metrics = aggregate_metrics(train_metrics)

    # iterate through model_params dict and convert to native HDF5 equivalent
    def recursively_save(h5group, data):
        """Recursively save params dictionary into HDF5 groups."""
        for key, value in data.items():
            if isinstance(
                value, dict
            ):  # If it's a dictionary, create a subgroup
                subgroup = h5group.create_group(key)
                recursively_save(subgroup, value)
            else:  # Assume it's a JAX array, convert to NumPy for HDF5 compatibility
                h5group.create_dataset(key, data=np.array(value))

    with h5py.File(filename, "w") as f:
        # Save params
        params_group = f.create_group("model_params")
        recursively_save(params_group, model_params)

        # Save config: Convert to JSON and store as a string dataset
        config_json = json.dumps(
            asdict(config)
        )  # Convert dictionary to JSON string
        f.create_dataset("config", data=config_json)

    with h5py.File(f"{savedir}/{LOSS_FNAME}", "w") as f:
        loss_group = f.create_group("losses")
        recursively_save(loss_group, asdict(metrics))

    print(f"Model saved to {filename}")


def load_model(savedir: str) -> ModelData:
    """Loads model parameters and configs"""
    filename = f"{savedir}/{MODEL_FNAME}"

    def recursively_load(h5group):
        """Recursively load parameters from HDF5 into a nested dict."""
        data = {}
        for key in h5group.keys():
            if isinstance(
                h5group[key], h5py.Group
            ):  # If it's a group, recurse
                data[key] = recursively_load(h5group[key])
            else:  # Load dataset
                data[key] = jnp.array(h5group[key][()])
        return data

    with h5py.File(filename, "r") as f:
        # Load params
        params = freeze(recursively_load(f["model_params"]))

        # Load config: Read JSON string and convert back to dictionary
        config = Config(**json.loads(f["config"][()].decode("utf-8")))

    return ModelData(params, config.latent_dim)
