import json
from dataclasses import asdict
from typing import List, Optional

import h5py
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import freeze
from flax.training.train_state import TrainState

from ..config import Config
from .data_containers import Losses, TrainValMetrics
from .model import ModelData
from starccato_jax import __version__ as CURRENT_VERSION

MODEL_FNAME = "model.h5"
LOSS_FNAME = "losses.h5"


def save_model(
    state: TrainState,
    config: Config,
    train_metrics: TrainValMetrics,
    savedir: str,
):
    """Saves model parameters, config and library version using h5py.

    Adds an HDF5 file attribute ``library_version`` so that downstream loads can
    detect mismatches between the saved artifact and the currently running
    code version, producing clearer guidance instead of opaque shape/field
    errors.
    """
    model_params = state.params
    filename = f"{savedir}/{MODEL_FNAME}"

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
                arr = np.array(value)
                if arr.dtype != np.dtype("O"):
                    h5group.create_dataset(key, data=arr)

    with h5py.File(filename, "w") as f:
        # Save params
        params_group = f.create_group("model_params")
        recursively_save(params_group, model_params)

        # Save config: Convert to JSON and store as a string dataset
        config_json = json.dumps(
            asdict(config)
        )  # Convert dictionary to JSON string
        f.create_dataset("config", data=config_json)
        # Store version metadata as a file attribute (scalar string)
        f.attrs["library_version"] = CURRENT_VERSION
        # Store architecture (shapes per parameter path) for forward compatibility
        def _flatten_shapes(d, prefix=""):
            out = {}
            for k, v in d.items():
                if isinstance(v, dict):
                    out.update(_flatten_shapes(v, f"{prefix}{k}/"))
                else:
                    try:
                        out[f"{prefix}{k}"] = list(np.array(v).shape)
                    except Exception:
                        out[f"{prefix}{k}"] = []
            return out
        arch = _flatten_shapes(model_params)
        f.create_dataset("architecture", data=json.dumps(arch))

    with h5py.File(f"{savedir}/{LOSS_FNAME}", "w") as f:
        loss_group = f.create_group("losses")
        recursively_save(loss_group, train_metrics.__dict__())


def _recursively_load(h5group):
    """Recursively load parameters from HDF5 into a nested dict."""
    data = {}
    for key in h5group.keys():
        if isinstance(h5group[key], h5py.Group):  # If it's a group, recurse
            data[key] = _recursively_load(h5group[key])
        else:  # Load dataset
            data[key] = jnp.array(h5group[key][()])
    return data


def get_model_version(savedir: str, model_fname: str = MODEL_FNAME) -> Optional[str]:
    """Return the stored library version for a saved model if present."""
    filename = f"{savedir}/{model_fname}"
    try:
        with h5py.File(filename, "r") as f:
            return f.attrs.get("library_version", None)
    except FileNotFoundError:
        return None


def load_model(savedir: str, model_fname: str = MODEL_FNAME) -> ModelData:
    """Loads model parameters and config, augmenting errors with version guidance.

    If a failure occurs while reconstructing the config/parameters (e.g. due to
    architectural or field changes), and version metadata is available, the
    raised error message will include a suggestion referencing the saved
    version vs the current version.
    """
    filename = f"{savedir}/{model_fname}"

    with h5py.File(filename, "r") as f:
        saved_version = f.attrs.get("library_version")
        try:
            params = freeze(_recursively_load(f["model_params"]))
            config_dict = json.loads(f["config"][()].decode("utf-8"))
            config = Config(**config_dict)
            saved_arch_json = None
            if "architecture" in f:
                saved_arch_json = f["architecture"][()].decode("utf-8")
        except Exception as e:  # noqa: BLE001 broad to augment message
            msg = f"Failed to load model from '{filename}': {e}"
            if saved_version is not None and saved_version != CURRENT_VERSION:
                msg += (
                    f" (weights saved with starccato_jax {saved_version}, "
                    f"current code {CURRENT_VERSION}. Consider installing the "
                    f"older version: 'pip install starccato_jax=={saved_version}' "
                    f"or retraining the model under the current version.)"
                )
            raise RuntimeError(msg) from e

    # Non-failing but version mismatch: optionally emit a warning for visibility
    major_minor_saved = (
        ".".join(saved_version.split(".")[:2]) if saved_version is not None else None
    )
    major_minor_current = ".".join(CURRENT_VERSION.split(".")[:2])


    if saved_version is not None and major_minor_saved != major_minor_current:
        # Lazy import to avoid circular logging dependency during early startup
        try:
            from starccato_jax.logging import logger  # type: ignore

            logger.warning(
                "Model artifact version %s differs from current %s. If you encounter "
                "shape or field errors later, retrain or pin the earlier version.",
                saved_version,
                CURRENT_VERSION,
            )
        except Exception:
            pass

    return ModelData(params, config.latent_dim, config.data_dim)


def load_loss_h5(fname: str) -> TrainValMetrics:
    """Loads loss metrics from HDF5 file"""
    with h5py.File(fname, "r") as f:
        # Load params
        data = _recursively_load(f["losses"])
        return TrainValMetrics(
            train_metrics=Losses(**data["train_metrics"]),
            val_metrics=Losses(**data["val_metrics"]),
            gradient_norms=data["gradient_norms"],
        )
