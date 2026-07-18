import hashlib
import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Optional

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
ARTIFACT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class SavedModelConfig:
    """Inference-only subset of the training configuration."""

    latent_dim: int
    data_dim: int
    normalize_decoder_output: bool

    @classmethod
    def from_dict(cls, config: dict, params: dict) -> "SavedModelConfig":
        known_fields = {item.name for item in fields(Config)}
        unknown_fields = set(config) - known_fields
        if unknown_fields:
            unknown = ", ".join(sorted(unknown_fields))
            raise TypeError(f"Unexpected saved config field(s): {unknown}")

        latent_dim = int(config["latent_dim"])
        data_dim = config.get("data_dim")
        if data_dim is None:
            data_dim = _infer_data_dim(params)
        if data_dim is None:
            raise ValueError("Saved model does not specify a data dimension")
        return cls(
            latent_dim=latent_dim,
            data_dim=int(data_dim),
            # Artifacts predating decoder normalization used raw output.
            normalize_decoder_output=bool(
                config.get("normalize_decoder_output", False)
            ),
        )


def array_sha256(data) -> str:
    """Return a stable checksum including an array's dtype and shape."""
    array = np.ascontiguousarray(np.asarray(data))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(json.dumps(array.shape).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def file_sha256(filename: str) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()
    with open(filename, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_model(
    state: TrainState,
    config: Config,
    train_metrics: TrainValMetrics,
    savedir: str,
    artifact_metadata: Optional[dict[str, Any]] = None,
):
    """Saves model parameters, config and library version using h5py.

    Adds an HDF5 file attribute ``library_version`` so downstream loads can
    detect mismatches between the saved artifact and running code, producing
    clearer guidance instead of opaque shape/field errors.
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
            else:
                # Convert JAX arrays to a native HDF5-compatible type.
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
        f.attrs["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION

        metadata = dict(artifact_metadata or {})
        metadata.setdefault("artifact_schema_version", ARTIFACT_SCHEMA_VERSION)
        f.create_dataset("artifact_metadata", data=json.dumps(metadata))

        # Store parameter shapes for forward compatibility.
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


def _infer_data_dim(params: dict) -> Optional[int]:
    decoder = params.get("decoder", {})
    for layer_name in ("fc4", "fc3"):
        layer = decoder.get(layer_name, {})
        kernel = layer.get("kernel")
        if kernel is not None and np.ndim(kernel) == 2:
            return int(np.shape(kernel)[-1])
    return None


def _decode_h5_string(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def get_model_version(
    savedir: str, model_fname: str = MODEL_FNAME
) -> Optional[str]:
    """Return the stored library version for a saved model if present."""
    filename = f"{savedir}/{model_fname}"
    try:
        with h5py.File(filename, "r") as f:
            return f.attrs.get("library_version", None)
    except FileNotFoundError:
        return None


def load_model(savedir: str, model_fname: str = MODEL_FNAME) -> ModelData:
    """Load parameters and config, augmenting errors with version guidance.

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
            config_dict = json.loads(_decode_h5_string(f["config"][()]))
            config = SavedModelConfig.from_dict(config_dict, params)
            artifact_metadata = {}
            if "artifact_metadata" in f:
                artifact_metadata = json.loads(
                    _decode_h5_string(f["artifact_metadata"][()])
                )
        except Exception as e:  # noqa: BLE001 broad to augment message
            msg = f"Failed to load model from '{filename}': {e}"
            if saved_version is not None and saved_version != CURRENT_VERSION:
                msg += (
                    f" (weights saved with starccato_jax {saved_version}, "
                    f"current code {CURRENT_VERSION}. Consider installing the "
                    "older version: "
                    f"'pip install starccato_jax=={saved_version}' "
                    f"or retraining the model under the current version.)"
                )
            raise RuntimeError(msg) from e

    return ModelData(
        params,
        config.latent_dim,
        config.data_dim,
        normalize_decoder_output=config.normalize_decoder_output,
        artifact_metadata=artifact_metadata,
        library_version=(
            _decode_h5_string(saved_version)
            if saved_version is not None
            else None
        ),
        artifact_sha256=file_sha256(filename),
    )


def load_model_metadata(
    savedir: str, model_fname: str = MODEL_FNAME
) -> dict[str, Any]:
    """Load artifact metadata without materializing the parameter arrays."""
    filename = f"{savedir}/{model_fname}"
    with h5py.File(filename, "r") as f:
        config = json.loads(_decode_h5_string(f["config"][()]))
        metadata = (
            json.loads(_decode_h5_string(f["artifact_metadata"][()]))
            if "artifact_metadata" in f
            else {}
        )
        return {
            "library_version": f.attrs.get("library_version"),
            "artifact_schema_version": int(
                f.attrs.get("artifact_schema_version", 1)
            ),
            "config": config,
            "artifact_metadata": metadata,
        }


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
