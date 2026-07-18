from .diagnostics import (
    decoder_collision_summary,
    decoder_jacobian_singular_values,
    reconstruction_fidelity,
    summarize_decoder_geometry,
    summarize_fidelity,
)
from .io import load_model, load_model_metadata
from .model import VAE, ModelData, encode, encode_mean, generate, reconstruct
from .trainer import train_vae

__all__ = [
    "VAE",
    "ModelData",
    "decoder_collision_summary",
    "decoder_jacobian_singular_values",
    "encode",
    "encode_mean",
    "generate",
    "load_model",
    "load_model_metadata",
    "reconstruct",
    "reconstruction_fidelity",
    "summarize_decoder_geometry",
    "summarize_fidelity",
    "train_vae",
]
