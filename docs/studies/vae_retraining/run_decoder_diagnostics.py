"""Measure VAE fidelity, local decoder conditioning, and latent collisions.

Examples:

    uv run python docs/studies/vae_retraining/run_decoder_diagnostics.py \
      --model-dir out_blip --dataset blip --output /tmp/blip_geometry.json

    uv run python docs/studies/vae_retraining/run_decoder_diagnostics.py \
      --model-dir out_blip --dataset blip \
      --posterior-samples /path/to/samples.npz \
      --map-diagnostics /path/to/diagnostics.json \
      --output /tmp/blip_posterior_geometry.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from starccato_jax.data import TrainValData
from starccato_jax.vae.core import (
    decoder_collision_summary,
    encode_mean,
    generate,
    load_model,
    reconstruction_fidelity,
    summarize_decoder_geometry,
    summarize_fidelity,
)


def latent_names(files: list[str]) -> list[str]:
    return sorted(
        (name for name in files if name.startswith("z_")),
        key=lambda name: int(name.split("_")[-1]),
    )


def posterior_latents(path: Path, n: int, seed: int) -> np.ndarray:
    with np.load(path) as samples:
        names = latent_names(list(samples.files))
        values = np.column_stack(
            [np.asarray(samples[name]).reshape(-1) for name in names]
        )
    if len(values) <= n:
        return values
    rng = np.random.default_rng(seed)
    return values[rng.choice(len(values), size=n, replace=False)]


def map_latents(path: Path) -> tuple[np.ndarray, list[float]]:
    document = json.loads(path.read_text())
    attempts = document["map"]["attempts"]
    names = latent_names(list(attempts[0]["values"]))
    values = np.asarray(
        [[attempt["values"][name] for name in names] for attempt in attempts]
    )
    log_density = [float(attempt["log_density"]) for attempt in attempts]
    return values, log_density


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("ccsne", "blip"), required=True)
    parser.add_argument("--posterior-samples", type=Path)
    parser.add_argument("--map-diagnostics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=float, default=4096.0)
    parser.add_argument("--flow", type=float, default=300.0)
    parser.add_argument("--fmax", type=float, default=800.0)
    parser.add_argument("--n-prior", type=int, default=256)
    parser.add_argument("--n-posterior", type=int, default=256)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260718)
    args = parser.parse_args()

    model_data = load_model(str(args.model_dir))
    validation = TrainValData.load(
        source=args.dataset, seed=args.data_seed
    ).val
    validation_z = np.asarray(encode_mean(validation, model_data))
    validation_reconstruction = np.asarray(generate(model_data, validation_z))
    fidelity = summarize_fidelity(
        reconstruction_fidelity(
            np.asarray(validation),
            validation_reconstruction,
            sample_rate=args.sample_rate,
            flow=args.flow,
            fmax=args.fmax,
        )
    )

    rng = np.random.default_rng(args.seed)
    prior_z = rng.normal(size=(args.n_prior, model_data.latent_dim))
    result = {
        "model_dir": str(args.model_dir.resolve()),
        "model_sha256": model_data.artifact_sha256,
        "dataset": args.dataset,
        "data_seed": args.data_seed,
        "frequency_band_hz": [args.flow, args.fmax],
        "sample_rate_hz": args.sample_rate,
        "validation_fidelity": fidelity,
        "prior_geometry": summarize_decoder_geometry(model_data, prior_z),
        "prior_collisions": decoder_collision_summary(model_data, prior_z),
        "validation_geometry": summarize_decoder_geometry(
            model_data,
            validation_z[: args.n_prior],
        ),
    }

    posterior_z = None
    if args.posterior_samples:
        posterior_z = posterior_latents(
            args.posterior_samples, args.n_posterior, args.seed
        )
        result["posterior_samples"] = str(args.posterior_samples.resolve())
        result["posterior_geometry"] = summarize_decoder_geometry(
            model_data, posterior_z
        )
        result["posterior_collisions"] = decoder_collision_summary(
            model_data, posterior_z
        )

    if args.map_diagnostics:
        mode_z, log_density = map_latents(args.map_diagnostics)
        result["map_diagnostics"] = str(args.map_diagnostics.resolve())
        result["map_log_density_descending"] = sorted(
            log_density, reverse=True
        )
        result["map_geometry"] = summarize_decoder_geometry(model_data, mode_z)
        result["map_collisions"] = decoder_collision_summary(
            model_data,
            mode_z,
            minimum_latent_distance=0.5,
            mismatch_threshold=1e-3,
        )
        if posterior_z is not None:
            distance = np.linalg.norm(
                mode_z[:, None, :] - posterior_z[None, :, :], axis=-1
            )
            result["map_to_posterior"] = [
                {
                    "attempt_index": int(index),
                    "log_density": float(log_density[index]),
                    "minimum_latent_distance": float(np.min(distance[index])),
                }
                for index in np.argsort(log_density)[::-1]
            ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
