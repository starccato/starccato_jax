"""Inference-facing diagnostics for trained Starccato VAEs."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .data_containers import ModelData
from .model import VAE, _has_legacy_decoder, encode_mean, generate


def _model(model_data: ModelData) -> VAE:
    return VAE(
        model_data.latent_dim,
        data_dim=model_data.data_dim,
        use_legacy_decoder=_has_legacy_decoder(model_data.params),
        normalize_decoder_output=model_data.normalize_decoder_output,
    )


def reconstruction_fidelity(
    target: np.ndarray,
    reconstruction: np.ndarray,
    *,
    sample_rate: float,
    flow: float | None = None,
    fmax: float | None = None,
    noise_psd: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Measure time- and frequency-domain reconstruction fidelity.

    ``noise_psd`` may be supplied on the rFFT grid to obtain a PSD-weighted
    overlap.  Without it, the overlap is band-limited but unweighted.  The
    absolute overlap treats an overall sign as an extrinsic ambiguity.
    """
    target = np.atleast_2d(np.asarray(target, dtype=float))
    reconstruction = np.atleast_2d(np.asarray(reconstruction, dtype=float))
    if target.shape != reconstruction.shape:
        raise ValueError(
            "target and reconstruction must have identical shapes; got "
            f"{target.shape} and {reconstruction.shape}"
        )
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    target_fd = np.fft.rfft(target, axis=-1)
    reconstruction_fd = np.fft.rfft(reconstruction, axis=-1)
    frequencies = np.fft.rfftfreq(target.shape[-1], d=1.0 / sample_rate)
    low = 0.0 if flow is None else float(flow)
    high = frequencies[-1] if fmax is None else float(fmax)
    band = (frequencies >= low) & (frequencies <= high)
    if not np.any(band):
        raise ValueError("The requested frequency band contains no rFFT bins")

    if noise_psd is None:
        weights = np.ones_like(frequencies[band])
    else:
        noise_psd = np.asarray(noise_psd, dtype=float)
        if noise_psd.shape != frequencies.shape:
            raise ValueError(
                "noise_psd must match the rFFT grid shape "
                f"{frequencies.shape}; got {noise_psd.shape}"
            )
        if np.any(~np.isfinite(noise_psd[band])) or np.any(
            noise_psd[band] <= 0
        ):
            raise ValueError("noise_psd must be finite and positive in band")
        weights = 1.0 / noise_psd[band]

    target_band = target_fd[:, band]
    reconstruction_band = reconstruction_fd[:, band]
    inner = np.real(
        np.sum(
            np.conjugate(target_band) * reconstruction_band * weights[None, :],
            axis=-1,
        )
    )
    target_norm = np.sum(np.abs(target_band) ** 2 * weights[None, :], axis=-1)
    reconstruction_norm = np.sum(
        np.abs(reconstruction_band) ** 2 * weights[None, :], axis=-1
    )
    denominator = np.sqrt(target_norm * reconstruction_norm)
    overlap = np.divide(
        np.abs(inner),
        denominator,
        out=np.zeros_like(inner),
        where=denominator > 0,
    )
    overlap = np.clip(overlap, 0.0, 1.0)

    tiny = np.finfo(float).tiny
    log_target = np.log(np.maximum(np.abs(target_band), tiny))
    log_reconstruction = np.log(np.maximum(np.abs(reconstruction_band), tiny))
    peak_target = np.argmax(np.abs(target), axis=-1)
    peak_reconstruction = np.argmax(np.abs(reconstruction), axis=-1)

    return {
        "time_mse": np.mean((target - reconstruction) ** 2, axis=-1),
        "mismatch": 1.0 - overlap,
        "log_spectral_distance": np.sqrt(
            np.mean((log_target - log_reconstruction) ** 2, axis=-1)
        ),
        "peak_time_error_ms": (
            np.abs(peak_target - peak_reconstruction) / sample_rate * 1e3
        ),
    }


def summarize_fidelity(metrics: dict[str, np.ndarray]) -> dict[str, float]:
    """Return robust scalar summaries for reconstruction metrics."""
    summary: dict[str, float] = {}
    for name, values in metrics.items():
        values = np.asarray(values, dtype=float)
        summary[f"{name}_median"] = float(np.median(values))
        summary[f"{name}_p90"] = float(np.percentile(values, 90.0))
    return summary


def decoder_jacobian_singular_values(
    model_data: ModelData, z: np.ndarray
) -> np.ndarray:
    """Return decoder Jacobian singular values at one or more latent points."""
    z = jnp.atleast_2d(jnp.asarray(z))
    if z.shape[-1] != model_data.latent_dim:
        raise ValueError(
            f"Expected latent dimension {model_data.latent_dim}; got "
            f"{z.shape[-1]}"
        )
    model = _model(model_data)

    def decode_one(point):
        return model.apply(
            {"params": model_data.params}, point, method=model.generate
        )

    jacobians = jax.vmap(jax.jacrev(decode_one))(z)
    return np.asarray(jnp.linalg.svd(jacobians, compute_uv=False))


def summarize_decoder_geometry(
    model_data: ModelData,
    z: np.ndarray,
    *,
    rank_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Summarize local conditioning and encoder-decoder round-trip error."""
    z = np.atleast_2d(np.asarray(z, dtype=float))
    singular_values = decoder_jacobian_singular_values(model_data, z)
    largest = singular_values[:, 0]
    smallest = singular_values[:, -1]
    condition = np.divide(
        largest,
        smallest,
        out=np.full_like(largest, np.inf),
        where=smallest > 0,
    )
    threshold = rank_tolerance * largest[:, None]
    effective_rank = np.sum(singular_values > threshold, axis=-1)
    waveforms = generate(model_data, jnp.asarray(z))
    encoded = encode_mean(waveforms, model_data)
    roundtrip = np.sqrt(np.mean((np.asarray(encoded) - z) ** 2, axis=-1))
    return {
        "n_points": int(len(z)),
        "singular_value_median": np.median(singular_values, axis=0).tolist(),
        "condition_number_median": float(np.median(condition)),
        "condition_number_p90": float(np.percentile(condition, 90.0)),
        "effective_rank_min": int(np.min(effective_rank)),
        "roundtrip_rmse_median": float(np.median(roundtrip)),
        "roundtrip_rmse_p90": float(np.percentile(roundtrip, 90.0)),
    }


def decoder_collision_summary(
    model_data: ModelData,
    z: np.ndarray,
    *,
    minimum_latent_distance: float = 1.0,
    mismatch_threshold: float = 1e-3,
) -> dict[str, Any]:
    """Find distant latent samples that decode to nearly identical shapes."""
    z = np.atleast_2d(np.asarray(z, dtype=float))
    if len(z) < 2:
        raise ValueError("At least two latent samples are required")
    waveforms = np.asarray(generate(model_data, jnp.asarray(z)))
    waveform_norm = np.linalg.norm(waveforms, axis=-1, keepdims=True)
    normalized = np.divide(
        waveforms,
        waveform_norm,
        out=np.zeros_like(waveforms),
        where=waveform_norm > 0,
    )
    mismatch = 1.0 - np.clip(np.abs(normalized @ normalized.T), 0.0, 1.0)
    latent_distance = np.linalg.norm(z[:, None, :] - z[None, :, :], axis=-1)
    upper = np.triu(np.ones_like(mismatch, dtype=bool), k=1)
    eligible = upper & (latent_distance >= minimum_latent_distance)
    if not np.any(eligible):
        return {
            "n_points": int(len(z)),
            "eligible_pairs": 0,
            "collision_pairs": 0,
            "minimum_mismatch": None,
        }
    eligible_mismatch = mismatch[eligible]
    eligible_matrix = np.where(eligible, mismatch, np.inf)
    first, second = np.unravel_index(
        np.argmin(eligible_matrix), eligible_matrix.shape
    )
    return {
        "n_points": int(len(z)),
        "eligible_pairs": int(np.sum(eligible)),
        "collision_pairs": int(
            np.sum(eligible_mismatch <= mismatch_threshold)
        ),
        "minimum_mismatch": float(np.min(eligible_mismatch)),
        "mismatch_p01": float(np.percentile(eligible_mismatch, 1.0)),
        "minimum_mismatch_pair": [int(first), int(second)],
        "minimum_mismatch_latent_distance": float(
            latent_distance[first, second]
        ),
    }
