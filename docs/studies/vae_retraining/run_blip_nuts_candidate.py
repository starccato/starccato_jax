"""Run a multistart-MAP plus four-chain NUTS check for a blip VAE.

This script uses ``starccato_lvk``. Launch it from that repository's
environment while pointing ``PYTHONPATH`` at the local ``starccato_jax``
source. It intentionally does not run nested sampling.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from numpyro.diagnostics import effective_sample_size, gelman_rubin
from numpyro.infer import init_to_value

from starccato_jax import StarccatoVAE
from starccato_lvk.analysis.jim_likelihood import (
    build_transient_likelihood,
    find_multistart_map,
    run_numpyro_sampling,
)
from starccato_lvk.analysis.jim_waveform import StarccatoGlitchWaveform
from starccato_lvk.analysis.main import _clone_no_response_detector
from starccato_lvk.analysis.multidet_data_prep import (
    prepare_multi_detector_data,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--flow", type=float, default=300.0)
    parser.add_argument("--fmax", type=float, default=800.0)
    parser.add_argument("--num-starts", type=int, default=20)
    parser.add_argument("--num-warmup", type=int, default=800)
    parser.add_argument("--num-samples", type=int, default=800)
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--max-tree-depth", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260718)
    parser.add_argument(
        "--map-only",
        action="store_true",
        help="run the multistart mode screen without NUTS",
    )
    args = parser.parse_args()

    prepared = prepare_multi_detector_data(
        ["L1"],
        bundle_paths={"L1": str(args.bundle)},
        flow=args.flow,
        fmax=args.fmax,
    )
    reference = prepared.detector_data["L1"]
    waveform = StarccatoGlitchWaveform(
        model=StarccatoVAE(str(args.model_dir)),
        sample_rate=1.0 / reference.dt,
        strain_scale=1e-21,
        window=prepared.window,
    )
    detector = _clone_no_response_detector(prepared.detectors[0])
    detector.set_frequency_bounds(*prepared.detectors[0].frequency_bounds)
    likelihood = build_transient_likelihood(
        [detector],
        waveform,
        trigger_time=prepared.trigger_time,
        duration=prepared.duration,
        post_trigger_duration=prepared.post_trigger_duration,
    )

    map_result = find_multistart_map(
        likelihood,
        latent_names=waveform.latent_names,
        fixed_params={},
        rng_key=jax.random.PRNGKey(args.seed),
        latent_sigma=1.0,
        log_amp_sigma=5.0,
        num_starts=args.num_starts,
        log_amp_starts=(-2.0, 0.0, 2.0, 4.0, 6.0),
        maxiter=400,
        noise_scale_marginal=True,
    )
    best_z = np.asarray(
        [map_result.values[name] for name in waveform.latent_names]
    )
    ranked_attempts = sorted(
        map_result.attempts,
        key=lambda attempt: float(attempt["log_density"]),
        reverse=True,
    )
    mode_screen = []
    for attempt in ranked_attempts:
        attempt_z = np.asarray(
            [attempt["values"][name] for name in waveform.latent_names]
        )
        mode_screen.append(
            {
                "log_density": float(attempt["log_density"]),
                "delta_log_density": float(
                    map_result.log_density - attempt["log_density"]
                ),
                "latent_distance_from_best": float(
                    np.linalg.norm(attempt_z - best_z)
                ),
                "success": bool(attempt["success"]),
                "values": attempt["values"],
            }
        )

    map_diagnostics = {
        "model_dir": str(args.model_dir.resolve()),
        "bundle": str(args.bundle.resolve()),
        "initialization": f"{args.num_starts}_start_bounded_lbfgs_map",
        "map": {
            "values": map_result.values,
            "log_density": map_result.log_density,
            "mode_screen": mode_screen,
        },
    }
    if args.map_only:
        args.outdir.mkdir(parents=True, exist_ok=True)
        (args.outdir / "diagnostics.json").write_text(
            json.dumps(map_diagnostics, indent=2) + "\n"
        )
        print(json.dumps(map_diagnostics, indent=2))
        return

    initial = {
        name: jnp.asarray(value) for name, value in map_result.values.items()
    }
    started = time.time()
    result = run_numpyro_sampling(
        likelihood,
        latent_names=waveform.latent_names,
        fixed_params={},
        rng_key=jax.random.PRNGKey(args.seed + 1),
        latent_sigma=1.0,
        log_amp_sigma=5.0,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        dense_mass=True,
        target_accept_prob=args.target_accept,
        max_tree_depth=args.max_tree_depth,
        init_strategy=init_to_value(values=initial),
        noise_scale_marginal=True,
    )
    grouped = result.extra["samples_grouped"]
    energy = np.asarray(result.extra["energy"]).reshape(
        args.num_chains, args.num_samples
    )
    num_steps = np.asarray(result.extra["num_steps"])
    diagnostics = {
        **map_diagnostics,
        "sampler": "NUTS",
        "likelihood": "noise_scale_marginal",
        "runtime_seconds": float(time.time() - started),
        "divergences": int(np.sum(np.asarray(result.extra["diverging"]))),
        "mean_accept_prob": float(
            np.mean(np.asarray(result.extra["accept_prob"]))
        ),
        "min_ess": {
            name: float(np.nanmin(np.asarray(effective_sample_size(values))))
            for name, values in grouped.items()
        },
        "max_rhat": {
            name: float(np.nanmax(np.asarray(gelman_rubin(values))))
            for name, values in grouped.items()
        },
        "chain_medians": {
            name: np.median(np.asarray(values), axis=1).tolist()
            for name, values in grouped.items()
        },
        "ebfmi_by_chain": (
            np.mean(np.diff(energy, axis=1) ** 2, axis=1)
            / np.var(energy, axis=1, ddof=1)
        ).tolist(),
        "max_num_steps": int(np.max(num_steps)),
        "fraction_at_max_tree": float(
            np.mean(num_steps >= (2**args.max_tree_depth - 1))
        ),
        "num_warmup": args.num_warmup,
        "num_samples_per_chain": args.num_samples,
        "num_chains": args.num_chains,
        "target_accept_prob": args.target_accept,
        "max_tree_depth": args.max_tree_depth,
    }
    args.outdir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.outdir / "samples.npz",
        **{
            name: np.asarray(values) for name, values in result.samples.items()
        },
    )
    np.savez(
        args.outdir / "sample_stats.npz",
        **{
            name: np.asarray(result.extra[name])
            for name in (
                "diverging",
                "accept_prob",
                "num_steps",
                "energy",
                "potential_energy",
            )
        },
    )
    (args.outdir / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
