#!/usr/bin/env python
"""

Tutorial script demonstrating how to analyse a supernova signal with the
Starccato VAE waveform model using the JIM inference framework.

This example:
  * Implements a JIM-compatible waveform wrapper around the Starccato CCSNe VAE
  * Generates a synthetic injection seen by the H1 and L1 detectors
  * Configures a simple uniform prior over the VAE latent space and nuisance
    parameters
  * Evaluates the JIM log-posterior at the injected parameters and shows how
    to launch a short FlowMC sampling run

The goal is to provide a compact reference for integrating Starccato models
with JIM's NumPyro/JAX-native tooling.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import matplotlib.pyplot as plt
from numpyro.contrib.nested_sampling import NestedSampler
from jax.scipy.special import logsumexp

jax.config.update("jax_enable_x64", True)
numpyro.enable_x64()

from astropy.time import Time

from jimgw.waveform import Waveform
from jimgw.detector import H1, L1, GroundBased2G
from jimgw.likelihood import TransientLikelihoodFD
from starccato_jax.waveforms import StarccatoCCSNe


def _to_numpy64(x):
    if isinstance(x, np.ndarray):
        return x.astype(np.float64, copy=False)
    try:
        x = jax.device_get(x)
    except Exception:
        pass
    return np.asarray(x, dtype=np.float64)


# ---------------------------------------------------------------------------
# Starccato waveform adapter
# ---------------------------------------------------------------------------


@dataclass
class StarccatoJimWaveform(Waveform):
    """
    Turn the Starccato CCSNe VAE into a JIM-compatible frequency-domain source.

    The class keeps all computations in JAX so that gradients and automatic
    batching continue to work seamlessly inside JIM/FlowMC.
    """

    model: StarccatoCCSNe
    sample_rate: float = 4096.0
    strain_scale: float = 1e-21
    reference_distance: float = 10.0

    def __post_init__(self):
        self.latent_dim = self.model.latent_dim
        self.dt = 1.0 / self.sample_rate
        test_waveform = self.model.generate(
            z=jnp.zeros((1, self.latent_dim)), rng=jax.random.PRNGKey(0)
        )[0]
        self.num_samples = int(test_waveform.shape[0])
        self.latent_names = [f"z_{i}" for i in range(self.latent_dim)]

    def _latent_vector(self, params: dict) -> jnp.ndarray:
        return jnp.array([params[name] for name in self.latent_names], dtype=jnp.float32)[
            None, :
        ]

    def _time_domain_waveform(self, params: dict) -> jnp.ndarray:
        latents = self._latent_vector(params)
        waveform = self.model.generate(z=latents)[0].astype(jnp.float64)

        distance = params.get("luminosity_distance", self.reference_distance)
        log_amp = params.get("log_amp", 0.0)
        intrinsic_amp = jnp.exp(log_amp)
        amplitude_scale = (
            intrinsic_amp * self.strain_scale * (self.reference_distance / distance)
        )
        return waveform * amplitude_scale

    def __call__(self, frequency: jnp.ndarray, params: dict) -> dict[str, jnp.ndarray]:
        waveform_td = self._time_domain_waveform(params)

        expected_shape = (self.num_samples // 2 + 1,)
        if frequency.shape != expected_shape:
            raise ValueError(
                f"Frequency grid mismatch: got shape {frequency.shape}, expected {expected_shape}."
            )

        waveform_fd = jnp.fft.rfft(waveform_td) * self.dt
        return {"p": waveform_fd, "c": waveform_fd}


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def build_synthetic_data(
    key: jax.Array,
    detectors: list[GroundBased2G],
    waveform: StarccatoJimWaveform,
    injection_params: dict,
    duration: float,
    psd_level: float = 1e-46,
) -> jnp.ndarray:
    """
    Populate each detector with a noisy realization of the injected waveform.
    """
    freqs = jnp.fft.rfftfreq(waveform.num_samples, 1.0 / waveform.sample_rate)
    df = freqs[1] - freqs[0]
    psd = jnp.ones_like(freqs) * psd_level

    h_sky = waveform(freqs, injection_params)
    epoch = duration / 2.0
    align_time = jnp.exp(-1j * 2.0 * jnp.pi * freqs * (epoch + injection_params["t_c"]))

    keys = jax.random.split(key, len(detectors))
    for subkey, detector in zip(keys, detectors, strict=True):
        noise_key, noise_key_im = jax.random.split(subkey)
        var = psd / (4.0 * df)
        noise_real = jax.random.normal(noise_key, shape=freqs.shape) * jnp.sqrt(var)
        noise_imag = jax.random.normal(noise_key_im, shape=freqs.shape) * jnp.sqrt(var)
        signal = detector.fd_response(freqs, h_sky, injection_params) * align_time

        detector.frequencies = freqs
        detector.psd = psd
        detector.data = signal + noise_real + 1j * noise_imag

    return freqs


def plot_waveform_and_psd_comparison(
    ifos,
    time_array,
    waveform,
    sample_rate,
    outpath,
    title_prefix="",
    log_f=True,
    include_data=True,
):
    waveform = _to_numpy64(waveform)
    time_array = _to_numpy64(time_array)

    fig, axes = plt.subplots(1, len(ifos) + 1, figsize=(6 * (len(ifos) + 1), 4))

    ax0 = axes[0]
    ax0.plot(time_array, waveform, color="tab:orange")
    ax0.set_title(f"{title_prefix} waveform (time domain)")
    ax0.set_xlabel("Time [s]")
    ax0.set_ylabel("Strain")
    ax0.grid(True, alpha=0.3)

    signal_fft = np.fft.rfft(waveform)
    freq_sig = np.fft.rfftfreq(len(time_array), d=1.0 / sample_rate)
    sig_psd = (np.abs(signal_fft) ** 2) / len(time_array)

    for idx, ifo in enumerate(ifos, start=1):
        if hasattr(ifo, "frequencies"):
            freqs = _to_numpy64(ifo.frequencies)
        elif hasattr(ifo, "frequency_array"):
            freqs = _to_numpy64(ifo.frequency_array)
        else:
            raise AttributeError("Detector object must have 'frequencies' or 'frequency_array'.")

        if hasattr(ifo, "psd"):
            noise_psd = _to_numpy64(ifo.psd)
        elif hasattr(ifo, "power_spectral_density_array"):
            noise_psd = _to_numpy64(ifo.power_spectral_density_array)
        else:
            raise AttributeError("Detector object must have 'psd' or 'power_spectral_density_array'.")

        ax = axes[idx]
        if log_f:
            ax.loglog(freqs, noise_psd, color="black", label="Noise PSD")
            ax.loglog(freq_sig, sig_psd, color="tab:orange", label="Signal PSD")
        else:
            ax.plot(freqs, noise_psd, color="black", label="Noise PSD")
            ax.plot(freq_sig, sig_psd, color="tab:orange", label="Signal PSD")

        data_psd = None
        if include_data and hasattr(ifo, "data") and getattr(ifo, "data") is not None:
            data_fd = _to_numpy64(ifo.data)
            data_psd = (np.abs(data_fd) ** 2)
            if log_f:
                ax.loglog(freqs, data_psd, color="tab:blue", alpha=0.7, label="Data PSD")
            else:
                ax.plot(freqs, data_psd, color="tab:blue", alpha=0.7, label="Data PSD")

        ax.set_title(f"{getattr(ifo, 'name', f'IFO {idx}') } PSD comparison")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Power spectral density [arb]")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def named_parameters(naming: list[str], params: dict[str, float]) -> jnp.ndarray:
    return jnp.array([params[name] for name in naming], dtype=jnp.float64)


# ---------------------------------------------------------------------------
# NumPyro sampling setup
# ---------------------------------------------------------------------------


def run_numpyro_sampling(
    likelihood: TransientLikelihoodFD,
    latent_names: list[str],
    fixed_params: dict[str, float],
    rng_key: jax.Array,
    *,
    latent_sigma: float | jnp.ndarray = 1.0,
    log_amp_sigma: float = 1.0,
    num_warmup: int = 500,
    num_samples: int = 1000,
    num_chains: int = 1,
    dense_mass: bool = True,
) -> MCMC:
    """
    Launch a NumPyro NUTS run using Normal priors for the latent space and
    fixed extrinsic parameters provided through ``fixed_params``.
    """

    latent_sigma = np.asarray(latent_sigma, dtype=np.float64)
    if latent_sigma.ndim == 0:
        latent_sigma = np.full((len(latent_names),), float(latent_sigma))
    elif latent_sigma.shape != (len(latent_names),):
        raise ValueError("latent_sigma must be a scalar or have length equal to latent_names")

    log_amp_sigma = float(log_amp_sigma)

    def model():
        params = {}
        for idx, name in enumerate(latent_names):
            params[name] = numpyro.sample(
                name, dist.Normal(0.0, latent_sigma[idx])
            )
        params["log_amp"] = numpyro.sample("log_amp", dist.Normal(0.0, log_amp_sigma))
        params.update(fixed_params)
        log_like = likelihood.evaluate(params, None)
        numpyro.factor("log_likelihood", log_like)

    kernel = NUTS(model, dense_mass=dense_mass)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=True,
    )
    mcmc.run(rng_key)
    return mcmc


def run_nested_sampling(
    likelihood: TransientLikelihoodFD,
    latent_names: list[str],
    fixed_params: dict[str, float],
    rng_key: jax.Array,
    *,
    latent_sigma: float | jnp.ndarray = 1.0,
    log_amp_sigma: float = 1.0,
    num_live_points: int = 500,
    max_samples: int = 20000,
    num_posterior_samples: int = 2000,
    verbose: bool = False,
):
    latent_sigma = np.asarray(latent_sigma, dtype=np.float64)
    if latent_sigma.ndim == 0:
        latent_sigma = np.full((len(latent_names),), float(latent_sigma))
    elif latent_sigma.shape != (len(latent_names),):
        raise ValueError("latent_sigma must be a scalar or have length equal to latent_names")

    log_amp_sigma = float(log_amp_sigma)

    def model():
        params = {}
        for idx, name in enumerate(latent_names):
            params[name] = numpyro.sample(name, dist.Normal(0.0, latent_sigma[idx]))
        params["log_amp"] = numpyro.sample("log_amp", dist.Normal(0.0, log_amp_sigma))
        params.update(fixed_params)
        log_like = likelihood.evaluate(params, None)
        numpyro.factor("log_likelihood", log_like)

    ns = NestedSampler(
        model,
        constructor_kwargs=dict(
            num_live_points=num_live_points,
            gradient_guided=True,
            verbose=verbose,
        ),
        termination_kwargs=dict(dlogZ=0.001, ess=500, max_samples=max_samples),
    )

    run_key, sample_key = jax.random.split(rng_key)

    t0 = time.process_time()
    ns.run(run_key)
    runtime = time.process_time() - t0
    print(f"Nested sampling completed in {runtime:.2f} seconds.")
    ns.print_summary()

    weighted_samples, log_weights = ns.get_weighted_samples()
    log_weights = np.asarray(log_weights)
    weights = np.exp(log_weights - logsumexp(log_weights))

    mean_params = {}
    for name in weighted_samples:
        values = np.asarray(weighted_samples[name])
        mean_params[name] = float(np.sum(values * weights))

    try:
        logZ = float(ns.evidence)
        logZ_err = float(ns.evidence_error)
    except AttributeError:
        logZ = None
        logZ_err = None

    posterior_samples = ns.get_samples(
        sample_key,
        num_posterior_samples,
        group_by_chain=False,
    )
    samples = {name: np.asarray(posterior_samples[name]) for name in posterior_samples}

    return {
        "samples": samples,
        "weighted_samples": {name: np.asarray(vals) for name, vals in weighted_samples.items()},
        "log_weights": log_weights,
        "weights": weights,
        "mean_params": mean_params,
        "logZ": logZ,
        "logZ_err": logZ_err,
        "runtime": runtime,
    }


# ---------------------------------------------------------------------------
# Main demonstration
# ---------------------------------------------------------------------------


def main():
    os.makedirs("jim_outdir", exist_ok=True)

    model = StarccatoCCSNe()
    waveform = StarccatoJimWaveform(model=model)

    detectors = [H1, L1]
    psd_level = 1e-46

    latent_names = waveform.latent_names
    true_params = {name: 0.0 for name in latent_names}
    true_params.update(
        {
            "log_amp": 0.0,
            "t_c": 0.0,
            "ra": 4.6499,
            "dec": -0.5063,
            "psi": 2.659,
            "luminosity_distance": 7.0,
        }
    )

    trigger_time = 1126259642.413
    duration = waveform.num_samples / waveform.sample_rate
    gmst = float(
        Time(trigger_time, format="gps").sidereal_time("apparent", "greenwich").rad
    )
    true_params["gmst"] = gmst
    time_axis = (
        np.arange(waveform.num_samples, dtype=np.float64) - waveform.num_samples / 2
    ) / waveform.sample_rate
    injection_waveform_td = _to_numpy64(waveform._time_domain_waveform(true_params))
    initial_freqs = np.fft.rfftfreq(waveform.num_samples, 1.0 / waveform.sample_rate)
    initial_psd = np.ones_like(initial_freqs) * psd_level
    initial_ifos = [
        SimpleNamespace(name=det.name, frequencies=initial_freqs, psd=initial_psd, data=None)
        for det in detectors
    ]
    plot_waveform_and_psd_comparison(
        initial_ifos,
        time_axis,
        injection_waveform_td,
        waveform.sample_rate,
        outpath=os.path.join("jim_outdir", "initial_signal_psd_comparison.png"),
        title_prefix="Initial",
        include_data=False,
    )
    freqs = build_synthetic_data(
        jax.random.PRNGKey(1234), detectors, waveform, true_params, duration, psd_level=psd_level
    )
    plot_waveform_and_psd_comparison(
        detectors,
        time_axis,
        injection_waveform_td,
        waveform.sample_rate,
        outpath=os.path.join("jim_outdir", "after_injection_psd_comparison.png"),
        title_prefix="After injection",
        include_data=True,
    )

    latent_sigma = np.ones(len(latent_names), dtype=float)
    log_amp_sigma = 1.0

    fixed_params = {
        key: true_params[key]
        for key in ("t_c", "ra", "dec", "psi", "luminosity_distance", "gmst")
    }

    likelihood = TransientLikelihoodFD(
        detectors=detectors,
        waveform=waveform,
        trigger_time=trigger_time,
        duration=float(duration),
        post_trigger_duration=float(duration / 2.0),
    )

    sample_names = latent_names + ["log_amp"]
    theta_true = named_parameters(sample_names, true_params)
    log_like = likelihood.evaluate(true_params, None)
    log_prior = sum(
        dist.Normal(0.0, float(latent_sigma[idx])).log_prob(theta_true[idx])
        for idx in range(len(latent_names))
    ) + dist.Normal(0.0, log_amp_sigma).log_prob(theta_true[-1])
    log_post = log_like + log_prior
    print(f"Posterior at injected parameters: {float(log_post):.3f}")

    nested_results = run_nested_sampling(
        likelihood,
        latent_names=latent_names,
        fixed_params=fixed_params,
        rng_key=jax.random.PRNGKey(1337),
        latent_sigma=latent_sigma,
        log_amp_sigma=log_amp_sigma,
        num_live_points=500,
        max_samples=20000,
        verbose=False,
    )
    np.savez(
        os.path.join("jim_outdir", "nested_samples.npz"),
        **{name: np.asarray(val) for name, val in nested_results["samples"].items()},
        log_weights=nested_results["log_weights"],
        weights=nested_results["weights"],
    )
    nested_params_full = {**fixed_params, **nested_results["mean_params"]}
    nested_waveform_td = _to_numpy64(waveform._time_domain_waveform(nested_params_full))
    plot_waveform_and_psd_comparison(
        detectors,
        time_axis,
        nested_waveform_td,
        waveform.sample_rate,
        outpath=os.path.join("jim_outdir", "posterior_nested_psd_comparison.png"),
        title_prefix="Posterior (nested mean)",
        include_data=True,
    )

    mcmc = run_numpyro_sampling(
        likelihood,
        latent_names=latent_names,
        fixed_params=fixed_params,
        rng_key=jax.random.PRNGKey(2025),
        latent_sigma=latent_sigma,
        log_amp_sigma=log_amp_sigma,
        num_warmup=500,
        num_samples=1000,
        num_chains=1,
    )
    mcmc.print_summary()

    samples = mcmc.get_samples()
    np.savez(
        os.path.join("jim_outdir", "numpyro_samples.npz"),
        **{name: np.asarray(value) for name, value in samples.items()},
    )
    print(f"Stored NumPyro samples in jim_outdir/numpyro_samples.npz")

    posterior_mean_params = {name: float(np.mean(samples[name])) for name in latent_names}
    posterior_mean_params["log_amp"] = float(np.mean(samples["log_amp"]))
    posterior_params_full = {**fixed_params, **posterior_mean_params}
    posterior_waveform_td = _to_numpy64(waveform._time_domain_waveform(posterior_params_full))
    plot_waveform_and_psd_comparison(
        detectors,
        time_axis,
        posterior_waveform_td,
        waveform.sample_rate,
        outpath=os.path.join("jim_outdir", "posterior_nuts_psd_comparison.png"),
        title_prefix="Posterior (NUTS mean)",
        include_data=True,
    )


if __name__ == "__main__":
    main()
