#!/usr/bin/env python
import os

import bilby
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from bilby.core.utils.random import seed

from starccato_jax.waveforms import StarccatoCCSNe

seed(123)
jax.config.update("jax_enable_x64", True)


def _to_numpy64(arr):
    """Convert JAX/array-like objects to float64 NumPy arrays."""
    if isinstance(arr, np.ndarray):
        return arr.astype(np.float64, copy=False)
    try:
        arr = jax.device_get(arr)
    except Exception:
        pass
    return np.asarray(arr, dtype=np.float64)

STARCCATO_MODEL = StarccatoCCSNe()
NUM_VAE_LATENTS = 32
LATENT_PARAMETER_NAMES = [f"z_{i}" for i in range(NUM_VAE_LATENTS)]
REFERENCE_DISTANCE_KPC = 10.0
REFERENCE_STRAIN_SCALE = 1e-21
SAMPLING_FREQUENCY = 4096.0
WAVEFORM_NUM_SAMPLES = int(
    jnp.asarray(STARCCATO_MODEL.generate(rng=jax.random.PRNGKey(0), n=1)[0]).shape[0]
)
DURATION = WAVEFORM_NUM_SAMPLES / SAMPLING_FREQUENCY

def starccato_supernova(
    time_array,
    luminosity_distance,
    *,
    model: StarccatoCCSNe,
    latent_parameter_names,
    reference_distance=REFERENCE_DISTANCE_KPC,
    strain_scale=REFERENCE_STRAIN_SCALE,
    **parameters,
):
    """
    Generate time-domain supernova waveforms using the Starccato VAE.

    Bilby passes the time array along with intrinsic/extrinsic parameters.
    This function builds the latent vector from the provided parameters,
    decodes a waveform with the Starccato VAE, rescales it to the requested
    luminosity distance, and returns plus/cross polarisations.
    """
    latents = jnp.array(
        [parameters.get(name, 0.0) for name in latent_parameter_names],
        dtype=jnp.float32,
    ).reshape(1, -1)

    waveform = model.generate(z=latents)[0].astype(jnp.float64)
    if waveform.size != time_array.size:
        raise ValueError(
            f"Starccato waveform length ({waveform.size}) does not match "
            f"bilby time array ({time_array.size}). "
            "Adjust `duration` or `sampling_frequency` to match the model."
        )

    distance_scale = reference_distance / jnp.asarray(luminosity_distance, dtype=jnp.float64)
    intrinsic_amp = jnp.asarray(parameters.get("intrinsic_amplitude", 1.0), dtype=jnp.float64)
    scaled_waveform = intrinsic_amp * strain_scale * distance_scale * waveform

    plus = _to_numpy64(scaled_waveform)
    return {"plus": plus, "cross": plus}


def plot_waveform_and_psd_comparison(
    ifos,
    time_array,
    waveform,
    outpath,
    title_prefix="",
    log_f=True,
):
    """
    Plot time-domain waveform and PSD comparisons for each interferometer.
    """
    waveform = _to_numpy64(waveform)
    time_array = _to_numpy64(time_array)

    fig, axes = plt.subplots(1, len(ifos) + 1, figsize=(6 * (len(ifos) + 1), 4))

    ax0 = axes[0]
    ax0.plot(time_array, waveform, color="tab:orange")
    ax0.set_title(f"{title_prefix} waveform (time domain)")
    ax0.set_xlabel("Time [s]")
    ax0.set_ylabel("Strain")
    ax0.grid(True, alpha=0.3)

    for i, ifo in enumerate(ifos, start=1):
        freqs = _to_numpy64(ifo.frequency_array)
        psd = _to_numpy64(ifo.power_spectral_density_array)

        hf = np.fft.rfft(waveform)
        freq_sig = np.fft.rfftfreq(len(time_array), d=1.0 / ifo.strain_data.sampling_frequency)
        sig_psd = (np.abs(hf) ** 2) / len(time_array)

        data_td = _to_numpy64(ifo.strain_data.time_domain_strain)
        hf_data = np.fft.rfft(data_td)
        data_psd = (np.abs(hf_data) ** 2) / len(data_td)

        ax = axes[i]
        if log_f:
            ax.loglog(freqs, psd, color="black", label="Noise PSD")
            ax.loglog(freq_sig, sig_psd, color="tab:orange", label="Signal PSD")
            ax.loglog(freq_sig, data_psd, color="tab:blue", alpha=0.7, label="Data PSD")
        else:
            ax.plot(freqs, psd, color="black", label="Noise PSD")
            ax.plot(freq_sig, sig_psd, color="tab:orange", label="Signal PSD")
            ax.plot(freq_sig, data_psd, color="tab:blue", alpha=0.7, label="Data PSD")

        ax.set_title(f"{ifo.name} PSD comparison")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Power spectral density [1/Hz]")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


# Set the duration and sampling frequency of the data segment that we're going
# to inject the signal into. These are fixed by the Starccato waveform length.
duration = DURATION
sampling_frequency = SAMPLING_FREQUENCY

# Specify the output directory and the name of the simulation.
outdir = "outdir"
label = "supernova"
bilby.core.utils.setup_logger(outdir=outdir, label=label)
os.makedirs(outdir, exist_ok=True)


# We are going to inject a supernova waveform.  We first establish a dictionary
# of parameters that includes all of the different waveform parameters that feed
# the Starccato generator.

injection_parameters = dict(
    luminosity_distance=7.0,
    ra=4.6499,
    dec=-0.5063,
    geocent_time=1126259642.413,
    psi=2.659,
    intrinsic_amplitude=1.0,
)
for name in LATENT_PARAMETER_NAMES:
    injection_parameters[name] = 0.0

# Create the waveform_generator using the Starccato VAE source function
waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    time_domain_source_model=starccato_supernova,
    parameters=injection_parameters,
    parameter_conversion=lambda parameters: (parameters, list()),
    waveform_arguments=dict(
        model=STARCCATO_MODEL,
        latent_parameter_names=LATENT_PARAMETER_NAMES,
        reference_distance=REFERENCE_DISTANCE_KPC,
        strain_scale=REFERENCE_STRAIN_SCALE,
    ),
)

# Set up interferometers.  In this case we'll use three interferometers
# (LIGO-Hanford (H1), LIGO-Livingston (L1), and Virgo (V1)).  These default to
# their design sensitivity
ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
ifos.set_strain_data_from_power_spectral_densities(
    sampling_frequency=sampling_frequency,
    duration=duration,
    start_time=injection_parameters["geocent_time"] - duration / 2,
)

preview_time = (
    np.arange(WAVEFORM_NUM_SAMPLES, dtype=np.float64) - WAVEFORM_NUM_SAMPLES / 2
) / sampling_frequency
preview_waveform = starccato_supernova(
    time_array=preview_time,
    luminosity_distance=injection_parameters["luminosity_distance"],
    model=STARCCATO_MODEL,
    latent_parameter_names=LATENT_PARAMETER_NAMES,
    reference_distance=REFERENCE_DISTANCE_KPC,
    strain_scale=REFERENCE_STRAIN_SCALE,
    **{k: v for k, v in injection_parameters.items() if k != "luminosity_distance"},
)["plus"]
preview_waveform = _to_numpy64(preview_waveform)
plot_waveform_and_psd_comparison(
    ifos=ifos,
    time_array=preview_time,
    waveform=preview_waveform,
    outpath=os.path.join(outdir, "initial_signal_psd_comparison.png"),
    title_prefix="Initial",
)
ifos.inject_signal(
    waveform_generator=waveform_generator,
    parameters=injection_parameters,
    raise_error=False,
)
plot_waveform_and_psd_comparison(
    ifos=ifos,
    time_array=preview_time,
    waveform=preview_waveform,
    outpath=os.path.join(outdir, "after_injection_psd_comparison.png"),
    title_prefix="After injection",
)

search_waveform_generator = bilby.gw.waveform_generator.WaveformGenerator(
    duration=duration,
    sampling_frequency=sampling_frequency,
    time_domain_source_model=starccato_supernova,
    parameter_conversion=lambda parameters: (parameters, list()),
    waveform_arguments=dict(
        model=STARCCATO_MODEL,
        latent_parameter_names=LATENT_PARAMETER_NAMES,
        reference_distance=REFERENCE_DISTANCE_KPC,
        strain_scale=REFERENCE_STRAIN_SCALE,
    ),
)

# Set up prior
priors = bilby.core.prior.PriorDict()
for key in ["psi", "geocent_time"]:
    priors[key] = injection_parameters[key]
priors["luminosity_distance"] = bilby.core.prior.Uniform(
    2, 20, "luminosity_distance", unit="$kpc$"
)
priors["intrinsic_amplitude"] = bilby.core.prior.Uniform(
    0.5, 1.5, "intrinsic_amplitude"
)
for name in LATENT_PARAMETER_NAMES:
    priors[name] = bilby.core.prior.Normal(mu=0.0, sigma=1.0, name=name)
priors["ra"] = bilby.core.prior.Uniform(
    minimum=0, maximum=2 * np.pi, name="ra", boundary="periodic"
)
priors["dec"] = bilby.core.prior.Sine(name="dec")
priors["geocent_time"] = bilby.core.prior.Uniform(
    injection_parameters["geocent_time"] - 1,
    injection_parameters["geocent_time"] + 1,
    "geocent_time",
    unit="$s$",
)

# Initialise the likelihood by passing in the interferometer data (IFOs) and
# the waveoform generator
likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
    interferometers=ifos, waveform_generator=search_waveform_generator
)

# Run sampler.
result = bilby.run_sampler(
    likelihood=likelihood,
    priors=priors,
    sampler="numpyro",
    sampler_name="NUTS",
    num_warmup=500,
    num_samples=1000,
    outdir=outdir,
    label=label,
)

# Plot the results of the parameter estimation.

best_idx = result.posterior.log_likelihood.idxmax()
best_sample = result.posterior.loc[best_idx].to_dict()
posterior_waveform_td = search_waveform_generator.time_domain_strain(best_sample)["plus"]
posterior_time = _to_numpy64(search_waveform_generator.time_array)
plot_waveform_and_psd_comparison(
    ifos=ifos,
    time_array=posterior_time,
    waveform=posterior_waveform_td,
    outpath=os.path.join(outdir, "posterior_bestfit_psd_comparison.png"),
    title_prefix="Posterior (best-fit)",
)

# plot the injection signal and the data and the posterior waveforms
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
waveform = waveform_generator.frequency_domain_strain(
    injection_parameters, f_lower=20.0
)
freqs = waveform_generator.frequency_array
ax.loglog(
    freqs,
    np.abs(waveform["plus"]),
    label="Injection plus polarisation",
)
ax.loglog(
    freqs,
    np.abs(waveform["cross"]),
    label="Injection cross polarisation",
)
for sample in result.posterior.sample(100):
    waveform = search_waveform_generator.frequency_domain_strain(
        sample, f_lower=20.0
    )
    ax.loglog(
        freqs,
        np.abs(waveform["plus"]),
        color="C1",
        alpha=0.1,
    )
    ax.loglog(
        freqs,
        np.abs(waveform["cross"]),
        color="C2",
        alpha=0.1,
    )
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("Strain")
ax.legend()
plt.savefig(f"{outdir}/{label}_waveforms.png")
