from __future__ import annotations

from gwpy.frequencyseries import FrequencySeries
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram, sosfilt
from scipy.signal.windows import tukey
from gwpy.timeseries import TimeSeries
from dataclasses import dataclass

import contextlib, io, logging
import shutil

import arviz as az
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, autoguide, init_to_value
from starccato_jax.waveforms import StarccatoCCSNe, StarccatoBlip
from morphZ import evidence as morph_evidence
from pathlib import Path
from typing import Optional

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent.resolve()
OUTROOT = HERE / "out"
OUTROOT.mkdir(parents=True, exist_ok=True)

DEFAULT_TRIGGER_TIME = 1187721217  # quiet segment
DEFAULT_DETECTOR = "H1"
FS = SAMPLE_RATE = 4096.0
BAND = (100.0, 1024.0)
DURATION = 32.0
N = ANALYSIS_LEN = 512
DEFAULT_SNR_RANGE = (5.0, 100.0)
ANALYSIS_DURATION = ANALYSIS_LEN / SAMPLE_RATE
WINDOW = jnp.array(tukey(ANALYSIS_LEN, 0.25))
U = jnp.mean(WINDOW ** 2)
Neff = N * U
C = jnp.sqrt(FS * Neff / 2.0)
WINDOW_NP = np.asarray(WINDOW)

# make a mask for the band
FREQ = np.fft.rfftfreq(ANALYSIS_LEN, 1.0 / FS)
FMASK = np.array((FREQ >= BAND[0]) & (FREQ <= BAND[1]))
FMASK_JAX = jnp.array(FMASK, dtype=bool)
_WINDOWED_RFFT = jax.jit(lambda wf: (jnp.fft.rfft(wf * WINDOW) / C)[FMASK_JAX])

def _psd_consistent(series: np.ndarray):
    """Compute PSD using the same window/C normalization as the likelihood.
    Supports series shape (..., N); returns Pxx with same leading dims.
    """
    series = np.asarray(series)
    X = np.fft.rfft(series * WINDOW_NP, axis=-1) / float(C)
    Pxx = np.abs(X) ** 2 * 2.0  # units 1/Hz
    return FREQ[FMASK], Pxx[..., FMASK]

def band_mask(f: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    return (f >= band[0]) & (f <= band[1])

N_LATENTS = 32
REFERENCE_DIST = 10.0
# Scale the waveform generator to roughly match observed strain levels.
# With AMP_LOGMEAN near -2.5 (median ~8e-2), a scale of 2e-20 yields
# typical strains on the order of 1e-21–1e-20 after the LogNormal draw,
# keeping the prior broad.
REFERENCE_SCALE = 1e-20
BASE_WF_LEN = len(StarccatoCCSNe().generate(z=np.zeros((1, N_LATENTS)))[0])
# Center amplitude prior nearer to injection amplitude and widen it a bit more.
AMP_LOGMEAN = -2.0
AMP_LOGSIGMA = 2.0

STARCCATO_SIGNAL = StarccatoCCSNe()
STARCCATO_GLITCH = StarccatoBlip()

INJECT_KIND = "glitch"  # default, overridden by main args
DEFAULT_CACHE_FILE = HERE / "cache" / "H-H1_GWOSC_O2_4KHZ_R1-1187721216-4096.hdf5"


@dataclass
class WhittleContext:
    y_obs: jnp.ndarray
    sigma_comp: jnp.ndarray
    PSD_k: jnp.ndarray
    f_k: jnp.ndarray
    Pxx_signal: np.ndarray
    Pxx_data: np.ndarray
    lnz_noise: float


def fetch_strain(trigger_time, duration, cache_file: Path, sample_rate: float = SAMPLE_RATE):
    cache_file = Path(cache_file).expanduser().resolve()
    strain = TimeSeries.read(cache_file, format="hdf5.gwosc")
    start = float(strain.t0.value)
    end = start + float(strain.duration.value)
    if trigger_time < start or trigger_time + duration > end:
        raise RuntimeError(
            f"Requested GPS window [{trigger_time}, {trigger_time + duration}] "
            f"lies outside cache range [{start}, {end}] from {cache_file}"
        )
    strain = strain.crop(trigger_time, trigger_time + duration)
    if sample_rate and float(strain.sample_rate.value) != float(sample_rate):
        strain = strain.resample(float(sample_rate))
    return strain


def inner_product(x: np.ndarray, y: np.ndarray, psd_vals: np.ndarray, fs: float) -> float:
    """Compute the noise-weighted inner product between two time series."""
    win = WINDOW_NP
    Xk = (np.fft.rfft(x * win) / float(C))[FMASK]
    Yk = (np.fft.rfft(y * win) / float(C))[FMASK]
    Sk = psd_vals[FMASK]
    integrand = (np.conj(Xk) * Yk) / Sk
    return 4.0 * np.real(np.sum(integrand)) / fs


def snr_from_inner_product(x: np.ndarray, psd_vals: np.ndarray, fs: float = FS) -> float:
    """Compute the optimal SNR of a time series given the PSD."""
    return float(np.sqrt(inner_product(x, x, psd_vals, fs)))


def snr_from_excess_power(x: np.ndarray, psd_vals: np.ndarray, fs=FS) -> float:
    """Compute the excess-power SNR of a time series given the PSD."""
    Xf = (np.abs(np.fft.rfft(x * WINDOW_NP) / float(C)))[FMASK]
    Sf = psd_vals[FMASK]
    return np.abs(float(np.sqrt(np.sum(Xf ** 2 / Sf))))


def inject_signal_into_data(analysis_segment: TimeSeries, psd: FrequencySeries, kind: str, rng_key):
    """Draw injection from the prior (amp, z) and add to data."""
    model = STARCCATO_SIGNAL if kind == "signal" else STARCCATO_GLITCH
    rng_amp, rng_z = jax.random.split(rng_key)
    amp = float(dist.LogNormal(AMP_LOGMEAN, AMP_LOGSIGMA).sample(rng_amp))
    z = np.array(dist.Normal(0, 1).sample(rng_z, (N_LATENTS,)), dtype=np.float64)

    wf = model.generate(z=z[None, :])[0]
    wf = np.array(wf, dtype=np.float64, copy=True)
    injection = wf * amp * REFERENCE_SCALE * (REFERENCE_DIST / 7.0)

    data = np.asarray(analysis_segment.value)
    # center-crop/pad injection to match analysis segment length
    if len(injection) > len(data):
        injection = injection[-len(data):]
    elif len(injection) < len(data):
        injection = np.pad(injection, (len(data) - len(injection), 0))

    analysis_segment = TimeSeries(data + injection, sample_rate=analysis_segment.sample_rate)
    return analysis_segment, injection


def plot_data(
        ctx: WhittleContext,
        inj_type: str,
        injection_snr: Optional[float] = None,
        posterior_signal: Optional[np.ndarray] = None,
        posterior_blip: Optional[np.ndarray] = None,
        posterior_signal_label: str = "Signal",
        posterior_blip_label: str = "Glitch",
        logBF: Optional[float] = None,
        status_text: Optional[str] = None,
        outpath: Optional[Path] = None,
):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(ctx.f_k, ctx.PSD_k, color="k", lw=1.2, label="Noise PSD (Welch)")
    ax.loglog(ctx.f_k, ctx.Pxx_data, color="0.6", alpha=0.4, label="Data periodogram")
    if inj_type != 'noise':
        ax.loglog(ctx.f_k, ctx.Pxx_signal, color="tab:orange", label="Injection")

    def _plot_post(label, color, series):
        if series is None or len(series) == 0:
            return
        f_p, Pxx_masked = _psd_consistent(series)  # series expected shape (draws, N)
        med = np.median(Pxx_masked, axis=0)

        for qtls in [[10, 90], [20, 80]]:
            lo = np.percentile(Pxx_masked, qtls[0], axis=0)
            hi = np.percentile(Pxx_masked, qtls[1], axis=0)
            ax.fill_between(f_p, lo, hi, color=color, alpha=0.2)
        ax.loglog(f_p, med, color=color, lw=1.1, label=label)

    if posterior_signal is not None and len(posterior_signal) > 0:
        _plot_post(posterior_signal_label, "tab:red", posterior_signal)
    if posterior_blip is not None and len(posterior_blip) > 0:
        _plot_post(posterior_blip_label, "tab:green", posterior_blip)

    ax.set_xlim(*BAND)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [1/Hz]")
    title_parts = []
    if injection_snr is not None:
        title_parts.append(f"SNR≈{injection_snr:.1f}")
    if logBF is not None and not np.isnan(logBF):
        title_parts.append(f"logBF≈{logBF:.1f}")
    title_parts.append(f"inj={inj_type}")
    if status_text:
        title_parts.append(status_text)
    if title_parts:
        ax.set_title(" | ".join(title_parts))
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def build_whittle_context(data_ts: TimeSeries, injected_wf: np.ndarray, psd: FrequencySeries, fs=FS) -> WhittleContext:
    win = WINDOW_NP
    data_ts_np = np.asarray(data_ts.value, dtype=np.float64) * win
    injected_wf_np = np.asarray(injected_wf, dtype=np.float64) * win

    Pxx_psd = np.asarray(psd.value, dtype=np.float64)

    # y_obs and sigma_comp use the same window and C normalisation as rfft_model_waveform_scaled,
    # so the Whittle likelihood is correctly normalized without double-scaling.
    D_full = np.fft.rfft(data_ts_np) / float(C)
    f_k = FREQ[FMASK]
    D_k = D_full[FMASK]
    PSD_k = Pxx_psd[FMASK]
    y_obs = jnp.stack([jnp.real(jnp.asarray(D_k)), jnp.imag(jnp.asarray(D_k))], axis=-1)
    sigma_comp = jnp.sqrt(jnp.asarray(PSD_k) / 2.0)

    lnz_noise = float(
        -jnp.sum(
            (jnp.abs(jnp.asarray(D_k)) ** 2) / jnp.asarray(PSD_k)
            + jnp.log(jnp.pi * jnp.asarray(PSD_k))
        )
    )

    # for plotting...
    _, Pxx_data = _psd_consistent(data_ts_np)
    _, Pxx_sig = _psd_consistent(injected_wf_np)

    return WhittleContext(
        y_obs=y_obs,
        sigma_comp=sigma_comp,
        PSD_k=jnp.asarray(PSD_k),
        f_k=jnp.asarray(f_k),
        Pxx_signal=Pxx_sig,
        Pxx_data=Pxx_data,
        lnz_noise=lnz_noise,
    )


def rfft_model_waveform_scaled(z, amp, which):
    model = STARCCATO_SIGNAL if which == "signal" else STARCCATO_GLITCH
    wf = model.generate(z=z[None, :])[0]
    wf = wf * amp * REFERENCE_SCALE * (REFERENCE_DIST / 7.0)
    return _WINDOWED_RFFT(wf)


def make_whittle_model(which: str, ctx: WhittleContext):
    def model():
        amp = numpyro.sample("amp", dist.LogNormal(AMP_LOGMEAN, AMP_LOGSIGMA))
        z = numpyro.sample("z", dist.Normal(0, 1).expand([N_LATENTS]))
        ll = log_likelihood_whittle(z, amp, which, ctx)
        numpyro.factor("whittle_ll", ll)
    return model


def _log_prior(amp, z):
    lp_amp = dist.LogNormal(AMP_LOGMEAN, AMP_LOGSIGMA).log_prob(amp)
    lp_z = jnp.sum(dist.Normal(0, 1).log_prob(z))
    return lp_amp + lp_z


def log_posterior_whittle(theta_vec, which: str, ctx: WhittleContext):
    amp = jnp.asarray(theta_vec[0])
    z = jnp.asarray(theta_vec[1:])
    return _log_prior(amp, z) + log_likelihood_whittle(z, amp, which, ctx)

def log_likelihood_whittle(z, amp, which: str, ctx: WhittleContext):
    Hk_scaled = rfft_model_waveform_scaled(z, amp, which)
    Hk = jnp.stack([jnp.real(Hk_scaled), jnp.imag(Hk_scaled)], axis=-1)
    residual = ctx.y_obs - Hk
    sigma = jnp.expand_dims(ctx.sigma_comp, -1)
    return -0.5 * jnp.sum((residual / sigma) ** 2 + 2.0 * jnp.log(sigma) + jnp.log(2.0 * jnp.pi))


def log_prior_samples(samples):
    amp = jnp.asarray(samples["amp"])
    z = jnp.asarray(samples["z"])
    lp_amp = dist.LogNormal(AMP_LOGMEAN, AMP_LOGSIGMA).log_prob(amp)
    lp_z = jnp.sum(dist.Normal(0, 1).log_prob(z), axis=1)
    return np.array(lp_amp + lp_z)


def estimate_evidence_with_morph(samples, ll_total, which: str, ctx: WhittleContext, outdir, n_resamples: int = 2000):
    theta_samples = np.concatenate(
        [np.array(samples["amp"])[..., None], np.array(samples["z"])], axis=1
    )
    log_post_fn = jax.vmap(lambda th: log_posterior_whittle(th, which, ctx))
    log_post_vals = np.array(log_post_fn(jnp.array(theta_samples)))
    param_names = ["amp"] + [f"z{i}" for i in range(N_LATENTS)]

    out = f"{outdir}/morph_{which}"
    results = morph_evidence(
        post_samples=theta_samples,
        log_posterior_values=log_post_vals,
        log_posterior_function=lambda th: log_posterior_whittle(th, which, ctx),
        n_resamples=n_resamples,
        morph_type="pair",
        kde_bw="silverman",
        param_names=param_names,
        output_path=out,
        n_estimations=1,
        verbose=False,
    )
    # delete output files other than the main results

    results = np.array(results)
    logz_mean = float(np.mean(results[:, 0]))
    logz_err = float(np.mean(results[:, 1]))
    print(f"morphZ LnZ[{which}] ≈ {logz_mean:.2f} ± {logz_err:.2f}")
    return logz_mean, logz_err


def run_model(model_fn, which, ctx: WhittleContext, rng_key, num_warmup=300, num_samples=300):
    nuts = NUTS(model_fn, dense_mass=True)
    mcmc = MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key)
    idata = az.from_numpyro(mcmc)
    samples = mcmc.get_samples()
    amp_arr = np.array(samples["amp"])
    z_arr = np.array(samples["z"])
    print(f"[samples] {which}: amp mean={amp_arr.mean():.3e}, std={amp_arr.std():.3e}, |z| mean={np.linalg.norm(z_arr, axis=1).mean():.3e}")
    ess = az.ess(idata, method="bulk")
    ess_vals = ess.to_array().values
    ess_min = float(np.min(ess_vals))
    ess_max = float(np.max(ess_vals))
    print(f"{which} model sampling complete. ESS (bulk): min={ess_min:.1f}, max={ess_max:.1f}")
    loglike_fn = jax.vmap(lambda a, z: log_likelihood_whittle(z, a, which, ctx))
    ll_total = np.array(loglike_fn(samples["amp"], samples["z"]))
    return samples, ll_total, idata


def run_vi(model_fn, which, ctx: WhittleContext, rng_key, num_steps=10000, num_samples=500, lr=1e-4):
    guide = autoguide.AutoNormal(model_fn)
    svi = SVI(model_fn, guide, numpyro.optim.Adam(lr), loss=Trace_ELBO())
    svi_result = svi.run(rng_key, num_steps=num_steps)
    losses = np.array(svi_result.losses)
    # smoothed ELBO for readability
    if len(losses) > 10:
        window = 50
        kernel = np.ones(window) / window
        smoothed = np.convolve(losses, kernel, mode="valid")
    else:
        smoothed = losses
    plt.figure(figsize=(6, 3))
    plt.plot(losses, lw=0.5, alpha=0.4, label="raw")
    plt.plot(np.arange(len(smoothed)) + (len(losses) - len(smoothed)), smoothed, lw=1.0, label="smoothed")
    plt.legend()
    plt.xlabel("Step")
    plt.ylabel("ELBO loss")
    plt.title(f"{which} SVI ELBO")
    plt.tight_layout()
    plt.savefig(OUTROOT / f"elbo_{which}.png", dpi=120)
    plt.close()
    rng_samples = jax.random.split(rng_key)[1]
    samples = guide.sample_posterior(rng_samples, svi_result.params, sample_shape=(num_samples,))
    amp_arr = np.array(samples["amp"])
    z_arr = np.array(samples["z"])
    print(f"[samples] {which} (SVI): amp mean={amp_arr.mean():.3e}, std={amp_arr.std():.3e}, |z| mean={np.linalg.norm(z_arr, axis=1).mean():.3e}")
    loglike_fn = jax.vmap(lambda a, z: log_likelihood_whittle(z, a, which, ctx))
    ll_total = np.array(loglike_fn(samples["amp"], samples["z"]))
    return {k: np.array(v) for k, v in samples.items()}, ll_total, svi_result


# ----------------------------------------------------------------------
# Posterior predictive & plotting
# ----------------------------------------------------------------------
def posterior_draw_time_series(samples, ctx: WhittleContext, which="signal", n_draws=20):
    draws = []
    n = min(n_draws, samples["amp"].shape[0])
    # sample a random subset to avoid any ordering bias
    rng = np.random.default_rng(0)
    idx = rng.choice(samples["amp"].shape[0], size=n, replace=False)
    for i in idx:
        amp = samples["amp"][i]
        z = samples["z"][i]
        Hk_scaled = rfft_model_waveform_scaled(z, amp, which)
        Hfull_scaled = jnp.zeros_like(jnp.fft.rfft(jnp.zeros(N)))
        Hfull_scaled = Hfull_scaled.at[FMASK_JAX].set(Hk_scaled)
        Hfull_unscaled = Hfull_scaled * C
        ht = jnp.fft.irfft(Hfull_unscaled, n=N)
        draws.append(np.asarray(ht))
    return np.array(draws)

def prior_draw_time_series(ctx: WhittleContext, which="signal", n_draws=50, rng_key=None):
    """Draw waveforms from the prior for sanity checks/CI plotting."""
    rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)
    rng_amp, rng_z = jax.random.split(rng_key)
    amps = dist.LogNormal(AMP_LOGMEAN, AMP_LOGSIGMA).sample(rng_amp, (n_draws,))
    zs = dist.Normal(0, 1).sample(rng_z, (n_draws, N_LATENTS))
    draws = []
    for amp, z in zip(amps, zs):
        Hk_scaled = rfft_model_waveform_scaled(z, amp, which)
        Hfull_scaled = jnp.zeros_like(jnp.fft.rfft(jnp.zeros(N)))
        Hfull_scaled = Hfull_scaled.at[FMASK_JAX].set(Hk_scaled)
        ht = jnp.fft.irfft(Hfull_scaled * C, n=N)
        draws.append(np.asarray(ht))
    return np.array(draws)


def _psd_levels(label, series, fs=SAMPLE_RATE):
    f, Pxx = periodogram(series, fs=fs, axis=-1)
    mask = band_mask(f, BAND)
    med = float(np.median(Pxx[..., mask]))
    q10 = float(np.percentile(Pxx[..., mask], 10))
    q90 = float(np.percentile(Pxx[..., mask], 90))
    print(f"[PSD] {label}: median={med:.3e}, 10%={q10:.3e}, 90%={q90:.3e}")


def plot_traces(samples, outpath: Path, which: str):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(8, 4), sharex=True)
    axes[0].plot(samples["amp"], color="tab:blue", lw=0.5)
    axes[0].set_ylabel("amp")
    axes[1].plot(samples["z"], color="0.5", lw=0.3)
    axes[1].set_ylabel("z components")
    axes[1].set_xlabel("draw")
    fig.suptitle(f"{which} trace")
    fig.tight_layout()
    fig.savefig(outpath, dpi=120)
    plt.close(fig)


def main(detector=DEFAULT_DETECTOR, trigger_time=DEFAULT_TRIGGER_TIME, inject_kind=INJECT_KIND, seed=0,
         cache_file: Optional[Path] = None,
         inference: str = "nuts",
         num_warmup: int = 2000,
         num_samples: int = 2000,
         vi_steps: int = 2000,
         vi_samples: int = 500,
         morph_resamples: int = 75,
         ):
    outdir = OUTROOT / f"{int(trigger_time)}_{inject_kind}_seed{seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    rng_master = jax.random.PRNGKey(seed)
    rng_inj, rng_prior_sig, rng_prior_gli, rng_sig, rng_gli = jax.random.split(rng_master, 5)

    strain = fetch_strain(trigger_time, DURATION, cache_file=cache_file, sample_rate=SAMPLE_RATE)
    # cut into PSD + analysis segments
    tstart = trigger_time
    tend = trigger_time + DURATION
    tanalysis = tend - (ANALYSIS_LEN / SAMPLE_RATE)

    psd_segment = strain.crop(tstart, tanalysis)
    analysis_segment = strain.crop(tanalysis, tend)

    psd = psd_segment.psd(
        fftlength=ANALYSIS_DURATION, overlap=0.5 * ANALYSIS_DURATION, method="median"
    )

    if inject_kind == "noise":
        injection = np.zeros(ANALYSIS_LEN)
    else:
        # Draw injection directly from the prior (amp, z)
        analysis_segment, injection = inject_signal_into_data(analysis_segment, psd, inject_kind, rng_inj)

    snr_excess = snr_from_excess_power(analysis_segment.value, psd.value)
    ctx = build_whittle_context(analysis_segment, injection, psd)
    print(f">>>> {inject_kind} injection SNR = {snr_excess:.2f} <<<<")
    print(f"Settings: detector={detector}, gps={trigger_time}, seed={seed}")

    # Plot prior draws for sanity before sampling
    prior_sig = prior_draw_time_series(ctx, "signal", n_draws=100, rng_key=rng_prior_sig)
    prior_gli = prior_draw_time_series(ctx, "glitch", n_draws=100, rng_key=rng_prior_gli)
    _psd_levels("injection", injection[np.newaxis, :])
    _psd_levels("prior signal", prior_sig)
    _psd_levels("prior glitch", prior_gli)
    plot_data(
        ctx,
        inj_type=inject_kind,
        posterior_signal=np.array(prior_sig) if len(prior_sig) > 0 else None,
        posterior_blip=np.array(prior_gli) if len(prior_gli) > 0 else None,
        posterior_signal_label="Prior signal",
        posterior_blip_label="Prior glitch",
        injection_snr=snr_excess,
        logBF=None,
        status_text=None,
        outpath=outdir / "prior_summary.png",
    )

    model_sig = make_whittle_model("signal", ctx)
    model_gli = make_whittle_model("glitch", ctx)

    if inference == "nuts":
        res_sig, ll_sig, idata_sig = run_model(
            model_sig, "signal", ctx, rng_sig, num_warmup=num_warmup, num_samples=num_samples
        )
        res_gli, ll_gli, idata_gli = run_model(
            model_gli, "glitch", ctx, rng_gli, num_warmup=num_warmup, num_samples=num_samples
        )
        # ArviZ trace plots for diagnostics (amp and latents)
        az.plot_trace(idata_sig, var_names=["amp", "z"])
        plt.tight_layout()
        plt.savefig(outdir / "trace_signal.png", dpi=120)
        plt.close()
        az.plot_trace(idata_gli, var_names=["amp", "z"])
        plt.tight_layout()
        plt.savefig(outdir / "trace_glitch.png", dpi=120)
        plt.close()
    else:
        res_sig, ll_sig, vi_sig = run_vi(
            model_sig, "signal", ctx, rng_sig, num_steps=vi_steps, num_samples=vi_samples
        )
        res_gli, ll_gli, vi_gli = run_vi(
            model_gli, "glitch", ctx, rng_gli, num_steps=vi_steps, num_samples=vi_samples
        )
        plot_traces(res_sig, outdir / "trace_signal.png", "signal")
        plot_traces(res_gli, outdir / "trace_glitch.png", "glitch")

    if morph_resamples > 0:
        lnz_sig, lnz_sig_err = estimate_evidence_with_morph(
            res_sig, ll_sig, "signal", ctx, outdir, n_resamples=morph_resamples
        )
        lnz_gli, lnz_gli_err = estimate_evidence_with_morph(
            res_gli, ll_gli, "glitch", ctx, outdir, n_resamples=morph_resamples
        )
    else:
        lnz_sig = lnz_sig_err = lnz_gli = lnz_gli_err = float("nan")
    lnz_noise = ctx.lnz_noise

    logBF_sig_alt = float("nan")
    status_text = None
    if morph_resamples > 0:
        print(f"LnZ[noise] (analytic)  = {lnz_noise:.2f}")
        print(f"morphZ LnZ[signal]     = {lnz_sig:.2f} ± {lnz_sig_err:.2f}")
        print(f"morphZ LnZ[glitch]     = {lnz_gli:.2f} ± {lnz_gli_err:.2f}")
        print(f"morphZ ΔLnZ(sig–noise) ≈ {lnz_sig - lnz_noise:.2f} ± {lnz_sig_err:.2f}")
        print(f"morphZ ΔLnZ(sig–gli)   ≈ {lnz_sig - lnz_gli:.2f} ± {np.hypot(lnz_sig_err, lnz_gli_err):.2f}")

        logZ_alt = jax.scipy.special.logsumexp(jnp.array([lnz_gli, lnz_noise]))
        logBF_sig_alt = lnz_sig - logZ_alt
        post_logs = jnp.array([lnz_sig, lnz_gli, lnz_noise])
        posts = jnp.exp(post_logs - jax.scipy.special.logsumexp(post_logs))
        p_sig, p_gli, p_noise = [float(x) for x in posts]
        print(f"Posterior probs (equal model priors): P(sig)={p_sig:.3f}, P(gli)={p_gli:.3f}, P(noise)={p_noise:.3f}")

        print("________________________________________________")
        print(f"log BF(signal | glitch/noise) ≈ {float(logBF_sig_alt):.2f}")
        print(f"Excess-power SNR (Whittle band) ≈ {snr_excess:.2f}")
        # summary -- if signal injection, log BF should be positive, if glitch/noise injection, log BF should be negative
        status_text = "status: n/a"
        if inject_kind == "signal":
            status_text = "status: PASS" if logBF_sig_alt > 0 else "status: FAIL"
            if logBF_sig_alt > 0:
                print(">>> Correctly recovered signal injection! <<<")
            else:
                print(">>> Failed to recover signal injection! <<<")
        elif inject_kind in ("glitch", "noise"):
            status_text = "status: PASS" if logBF_sig_alt < 0 else "status: FAIL"
        print("________________________________________________")

    ts_sig = posterior_draw_time_series(res_sig, ctx, "signal", n_draws=100)
    ts_gli = posterior_draw_time_series(res_gli, ctx, "glitch", n_draws=100)

    plot_data(
        ctx,
        inj_type=inject_kind,
        posterior_signal=np.array(ts_sig) if len(ts_sig) > 0 else None,
        posterior_blip=np.array(ts_gli) if len(ts_gli) > 0 else None,
        posterior_signal_label="Posterior signal",
        posterior_blip_label="Posterior glitch",
        injection_snr=snr_excess,
        logBF=float(logBF_sig_alt),
        status_text=status_text,
        outpath=outdir / "frequency_summary.png",
    )
    # Time-domain overlay with injection + prior/posterior bands (80% CI)
    plt.figure(figsize=(8, 4))
    t = np.arange(len(analysis_segment.value)) / float(SAMPLE_RATE)
    plt.plot(t, injection, color="tab:orange", lw=0.9, label="Injection")

    def _coverage(series, ref, lo_q=10, hi_q=90):
        """Return the fraction of points where ref lies inside the quantile band."""
        if series is None or len(series) == 0:
            return float("nan")
        arr = np.asarray(series)
        lo = np.percentile(arr, lo_q, axis=0)
        hi = np.percentile(arr, hi_q, axis=0)
        ref = np.asarray(ref)
        m = min(len(lo), len(ref))
        inside = np.logical_and(ref[:m] >= lo[:m], ref[:m] <= hi[:m])
        return float(np.mean(inside))

    def _band(series, color, label):
        if series is None or len(series) == 0:
            return
        arr = np.asarray(series)
        med = np.median(arr, axis=0)
        lo = np.percentile(arr, 10, axis=0)
        hi = np.percentile(arr, 90, axis=0)
        lo_mid = np.percentile(arr, 25, axis=0)
        hi_mid = np.percentile(arr, 75, axis=0)
        # quick diagnostic for band width
        spread = float(np.mean(hi - lo))
        print(f"[band] {label}: mean spread (10-90) = {spread:.3e}")
        plt.fill_between(t[: len(med)], lo[: len(med)], hi[: len(med)], color=color, alpha=0.30)
        plt.fill_between(t[: len(med)], lo_mid[: len(med)], hi_mid[: len(med)], color=color, alpha=0.50)
        plt.plot(t[: len(med)], med[: len(med)], color=color, lw=1.0, label=label)
        cov = _coverage(series, injection, 10, 90)
        print(f"[coverage] {label}: injection in 10-90 band ≈ {cov:.3f}")

    _band(prior_sig, "tab:red", "Prior signal")
    _band(prior_gli, "tab:green", "Prior glitch")
    _band(ts_sig, "darkred", "Posterior signal")
    _band(ts_gli, "darkgreen", "Posterior glitch")

    plt.xlabel("Time [s]")
    plt.ylabel("Strain")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "time_domain_overlay.png", dpi=120)
    plt.close()

    # Save metrics
    metrics_path = outdir / "metrics.csv"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(
            "detector,gps,inject,seed,snr_excess,lnz_noise,lnz_signal,lnz_signal_err,lnz_glitch,lnz_glitch_err,logBF_sig_alt\n"
        )
        f.write(
            f"{detector},"
            f"{trigger_time},"
            f"{inject_kind},"
            f"{seed},"
            f"{snr_excess},"
            f"{lnz_noise:.6f},"
            f"{lnz_sig:.6f},"
            f"{lnz_sig_err:.6f},"
            f"{lnz_gli:.6f},"
            f"{lnz_gli_err:.6f},"
            f"{float(logBF_sig_alt):.6f}\n"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="One-detector Starccato analysis")
    parser.add_argument("--detector", default=DEFAULT_DETECTOR, help="Detector (e.g., H1)")
    parser.add_argument("--gps", type=float, default=DEFAULT_TRIGGER_TIME, help="GPS start time")
    parser.add_argument("--inject", choices=["signal", "glitch", "noise"], default=INJECT_KIND, help="Injection kind")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--cache-file", type=str, default=str(DEFAULT_CACHE_FILE),
                        help="Path to a GWOSC HDF5 cache file covering the GPS window.")
    parser.add_argument("--inference", choices=["svi", "nuts"], default="nuts", help="Inference method.")
    parser.add_argument("--num-warmup", type=int, default=2000, help="NUTS warmup (if inference=nuts).")
    parser.add_argument("--num-samples", type=int, default=2000, help="NUTS samples (if inference=nuts).")
    parser.add_argument("--vi-steps", type=int, default=2000, help="SVI steps (if inference=svi).")
    parser.add_argument("--vi-samples", type=int, default=500, help="SVI posterior draws (if inference=svi).")
    parser.add_argument("--morph-resamples", type=int, default=75, help="Number of morphZ resamples.")
    args = parser.parse_args()

    main(
        detector=args.detector,
        trigger_time=args.gps,
        inject_kind=args.inject,
        seed=args.seed,
        cache_file=args.cache_file,
        inference=args.inference,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        vi_steps=args.vi_steps,
        vi_samples=args.vi_samples,
        morph_resamples=args.morph_resamples,
    )
