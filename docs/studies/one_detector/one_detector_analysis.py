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
from numpyro.infer import MCMC, NUTS, log_likelihood
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

N_LATENTS = 32
REFERENCE_DIST = 10.0
# Scale the waveform generator to roughly match observed strain levels.
# With AMP_LOGMEAN near -30 (median ~5e-14), a scale of 1e-7 yields
# typical strains on the order of 1e-21 after the LogNormal draw.
REFERENCE_SCALE = 1e-7
BASE_WF_LEN = len(StarccatoCCSNe().generate(z=np.zeros((1, N_LATENTS)))[0])
# Center amplitude prior near unity so the in-band rescaling can be matched by the sampler.
# A wide LogNormal keeps support for weaker injections.
AMP_LOGMEAN = 0.0
AMP_LOGSIGMA = 1.5

STARCCATO_SIGNAL = StarccatoCCSNe()
STARCCATO_GLITCH = StarccatoBlip()

INJECT_KIND = "glitch"  # default, overridden by main args


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


def inject_signal_into_data(analysis_segment: TimeSeries, psd: FrequencySeries, kind: str, target_snr: float):
    z = np.zeros(N_LATENTS)
    model = STARCCATO_SIGNAL if kind == "signal" else STARCCATO_GLITCH
    distance = 7.0  # Mpc
    amp = 1.0
    wf = model.generate(z=z[None, :])[0]
    wf = np.array(wf, dtype=np.float64, copy=True)
    injection = wf * amp * REFERENCE_SCALE * (REFERENCE_DIST / distance)

    data = np.asarray(analysis_segment.value)
    # center-crop/pad injection to match analysis segment length
    if len(injection) > len(data):
        injection = injection[-len(data):]
    elif len(injection) < len(data):
        injection = np.pad(injection, (len(data) - len(injection), 0))

    snr_injection = snr_from_excess_power(injection, psd.value)

    # now we need to scale the injection to reach the target SNR
    scale = target_snr / snr_injection if snr_injection > 0 else 1.0
    injection = injection * scale
    analysis_segment = TimeSeries(data + injection, sample_rate=analysis_segment.sample_rate)
    return analysis_segment, injection


def plot_data(
        ctx: WhittleContext,
        inj_type: str,
        injection_snr: Optional[float] = None,
        posterior_signal: Optional[np.ndarray] = None,
        posterior_blip: Optional[np.ndarray] = None,
        logBF: Optional[float] = None,
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
        f_p, Pxx_p = periodogram(series * WINDOW_NP, fs=SAMPLE_RATE, axis=-1)
        med = np.median(Pxx_p[:, FMASK], axis=0)

        for qtls in [[10, 90], [20, 80]]:
            lo = np.percentile(Pxx_p[:, FMASK], qtls[0], axis=0)
            hi = np.percentile(Pxx_p[:, FMASK], qtls[1], axis=0)
            ax.fill_between(f_p[FMASK], lo, hi, color=color, alpha=0.2)
        ax.loglog(f_p[FMASK], med, color=color, lw=1.1, label=label)

    if posterior_signal is not None and len(posterior_signal) > 0:
        _plot_post("Signal", "tab:red", posterior_signal)
    if posterior_blip is not None and len(posterior_blip) > 0:
        _plot_post("Glitch", "tab:green", posterior_blip)

    ax.set_xlim(*BAND)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [1/Hz]")
    title_parts = []
    if injection_snr is not None:
        title_parts.append(f"SNR≈{injection_snr:.1f}")
    if logBF is not None:
        title_parts.append(f"logBF≈{logBF:.1f}")
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
    _, Pxx_data = periodogram(data_ts_np, fs=fs)
    _, Pxx_sig = periodogram(injected_wf_np, fs=fs)
    Pxx_data = Pxx_data[FMASK]
    Pxx_sig = Pxx_sig[FMASK]

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
    def model(y=None):
        amp = numpyro.sample("amp", dist.LogNormal(AMP_LOGMEAN, AMP_LOGSIGMA))
        z = numpyro.sample("z", dist.Normal(0, 1).expand([N_LATENTS]))
        Hk = rfft_model_waveform_scaled(z, amp, which)
        mu = jnp.stack([jnp.real(Hk), jnp.imag(Hk)], axis=-1)
        numpyro.sample(
            "y",
            dist.Normal(mu, jnp.expand_dims(jnp.array(ctx.sigma_comp), -1)),
            obs=y,
        )
    return model


def _log_prior(amp, z):
    lp_amp = dist.LogNormal(AMP_LOGMEAN, AMP_LOGSIGMA).log_prob(amp)
    lp_z = jnp.sum(dist.Normal(0, 1).log_prob(z))
    return lp_amp + lp_z


def log_posterior_whittle(theta_vec, which: str, ctx: WhittleContext):
    amp = jnp.asarray(theta_vec[0])
    if amp <= 0.0:
        return -jnp.inf
    z = jnp.asarray(theta_vec[1:])
    lp = _log_prior(amp, z)
    Hk = rfft_model_waveform_scaled(z, amp, which)
    mu = jnp.stack([jnp.real(Hk), jnp.imag(Hk)], axis=-1)
    sigma = jnp.asarray(ctx.sigma_comp)
    resid = jnp.asarray(ctx.y_obs) - mu
    ll = -0.5 * jnp.sum(
        (resid / sigma[:, None]) ** 2 + 2.0 * jnp.log(sigma[:, None]) + jnp.log(2.0 * jnp.pi)
    )
    return float(lp + ll)


def log_prior_samples(samples):
    amp = jnp.asarray(samples["amp"])
    z = jnp.asarray(samples["z"])
    lp_amp = dist.LogNormal(AMP_LOGMEAN, AMP_LOGSIGMA).log_prob(amp)
    lp_z = jnp.sum(dist.Normal(0, 1).log_prob(z), axis=1)
    return np.array(lp_amp + lp_z)


def estimate_evidence_with_morph(samples, ll_total, which: str, ctx: WhittleContext, outdir):
    theta_samples = np.concatenate(
        [np.array(samples["amp"])[..., None], np.array(samples["z"])], axis=1
    )
    log_post_vals = log_prior_samples(samples) + np.array(ll_total)
    param_names = ["amp"] + [f"z{i}" for i in range(N_LATENTS)]

    # _buf_out = io.StringIO()
    # _buf_err = io.StringIO()
    # root_logger = logging.getLogger()
    # prev_level = root_logger.level
    # root_logger.setLevel(logging.ERROR)
    # try:
    #     with contextlib.redirect_stdout(_buf_out), contextlib.redirect_stderr(_buf_err):
    #         out = f"{outdir}/morph_{which}"
    #         results = morph_evidence(
    #             post_samples=theta_samples,
    #             log_posterior_values=log_post_vals,
    #             log_posterior_function=lambda th: log_posterior_whittle(th, which, ctx),
    #             n_resamples=2000,
    #             morph_type="pair",
    #             kde_bw="silverman",
    #             param_names=param_names,
    #             output_path=out,
    #             n_estimations=1,
    #             verbose=False,
    #         )
    #         # delete output files other than the main results
    #         shutil.rmtree(out, ignore_errors=True)
    # finally:
    #     root_logger.setLevel(prev_level)

    out = f"{outdir}/morph_{which}"
    results = morph_evidence(
        post_samples=theta_samples,
        log_posterior_values=log_post_vals,
        log_posterior_function=lambda th: log_posterior_whittle(th, which, ctx),
        n_resamples=2000,
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


def run_model(model_fn, y, name, rng_key, num_warmup=600, num_samples=600):
    nuts = NUTS(model_fn)
    mcmc = MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, y=y)
    idata = az.from_numpyro(mcmc)
    ess = az.ess(idata, method="bulk")
    ess_vals = ess.to_array().values
    ess_min = float(np.min(ess_vals))
    ess_max = float(np.max(ess_vals))
    print(f"{name} model sampling complete. ESS (bulk): min={ess_min:.1f}, max={ess_max:.1f}")
    samples = mcmc.get_samples()
    ll = log_likelihood(model_fn, samples, y=y)["y"]
    ll_total = jnp.sum(ll, axis=(1, 2))
    return samples, np.array(ll_total)


# ----------------------------------------------------------------------
# Posterior predictive & plotting
# ----------------------------------------------------------------------
def posterior_draw_time_series(samples, ctx: WhittleContext, which="signal", n_draws=20):
    draws = []
    n = min(n_draws, samples["amp"].shape[0])
    for i in range(n):
        amp = samples["amp"][i]
        z = samples["z"][i]
        Hk_scaled = rfft_model_waveform_scaled(z, amp, which)
        Hfull_scaled = jnp.zeros_like(jnp.fft.rfft(jnp.zeros(N)))
        Hfull_scaled = Hfull_scaled.at[FMASK_JAX].set(Hk_scaled)
        Hfull_unscaled = Hfull_scaled * C
        ht = jnp.fft.irfft(Hfull_unscaled, n=N)
        draws.append(np.asarray(ht))
    return np.array(draws)


def main(detector=DEFAULT_DETECTOR, trigger_time=DEFAULT_TRIGGER_TIME, inject_kind=INJECT_KIND, seed=0,
         cache_file: Optional[Path] = None):
    outdir = OUTROOT / f"{int(trigger_time)}_{inject_kind}_seed{seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    rng_master = jax.random.PRNGKey(seed)

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
        # Default to uniform in [5, 100] for signals and [0, 100] for glitches if no range is provided.
        lo, hi = DEFAULT_SNR_RANGE
        snr_target = float(jax.random.uniform(rng_master, (), minval=lo, maxval=hi))
        analysis_segment, injection, = inject_signal_into_data(analysis_segment, psd, inject_kind, snr_target)

    snr_excess = snr_from_excess_power(analysis_segment.value, psd.value)
    ctx = build_whittle_context(analysis_segment, injection, psd)
    print(f">>>> {inject_kind} injection SNR = {snr_excess:.2f} <<<<")

    rng_sig, rng_gli = jax.random.split(rng_master)
    model_sig = make_whittle_model("signal", ctx)
    model_gli = make_whittle_model("glitch", ctx)

    res_sig, ll_sig = run_model(model_sig, ctx.y_obs, "signal", rng_sig)
    res_gli, ll_gli = run_model(model_gli, ctx.y_obs, "glitch", rng_gli)

    print("\nEstimating evidences with morphZ...")
    lnz_sig_morph, lnz_sig_err = estimate_evidence_with_morph(res_sig, ll_sig, "signal", ctx, outdir)
    lnz_gli_morph, lnz_gli_err = estimate_evidence_with_morph(res_gli, ll_gli, "glitch", ctx, outdir)
    lnz_noise = ctx.lnz_noise

    print(f"LnZ[noise] (analytic)  = {lnz_noise:.2f}")
    print(f"morphZ LnZ[signal]     = {lnz_sig_morph:.2f} ± {lnz_sig_err:.2f}")
    print(f"morphZ LnZ[glitch]     = {lnz_gli_morph:.2f} ± {lnz_gli_err:.2f}")
    print(f"morphZ ΔLnZ(sig–noise) ≈ {lnz_sig_morph - lnz_noise:.2f} ± {lnz_sig_err:.2f}")
    print(f"morphZ ΔLnZ(sig–gli)   ≈ {lnz_sig_morph - lnz_gli_morph:.2f} ± {np.hypot(lnz_sig_err, lnz_gli_err):.2f}")

    logZ_alt = jax.scipy.special.logsumexp(jnp.array([lnz_gli_morph, lnz_noise]))
    logBF_sig_alt = lnz_sig_morph - logZ_alt
    post_logs = jnp.array([lnz_sig_morph, lnz_gli_morph, lnz_noise])
    posts = jnp.exp(post_logs - jax.scipy.special.logsumexp(post_logs))
    p_sig, p_gli, p_noise = [float(x) for x in posts]
    print(f"Posterior probs (equal model priors): P(sig)={p_sig:.3f}, P(gli)={p_gli:.3f}, P(noise)={p_noise:.3f}")

    print("________________________________________________")
    print(f"log BF(signal | glitch/noise) ≈ {float(logBF_sig_alt):.2f}")
    print(f"Excess-power SNR (Whittle band) ≈ {snr_excess:.2f}")
    # summary -- if signal injection, log BF should be positive, if glitch/noise injection, log BF should be negative
    if inject_kind == "signal":
        if logBF_sig_alt > 0:
            print(">>> Correctly recovered signal injection! <<<")
        else:
            print(">>> Failed to recover signal injection! <<<")


    print("________________________________________________")

    ts_sig = posterior_draw_time_series(res_sig, ctx, "signal", n_draws=100)
    ts_gli = posterior_draw_time_series(res_gli, ctx, "glitch", n_draws=100)

    plot_data(
        ctx,
        inj_type=inject_kind,
        posterior_signal=np.array(ts_sig) if len(ts_sig) > 0 else None,
        posterior_blip=np.array(ts_gli) if len(ts_gli) > 0 else None,
        injection_snr=snr_excess,
        logBF=float(logBF_sig_alt),
        outpath=outdir / "frequency_summary.png",
    )

    # Save metrics
    metrics_path = outdir / "metrics.csv"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(
            "detector,gps,inject,seed,snr_excess,lnz_noise,lnz_signal,lnz_signal_err,lnz_glitch,lnz_glitch_err,logBF_sig_alt\n")
        f.write(
            f"{detector},"
            f"{trigger_time},"
            f"{inject_kind},"
            f"{seed},"
            f"{snr_excess},"
            f"{lnz_noise:.6f},"
            f"{lnz_sig_morph:.6f},"
            f"{lnz_sig_err:.6f},"
            f"{lnz_gli_morph:.6f},"
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
    parser.add_argument("--cache-file", type=str, default='cache/H-H1_GWOSC_O2_4KHZ_R1-1187721216-4096.hdf5',
                        help="Path to a GWOSC HDF5 cache file covering the GPS window.")
    args = parser.parse_args()

    main(
        detector=args.detector,
        trigger_time=args.gps,
        inject_kind=args.inject,
        seed=args.seed,
        cache_file=args.cache_file,
    )
