# %% [markdown]
# Starccato + NumPyro with Whittle likelihood, morphZ evidence, and SNR/PSD summaries.

# %%
# !pip install gwpy numpyro starccato-jax matplotlib morphZ scipy -q

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, periodogram, butter, sosfilt
from scipy.signal.windows import tukey
from gwpy.timeseries import TimeSeries

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

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
HERE = Path(__file__).parent.resolve()
OUTROOT = HERE / "out"
OUTROOT.mkdir(parents=True, exist_ok=True)
DEFAULT_CACHE_DIR = HERE / "cache"
DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TRIGGER_TIME = 1186741733  # quiet segment
DEFAULT_DETECTOR = "H1"
SAMPLE_RATE = 4096.0
BAND = (100.0, 1024.0)
DURATION = 4.0
ANALYSIS_LEN = 512
DEFAULT_SNR_RANGE = (5.0, 100.0)
GLITCH_SNR_EXCESS_RANGE = (0.0, 100.0)

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

INJECT_KIND = "signal"  # default, overridden by main args
SNR_INJECTION = 100.0


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def band_mask(f: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    return (f >= band[0]) & (f <= band[1])


def _strain_has_invalid(strain: TimeSeries) -> bool:
    data = strain.value
    return not np.all(np.isfinite(data))


def _fetch_and_cache(detector, trigger_time, duration, sample_rate, cache_file: Path):
    print(f"Fetching {detector} data around GPS {trigger_time}...")
    strain = TimeSeries.fetch_open_data(
        detector,
        trigger_time - 2,
        trigger_time + duration + 2,
        sample_rate=sample_rate,
    ).crop(trigger_time, trigger_time + duration)
    if _strain_has_invalid(strain):
        raise RuntimeError("Fetched strain contains NaNs or infs; aborting cache write.")
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    try:
        cache_file.unlink()
    except FileNotFoundError:
        pass
    print(f"Saving to cache: {cache_file}")
    strain.write(cache_file)
    return strain


def _parse_gwosc_filename(path: Path) -> Optional[tuple[int, int]]:
    name = path.stem
    parts = name.split("-")
    if len(parts) < 2:
        return None
    try:
        start = int(parts[-2])
        dur = int(parts[-1])
    except ValueError:
        return None
    return start, dur


def _find_local_gwosc_file(cache_dir: Path, detector: str, trigger_time: float, duration: float) -> Optional[Path]:
    if cache_dir is None or not cache_dir.exists():
        return None

    trig_start = float(trigger_time)
    trig_end = trig_start + duration
    detector_tag = detector.upper()

    best_match = None
    best_start = -np.inf

    for candidate in sorted(cache_dir.glob("*.hdf5")):
        if detector_tag not in candidate.name.upper():
            continue
        parsed = _parse_gwosc_filename(candidate)
        if not parsed:
            continue
        start, dur = parsed
        file_end = start + dur
        if start <= trig_start and trig_end <= file_end:
            if start > best_start:
                best_match = candidate
                best_start = start

    return best_match


def _read_timeseries_from_file(file_path: Path, detector: str) -> TimeSeries:
    suffix = file_path.suffix.lower()
    is_hdf5 = suffix in {".hdf5", ".hdf", ".h5"}
    channel = f"{detector}:GWOSC-4KHZ_R1_STRAIN"
    attempts = []
    if is_hdf5:
        attempts.append({"format": "hdf5.gwosc"})
    attempts.append({"channel": channel})
    attempts.append({})

    last_err = None
    for options in attempts:
        try:
            return TimeSeries.read(file_path, **options)
        except Exception as exc:  # pragma: no cover - fallback path
            last_err = exc
    raise RuntimeError(f"Failed to read {file_path}: {last_err}")


def _compute_inband_snr(injected_wf: np.ndarray, ctx: dict) -> float:
    if injected_wf is None or injected_wf.size == 0:
        return 0.0
    H_inj = np.fft.rfft(injected_wf * ctx["win"]) / ctx["C"]
    return float(np.sqrt(np.sum(np.abs(H_inj[ctx["mask_r"]]) ** 2 / ctx["S_k"])))


def _compute_excess_power_snr_injection(injected_wf: np.ndarray, ctx: dict) -> float:
    """
    Excess-power SNR of the injection alone using the Whittle band/PSD in the context.
    Keeps the scaling step independent of the stochastic noise realization.
    """
    if injected_wf is None or injected_wf.size == 0:
        return 0.0
    inj_seg = _center_crop_or_pad(np.asarray(injected_wf)[-ctx["N"] :], ctx["N"])
    H_inj = np.fft.rfft(inj_seg * ctx["win"]) / ctx["C"]
    return float(np.sqrt(np.sum(np.abs(H_inj[ctx["mask_r"]]) ** 2 / ctx["S_k"])))


def _load_local_segment(cache_dir: Path, detector: str, trigger_time: float, duration: float) -> Optional[TimeSeries]:
    local_file = _find_local_gwosc_file(cache_dir, detector, trigger_time, duration)
    if local_file is None:
        return None

    print(f"Loading local GWOSC file {local_file}...")
    ts_full = _read_timeseries_from_file(local_file, detector)
    segment = ts_full.crop(trigger_time, trigger_time + duration)
    if len(segment) == 0:
        raise RuntimeError(f"Local file {local_file} does not cover requested GPS range.")
    if _strain_has_invalid(segment):
        raise RuntimeError(f"Local strain segment from {local_file} contains NaNs or infs.")
    return segment


def fetch_strain(detector, trigger_time, duration, sample_rate, cache_dir: Optional[Path] = None):
    """
    Load strain from a local GWOSC cache (HDF5) using the standard channel/format.
    """
    cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_segment = _load_local_segment(cache_dir, detector, trigger_time, duration)
    if local_segment is not None:
        return local_segment

    cache_file = cache_dir / f"{detector}_{trigger_time}_{duration}s_{int(sample_rate)}Hz.hdf5"
    channel = f"{detector}:GWOSC-4KHZ_R1_STRAIN"
    try:
        if cache_file.exists():
            print(f"Loading cached data from {cache_file}...")
            strain = TimeSeries.read(cache_file, format="hdf5.gwosc", channel=channel)
        else:
            raise FileNotFoundError(f"{cache_file} not found in cache_dir {cache_dir}")
    except Exception as exc:
        raise RuntimeError(f"Failed to load strain from {cache_file}: {exc}")

    strain = strain.crop(trigger_time, trigger_time + duration)
    if _strain_has_invalid(strain):
        raise RuntimeError("Strain contains NaNs or infs.")
    return strain


def gen_waveform(latents: np.ndarray, amp: float = 1.0, distance: float = 7.0, model=None):
    model = STARCCATO_SIGNAL if model is None else model
    wf = model.generate(z=latents[None, :])[0]
    wf = np.array(wf, dtype=np.float64, copy=True)
    return wf * amp * REFERENCE_SCALE * (REFERENCE_DIST / distance)


def inject_signal_into_data(raw_data: np.ndarray, kind: str, target_snr: float, fs: float, sos):
    """
    Inject Starccato signal/glitch with a target optimal SNR in the analysis band.

    We compute a reference optimal SNR for the base waveform using the Welch PSD
    of the *off-source* portion (excluding the tail where we inject), then scale to
    the desired target SNR. Injection is placed at the end of the segment.
    """
    z = np.zeros(N_LATENTS)
    model = STARCCATO_SIGNAL if kind == "signal" else STARCCATO_GLITCH
    wf0 = gen_waveform(z, model=model)

    N_wf = min(len(wf0), len(raw_data))
    wf0 = wf0[:N_wf]
    start = len(raw_data) - N_wf  # inject at the end
    data_crop = raw_data[start : start + N_wf]

    # Off-source portion excludes the injection tail
    data_off = raw_data[: start] if start > 0 else raw_data

    win_local = tukey(N_wf, 0.1)
    data_off_win = data_off * tukey(len(data_off), 0.1) if len(data_off) > 0 else data_off
    nper = 512 if len(data_off) >= 512 else max(len(data_off), 256)
    freqs_full_local, psd_full_local = welch(data_off_win, fs=fs, nperseg=nper)

    U_local = np.mean(win_local**2)
    C_local = np.sqrt(fs * (N_wf * U_local) / 2.0)

    f_rfft_local = np.fft.rfftfreq(N_wf, 1.0 / fs)
    mask_local = band_mask(f_rfft_local, BAND)

    wf_win = sosfilt(sos, wf0) * win_local
    Hk_scaled = np.fft.rfft(wf_win) / C_local
    S_k_local = np.interp(
        f_rfft_local[mask_local],
        freqs_full_local,
        psd_full_local,
        left=psd_full_local[0],
        right=psd_full_local[-1],
    )
    rho2 = np.sum(np.abs(Hk_scaled[mask_local]) ** 2 / S_k_local)
    rho_ref = np.sqrt(max(rho2, 1e-30))

    scale = target_snr / rho_ref
    wf_scaled = sosfilt(sos, wf0) * scale
    injected = data_crop + wf_scaled
    # Report achieved SNR with the same PSD/mask
    Hk_scaled_target = np.fft.rfft(wf_scaled * win_local) / C_local
    rho2_target = np.sum(np.abs(Hk_scaled_target[mask_local]) ** 2 / S_k_local)
    rho_target = np.sqrt(max(rho2_target, 1e-30))
    return injected, wf_scaled, rho_target


def _center_crop_or_pad(x: np.ndarray, length: int) -> np.ndarray:
    """Center-crop or zero-pad 1D array to the desired length."""
    n = len(x)
    if n == length:
        return x
    if n > length:
        start = (n - length) // 2
        return x[start : start + length]
    pad = np.zeros(length, dtype=x.dtype)
    start = (length - n) // 2
    pad[start : start + n] = x
    return pad


def initial_plot(time, data_vis, fs, detector, outdir):
    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(14, 4))
    ax_t.plot(time - time[0], data_vis, lw=0.6, color="k")
    ax_t.set_title(f"{detector} strain (band-limited {BAND[0]}–{BAND[1]} Hz)")
    ax_t.set_xlabel("Time [s]")
    ax_t.set_ylabel("Strain")
    ax_t.grid(alpha=0.3)

    data_vis_win = data_vis * tukey(len(data_vis), 0.1)
    f_init, Pxx_init = welch(data_vis_win, fs=fs, nperseg=256)
    mask_init = band_mask(f_init, BAND)
    ax_f.loglog(f_init[mask_init], Pxx_init[mask_init], lw=1.0, color="k")
    ax_f.set_title("Data PSD (Welch)")
    ax_f.set_xlabel("Frequency [Hz]")
    ax_f.set_ylabel("PSD [1/Hz]")
    ax_f.set_xlim(*BAND)
    ax_f.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(outdir / "data.png", dpi=150)
    plt.close(fig)


def plot_data(
    raw_data: np.ndarray,
    injection: Optional[np.ndarray],
    welch_psd: tuple[np.ndarray, np.ndarray],
    posterior_signal: Optional[np.ndarray] = None,
    posterior_blip: Optional[np.ndarray] = None,
    injection_snr: Optional[float] = None,
    logBF: Optional[float] = None,
    outpath: Optional[Path] = None,
):
    freqs_welch, psd_welch = welch_psd
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(freqs_welch, np.maximum(psd_welch, 1e-30), color="k", lw=1.2, label="Noise PSD (Welch)")

    f_data, Pxx_data = periodogram(raw_data * tukey(len(raw_data), 0.1), fs=SAMPLE_RATE)
    m_data = band_mask(f_data, BAND)
    ax.loglog(f_data[m_data], np.maximum(Pxx_data[m_data], 1e-30), color="0.6", alpha=0.4, label="Data periodogram")

    if injection is not None and injection.size > 0:
        f_inj, Pxx_inj = periodogram(injection * tukey(len(injection), 0.1), fs=SAMPLE_RATE)
        m_inj = band_mask(f_inj, BAND)
        ax.loglog(f_inj[m_inj], np.maximum(Pxx_inj[m_inj], 1e-30), color="tab:orange", lw=1.1, label="Injection PSD")

    def _plot_post(label, color, series):
        if series is None or len(series) == 0:
            return
        f_p, Pxx_p = periodogram(series, fs=SAMPLE_RATE, axis=-1)
        mask_p = band_mask(f_p, BAND)
        med = np.median(Pxx_p[:, mask_p], axis=0)
        lo = np.percentile(Pxx_p[:, mask_p], 5, axis=0)
        hi = np.percentile(Pxx_p[:, mask_p], 95, axis=0)
        ax.fill_between(f_p[mask_p], lo, hi, color=color, alpha=0.2, label=f"{label} 90% CI")
        ax.loglog(f_p[mask_p], med, color=color, lw=1.1, label=f"{label} median")

    _plot_post("Signal", "tab:red", posterior_signal)
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
    if outpath:
        fig.savefig(outpath, dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------
# Whittle helpers
# ----------------------------------------------------------------------
def build_whittle_context(data_ts, injected_wf, fs, analysis_len, psd_ref_ts=None):
    data_ts = np.asarray(data_ts)
    injected_wf = np.asarray(injected_wf)
    psd_source = data_ts if psd_ref_ts is None else np.asarray(psd_ref_ts)
    # Analysis window: tail (to include injected signal)
    data_seg = _center_crop_or_pad(data_ts[-analysis_len:], analysis_len)
    inj_seg = _center_crop_or_pad(injected_wf[-analysis_len:], analysis_len)

    N = analysis_len
    win = tukey(N, 0.1)
    data_win = data_seg * win

    # Off-source PSD from the leading portion (avoid the injected tail). Use a
    # reference noise segment to keep the PSD estimate injection-independent.
    off_len = max(8, min(len(psd_source), analysis_len // 2))
    data_off = psd_source[:off_len]
    data_off_win = data_off * tukey(len(data_off), 0.1) if len(data_off) > 0 else data_off
    nper = min(512, len(data_off)) if len(data_off) > 0 else 256
    nper = max(nper, 8)
    nover = int(0.5 * nper)
    freqs_full, psd_full = welch(data_off_win, fs=fs, nperseg=nper, noverlap=nover)

    mask_welch = band_mask(freqs_full, BAND)
    freqs_plot, psd_plot = freqs_full[mask_welch], psd_full[mask_welch]

    f_data, Pxx_data = periodogram(data_win, fs=fs)
    mask_data = band_mask(f_data, BAND)

    f_sig, Pxx_sig = periodogram(inj_seg, fs=fs)
    mask_sig = band_mask(f_sig, BAND)

    U = np.mean(win**2)
    Neff = N * U
    C = np.sqrt(fs * Neff / 2.0)

    D_full = np.fft.rfft(data_win) / C
    f_rfft = np.fft.rfftfreq(N, 1.0 / fs)
    mask_r = band_mask(f_rfft, BAND)
    f_k = f_rfft[mask_r]
    D_k = D_full[mask_r]
    S_k = np.interp(f_k, freqs_full, psd_full, left=psd_full[0], right=psd_full[-1])
    S_k = np.clip(S_k, 1e-30, None)

    y_obs = np.stack([D_k.real, D_k.imag], axis=-1)
    sigma_comp = np.sqrt(S_k / 2.0)

    return {
        "N": N,
        "win": win,
        "C": C,
        "mask_r": mask_r,
        "sigma_comp": sigma_comp,
        "y_obs": y_obs,
        "D_k": D_k,
        "S_k": S_k,
        "f_k": f_k,
        "freqs_full": freqs_full,
        "psd_full": psd_full,
        "freqs_plot": freqs_plot,
        "psd_plot": psd_plot,
        "f_data": f_data[mask_data],
        "Pxx_data": Pxx_data[mask_data],
        "f_sig": f_sig[mask_sig],
        "Pxx_sig": Pxx_sig[mask_sig],
        "time_used": np.arange(N) / fs,
        "data_win": data_win,
        "injected_wf": injected_wf,
        "data_ts": data_ts,
    }


def rfft_model_waveform_scaled(z, amp, which, ctx):
    model = STARCCATO_SIGNAL if which == "signal" else STARCCATO_GLITCH
    wf = model.generate(z=z[None, :])[0]
    wf = wf * amp * REFERENCE_SCALE * (REFERENCE_DIST / 7.0)

    # Center-pad/crop waveform to the data length used in the likelihood.
    N = ctx["N"]
    L = wf.shape[0]
    if L >= N:
        wf = wf[:N]
    else:
        pad = jnp.zeros(N, dtype=wf.dtype)
        start = (N - L) // 2
        pad = pad.at[start : start + L].set(wf)
        wf = pad

    wf = wf * jnp.array(ctx["win"])
    Hf = jnp.fft.rfft(wf) / ctx["C"]
    return Hf[ctx["mask_r"]]


def make_whittle_model(which, ctx):
    def model(y=None):
        amp = numpyro.sample("amp", dist.LogNormal(AMP_LOGMEAN, AMP_LOGSIGMA))
        z = numpyro.sample("z", dist.Normal(0, 1).expand([N_LATENTS]))
        Hk = rfft_model_waveform_scaled(z, amp, which, ctx)
        mu = jnp.stack([jnp.real(Hk), jnp.imag(Hk)], axis=-1)
        numpyro.sample(
            "y",
            dist.Normal(mu, jnp.expand_dims(jnp.array(ctx["sigma_comp"]), -1)),
            obs=y,
        )

    return model


def _log_prior(amp, z):
    lp_amp = dist.LogNormal(AMP_LOGMEAN, AMP_LOGSIGMA).log_prob(amp)
    lp_z = jnp.sum(dist.Normal(0, 1).log_prob(z))
    return lp_amp + lp_z


def log_posterior_whittle(theta_vec, which, ctx):
    amp = jnp.asarray(theta_vec[0])
    if amp <= 0.0:
        return -jnp.inf
    z = jnp.asarray(theta_vec[1:])
    lp = _log_prior(amp, z)
    Hk = rfft_model_waveform_scaled(z, amp, which, ctx)
    mu = jnp.stack([jnp.real(Hk), jnp.imag(Hk)], axis=-1)
    sigma = jnp.asarray(ctx["sigma_comp"])
    resid = jnp.asarray(ctx["y_obs"]) - mu
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


def estimate_evidence_with_morph(samples, ll_total, which, ctx, outdir):
    theta_samples = np.concatenate(
        [np.array(samples["amp"])[..., None], np.array(samples["z"])], axis=1
    )
    log_post_vals = log_prior_samples(samples) + np.array(ll_total)
    param_names = ["amp"] + [f"z{i}" for i in range(N_LATENTS)]
    results = morph_evidence(
        post_samples=theta_samples,
        log_posterior_values=log_post_vals,
        log_posterior_function=lambda th: log_posterior_whittle(th, which, ctx),
        n_resamples=2000,
        morph_type="pair",
        kde_bw="silverman",
        param_names=param_names,
        output_path=str(outdir / f"morph_{which}"),
        n_estimations=1,
        verbose=False,
    )
    results = np.array(results)
    logz_mean = float(np.mean(results[:, 0]))
    logz_err = float(np.mean(results[:, 1]))
    print(f"morphZ LnZ[{which}] ≈ {logz_mean:.2f} ± {logz_err:.2f}")
    return logz_mean, logz_err


def matched_filter_snr(samples, ll_total, ctx):
    log_post_vals = log_prior_samples(samples) + np.array(ll_total)
    idx = int(np.argmax(log_post_vals))
    amp = float(samples["amp"][idx])
    z = np.array(samples["z"][idx])
    Hk = np.array(rfft_model_waveform_scaled(z, amp, "signal", ctx))
    num = np.sum(np.conj(Hk) * ctx["D_k"] / ctx["S_k"])
    den = np.sum(np.abs(Hk) ** 2 / ctx["S_k"])
    snr = float(np.real(num) / np.sqrt(max(den, 1e-30)))
    return snr


def matched_filter_snr_glitch(samples, ll_total, ctx):
    log_post_vals = log_prior_samples(samples) + np.array(ll_total)
    idx = int(np.argmax(log_post_vals))
    amp = float(samples["amp"][idx])
    z = np.array(samples["z"][idx])
    Hk = np.array(rfft_model_waveform_scaled(z, amp, "glitch", ctx))
    num = np.sum(np.conj(Hk) * ctx["D_k"] / ctx["S_k"])
    den = np.sum(np.abs(Hk) ** 2 / ctx["S_k"])
    snr = float(np.real(num) / np.sqrt(max(den, 1e-30)))
    return snr


def excess_power_snr(ctx):
    """cWB-like excess power SNR using the Whittle band."""
    return float(np.sqrt(np.sum(np.abs(ctx["D_k"]) ** 2 / ctx["S_k"])))


def run_model(model_fn, y, name, rng_key, num_warmup=500, num_samples=500):
    nuts = NUTS(model_fn)
    mcmc = MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, y=y)
    samples = mcmc.get_samples()
    ll = log_likelihood(model_fn, samples, y=y)["y"]
    ll_total = jnp.sum(ll, axis=(1, 2))
    return samples, np.array(ll_total)


# ----------------------------------------------------------------------
# Posterior predictive & plotting
# ----------------------------------------------------------------------
def posterior_draw_time_series(samples, ctx, which="signal", n_draws=20):
    draws = []
    n = min(n_draws, samples["amp"].shape[0])
    for i in range(n):
        amp = samples["amp"][i]
        z = samples["z"][i]
        Hk_scaled = rfft_model_waveform_scaled(z, amp, which, ctx)
        Hfull_scaled = jnp.zeros_like(jnp.fft.rfft(jnp.zeros(ctx["N"])))
        Hfull_scaled = Hfull_scaled.at[ctx["mask_r"]].set(Hk_scaled)
        Hfull_unscaled = Hfull_scaled * ctx["C"]
        ht = jnp.fft.irfft(Hfull_unscaled, n=ctx["N"])
        draws.append(np.asarray(ht))
    return np.array(draws)


def whiten(ts, psd_full, freqs_full, fs):
    """Simple frequency-domain whitening."""
    f_rfft = np.fft.rfftfreq(len(ts), 1.0 / fs)
    psd_interp = np.interp(f_rfft, freqs_full, psd_full, left=psd_full[0], right=psd_full[-1])
    ts_rfft = np.fft.rfft(ts)
    ts_white = np.fft.irfft(ts_rfft / np.sqrt(np.maximum(psd_interp, 1e-30)))
    return ts_white


def _psd_ci(draws):
    if len(draws) == 0:
        return None
    f, Pxx = periodogram(draws, fs=SAMPLE_RATE, axis=-1)
    m = band_mask(f, BAND)
    med = np.median(Pxx[:, m], axis=0)
    lo = np.percentile(Pxx[:, m], 5, axis=0)
    hi = np.percentile(Pxx[:, m], 95, axis=0)
    return f[m], med, lo, hi




# ----------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------
def main(detector=DEFAULT_DETECTOR, trigger_time=DEFAULT_TRIGGER_TIME, inject_kind=INJECT_KIND, seed=0, snr=100.0, snr_min=None, snr_max=None, cache_dir: Optional[Path] = None):
    outdir = OUTROOT / f"{int(trigger_time)}_{inject_kind}_seed{seed}"
    outdir.mkdir(parents=True, exist_ok=True)

    rng_master = jax.random.PRNGKey(seed)

    strain = fetch_strain(detector, trigger_time, DURATION, SAMPLE_RATE, cache_dir=cache_dir)
    data = strain.value
    time = strain.times.value
    fs = strain.sample_rate.value

    sos = butter(4, BAND, btype="band", fs=fs, output="sos")
    data_vis = sosfilt(sos, data)
    data_base = data_vis[-ANALYSIS_LEN:]
    snr_target = None
    if inject_kind == "noise":
        analysis_len = ANALYSIS_LEN
        injected_full = np.zeros_like(data_base)
        data_used_full = data_base.copy()
    else:
        # Default to uniform in [5, 100] for signals and [0, 100] for glitches if no range is provided.
        default_min, default_max = GLITCH_SNR_EXCESS_RANGE if inject_kind == "glitch" else DEFAULT_SNR_RANGE
        lo = default_min if snr_min is None else snr_min
        hi = default_max if snr_max is None else snr_max
        if hi > lo:
            snr_target = float(jax.random.uniform(rng_master, (), minval=lo, maxval=hi))
        else:
            snr_target = lo
        data_used_seg, injected_wf_seg, _ = inject_signal_into_data(data_base, inject_kind, snr_target, fs, sos)
        analysis_len = len(injected_wf_seg)
        injected_full = injected_wf_seg
        data_used_full = data_base + injected_full

    ctx = build_whittle_context(data_used_full, injected_full, fs, analysis_len, psd_ref_ts=data_base)

    snr_inband = None
    snr_excess_target = snr_target if inject_kind == "glitch" else None
    if inject_kind != "noise":
        snr_inband = _compute_inband_snr(injected_full, ctx)
        if inject_kind == "glitch":
            snr_excess = excess_power_snr(ctx)
            inj_excess = _compute_excess_power_snr_injection(injected_full, ctx)
            if snr_excess_target is not None and inj_excess > 0.0:
                scale = snr_excess_target / inj_excess
                injected_full = injected_full * scale
                data_used_full = data_base + injected_full
                ctx = build_whittle_context(data_used_full, injected_full, fs, analysis_len, psd_ref_ts=data_base)
                snr_inband = _compute_inband_snr(injected_full, ctx)
                snr_excess = excess_power_snr(ctx)
                inj_excess = _compute_excess_power_snr_injection(injected_full, ctx)
                print(
                    f"Rescaled glitch by {scale:.3g} to target excess-power SNR ≈ {snr_excess_target:.1f}; "
                    f"injection-only excess ≈ {inj_excess:.2f}; excess now ≈ {snr_excess:.2f}; in-band ≈ {snr_inband:.2f}"
                )
            else:
                print(
                    f"Glitch injection SNRs: excess≈{snr_excess:.2f} (target≈{snr_excess_target or 0:.2f}), "
                    f"injection-only excess≈{inj_excess:.2f}, in-band≈{snr_inband:.2f}"
                )
        else:
            if snr_target is not None and snr_inband > 0.0:
                scale = snr_target / snr_inband
                injected_full = injected_full * scale
                data_used_full = data_base + injected_full
                ctx = build_whittle_context(data_used_full, injected_full, fs, analysis_len, psd_ref_ts=data_base)
                snr_inband = _compute_inband_snr(injected_full, ctx)
                print(
                    f"Rescaled injection by {scale:.3g} to target in-band SNR ≈ {snr_target:.1f}; "
                    f"in-band now ≈ {snr_inband:.2f}"
                )
            else:
                print(f"In-band injection SNR (Whittle band) ≈ {snr_inband:.2f} (target ≈ {snr_target or 0:.2f})")
    rng_sig, rng_gli = jax.random.split(rng_master)
    model_sig = make_whittle_model("signal", ctx)
    model_gli = make_whittle_model("glitch", ctx)

    res_sig, ll_sig = run_model(model_sig, ctx["y_obs"], "signal", rng_sig)
    res_gli, ll_gli = run_model(model_gli, ctx["y_obs"], "glitch", rng_gli)

    print("\nEstimating evidences with morphZ...")
    lnz_sig_morph, lnz_sig_err = estimate_evidence_with_morph(res_sig, ll_sig, "signal", ctx, outdir)
    lnz_gli_morph, lnz_gli_err = estimate_evidence_with_morph(res_gli, ll_gli, "glitch", ctx, outdir)

    lnz_noise = -np.sum((np.abs(ctx["D_k"]) ** 2) / ctx["S_k"] + np.log(np.pi * ctx["S_k"]))
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
    print(f"log BF(signal | glitch/noise) ≈ {float(logBF_sig_alt):.2f}")
    print(f"Posterior probs (equal model priors): P(sig)={p_sig:.3f}, P(gli)={p_gli:.3f}, P(noise)={p_noise:.3f}")

    snr_mf_sig = matched_filter_snr(res_sig, ll_sig, ctx)
    snr_mf_gli = matched_filter_snr_glitch(res_gli, ll_gli, ctx)
    snr_excess = excess_power_snr(ctx)
    print(f"Matched-filter SNR (data|MAP signal) ≈ {snr_mf_sig:.2f}")
    print(f"Matched-filter SNR (data|MAP glitch) ≈ {snr_mf_gli:.2f}")
    print(f"Excess-power SNR (Whittle band) ≈ {snr_excess:.2f}")

    ts_sig = posterior_draw_time_series(res_sig, ctx, "signal", n_draws=20)
    ts_gli = posterior_draw_time_series(res_gli, ctx, "glitch", n_draws=20)

    lnBF_sig_noise = lnz_sig_morph - lnz_noise
    lnBF_sig_glitch = lnz_sig_morph - lnz_gli_morph

    plot_data(
        raw_data=ctx["data_ts"],
        injection=ctx["injected_wf"] if inject_kind != "noise" else None,
        welch_psd=(ctx["freqs_plot"], ctx["psd_plot"]),
        posterior_signal=np.array(ts_sig) if len(ts_sig) > 0 else None,
        posterior_blip=np.array(ts_gli) if len(ts_gli) > 0 else None,
        injection_snr=snr_inband if inject_kind != "noise" else None,
        logBF=float(logBF_sig_alt),
        outpath=outdir / "frequency_summary.png",
    )

    sig_ci = _psd_ci(ts_sig)
    gli_ci = _psd_ci(ts_gli)


    # Save metrics
    metrics_path = outdir / "metrics.csv"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("detector,gps,inject,seed,target_snr,achieved_snr,lnz_noise,lnz_signal,lnz_signal_err,lnz_glitch,lnz_glitch_err,logBF_sig_alt,snr_mf_sig,snr_mf_glitch,snr_excess\n")
        f.write(
            f"{detector},{trigger_time},{inject_kind},{seed},"
            f"{snr_target if inject_kind!='noise' else ''},"
            f"{lnz_noise:.6f},{lnz_sig_morph:.6f},{lnz_sig_err:.6f},"
            f"{lnz_gli_morph:.6f},{lnz_gli_err:.6f},"
            f"{float(logBF_sig_alt):.6f},{snr_mf_sig:.6f},{snr_mf_gli:.6f},{snr_excess:.6f}\n"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="One-detector Starccato analysis")
    parser.add_argument("--detector", default=DEFAULT_DETECTOR, help="Detector (e.g., H1)")
    parser.add_argument("--gps", type=float, default=DEFAULT_TRIGGER_TIME, help="GPS start time")
    parser.add_argument("--inject", choices=["signal", "glitch", "noise"], default=INJECT_KIND, help="Injection kind")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--snr", type=float, default=SNR_INJECTION, help="Target injection SNR (used if no range)")
    parser.add_argument("--snr-min", type=float, default=None, help="Min SNR (if set with --snr-max, sample uniform)")
    parser.add_argument("--snr-max", type=float, default=None, help="Max SNR (if set with --snr-min, sample uniform)")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=str(DEFAULT_CACHE_DIR),
        help="Directory containing GWOSC HDF5 files or where new caches will be stored.",
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    main(
        detector=args.detector,
        trigger_time=args.gps,
        inject_kind=args.inject,
        seed=args.seed,
        snr=args.snr,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        cache_dir=cache_dir,
    )
