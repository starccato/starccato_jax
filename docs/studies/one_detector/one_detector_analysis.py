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

jax.config.update("jax_enable_x64", True)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
HERE = Path(__file__).parent.resolve()
OUTROOT = HERE / "out"
OUTROOT.mkdir(parents=True, exist_ok=True)
CACHE_DIR = HERE / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_TRIGGER_TIME = 1186741733  # quiet segment
DEFAULT_DETECTOR = "H1"
SAMPLE_RATE = 4096.0
BAND = (100.0, 1024.0)
DURATION = 4.0

N_LATENTS = 32
REFERENCE_DIST = 10.0
REFERENCE_SCALE = 1e-21

STARCCATO_SIGNAL = StarccatoCCSNe()
STARCCATO_GLITCH = StarccatoBlip()

INJECT_KIND = "signal"  # default, overridden by main args
SNR_INJECTION = 100.0


# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------
def band_mask(f: np.ndarray, band: tuple[float, float]) -> np.ndarray:
    return (f >= band[0]) & (f <= band[1])


def fetch_strain(detector, trigger_time, duration, sample_rate):
    cache_file = CACHE_DIR / f"{detector}_{trigger_time}_{duration}s_{int(sample_rate)}Hz.hdf5"
    if cache_file.exists():
        print(f"Loading cached data from {cache_file}...")
        strain = TimeSeries.read(cache_file)
    else:
        print(f"Fetching {detector} data around GPS {trigger_time}...")
        strain = TimeSeries.fetch_open_data(
            detector,
            trigger_time - 2,
            trigger_time + duration + 2,
            sample_rate=sample_rate,
        ).crop(trigger_time, trigger_time + duration)
        print(f"Saving to cache: {cache_file}")
        strain.write(cache_file)
    return strain


def gen_waveform(latents: np.ndarray, amp: float = 1.0, distance: float = 7.0, model=None):
    model = STARCCATO_SIGNAL if model is None else model
    wf = model.generate(z=latents[None, :])[0]
    wf = np.array(wf, dtype=np.float64, copy=True)
    return wf * amp * REFERENCE_SCALE * (REFERENCE_DIST / distance)


def inject_signal_into_data(raw_data: np.ndarray, kind: str, snr: float, fs: float, sos):
    """Inject Starccato signal/glitch with a target optimal SNR in analysis band."""
    z = np.zeros(N_LATENTS)
    model = STARCCATO_SIGNAL if kind == "signal" else STARCCATO_GLITCH
    wf0 = gen_waveform(z, model=model)

    N_wf = min(len(wf0), len(raw_data))
    wf0 = wf0[:N_wf]
    start = (len(raw_data) - N_wf) // 2
    data_crop = raw_data[start : start + N_wf]

    win_local = tukey(N_wf, 0.1)
    data_crop_win = data_crop * win_local
    freqs_full_local, psd_full_local = welch(data_crop_win, fs=fs, nperseg=256)

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
    rho = np.sqrt(max(rho2, 1e-30))

    scale = snr / rho
    wf_scaled = sosfilt(sos, wf0) * scale
    injected = data_crop + wf_scaled
    return injected, wf_scaled


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


# ----------------------------------------------------------------------
# Whittle helpers
# ----------------------------------------------------------------------
def build_whittle_context(data_win, injected_wf, fs):
    N = len(data_win)
    win = tukey(N, 0.1)
    data_win = data_win * win

    freqs_full, psd_full = welch(data_win, fs=fs, nperseg=256)
    mask_welch = band_mask(freqs_full, BAND)
    freqs_plot, psd_plot = freqs_full[mask_welch], psd_full[mask_welch]

    f_data, Pxx_data = periodogram(data_win, fs=fs)
    mask_data = band_mask(f_data, BAND)

    f_sig, Pxx_sig = periodogram(injected_wf, fs=fs)
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
    }


def rfft_model_waveform_scaled(z, amp, which, ctx):
    model = STARCCATO_SIGNAL if which == "signal" else STARCCATO_GLITCH
    wf = model.generate(z=z[None, :])[0]
    wf = wf * amp * REFERENCE_SCALE * (REFERENCE_DIST / 7.0)
    wf = wf * jnp.array(ctx["win"])
    Hf = jnp.fft.rfft(wf) / ctx["C"]
    return Hf[ctx["mask_r"]]


def make_whittle_model(which, ctx):
    def model(y=None):
        amp = numpyro.sample("amp", dist.LogNormal(0.0, 0.5))
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
    lp_amp = dist.LogNormal(0.0, 0.5).log_prob(amp)
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
    lp_amp = dist.LogNormal(0.0, 0.5).log_prob(amp)
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


def run_model(model_fn, y, name, rng_key):
    nuts = NUTS(model_fn)
    mcmc = MCMC(nuts, num_warmup=2000, num_samples=2000)
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


def plot_frequency_summary(f_data, P_data, psd_used, f_inj, P_inj, sig_ci, gli_ci, outpath):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(f_data, P_data, color="k", lw=1.2, label="Data periodogram")
    ax.loglog(psd_used[0], psd_used[1], color="0.3", lw=1.0, ls="--", label="PSD (Whittle)")
    if f_inj is not None and P_inj is not None:
        ax.loglog(f_inj, P_inj, color="tab:orange", lw=1.0, alpha=0.8, label="Injection")
    if sig_ci is not None:
        f_s, med_s, lo_s, hi_s = sig_ci
        ax.fill_between(f_s, lo_s, hi_s, color="tab:red", alpha=0.25, label="Signal 90% CI")
        ax.loglog(f_s, med_s, color="tab:red", lw=1.1, label="Signal median")
    if gli_ci is not None:
        f_g, med_g, lo_g, hi_g = gli_ci
        ax.fill_between(f_g, lo_g, hi_g, color="tab:green", alpha=0.25, label="Glitch 90% CI")
        ax.loglog(f_g, med_g, color="tab:green", lw=1.1, label="Glitch median")
    ax.set_xlim(*BAND)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("PSD [1/Hz]")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)
    print(f"Saved frequency-domain summary to {outpath}")


def plot_summary(time, data_ts, injected_ts, freqs, psd, posterior_sig_ts, posterior_gli_ts, title, snr=None, lnBF_sig_noise=None, lnBF_sig_glitch=None, injected_raw=None):
    fig, (ax_t, ax_p) = plt.subplots(2, 1, figsize=(9, 6), gridspec_kw={"height_ratios": [1, 1.2]})
    ax_t.plot(time, data_ts, color="k", lw=0.8, label="Data")
    if injected_ts is not None:
        ax_t.plot(time, injected_ts, color="tab:orange", lw=1, label="Injection")
    for draw in posterior_sig_ts:
        ax_t.plot(time, draw, color="tab:red", alpha=0.25)
    for draw in posterior_gli_ts:
        ax_t.plot(time, draw, color="tab:green", alpha=0.20)
    ax_t.set_ylabel("Strain")
    ax_t.legend(fontsize=9)
    txt = []
    if snr is not None:
        txt.append(f"SNR ≈ {snr:.1f}")
    if lnBF_sig_noise is not None:
        txt.append(f"ΔLnZ(sig–noise) ≈ {lnBF_sig_noise:.1f}")
    if lnBF_sig_glitch is not None:
        txt.append(f"ΔLnZ(sig–gli) ≈ {lnBF_sig_glitch:.1f}")
    if txt:
        ax_t.text(
            0.02,
            0.9,
            "".join(txt),
            transform=ax_t.transAxes,
            fontsize=10,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.3", fc="w", alpha=0.5),
        )
    ax_t.set_title(title)

    ax_p.loglog(freqs, psd, "k", lw=1.2, label="Noise PSD (Welch)")
    if injected_raw is not None:
        f_sig_inj, Pxx_sig_inj = periodogram(injected_raw, fs=SAMPLE_RATE)
        m_sig = band_mask(f_sig_inj, BAND)
        ax_p.loglog(f_sig_inj[m_sig], Pxx_sig_inj[m_sig], "tab:orange", alpha=0.7, label="Injection PSD")
    if len(posterior_sig_ts) > 0:
        f_s, Pxx_s = periodogram(posterior_sig_ts, fs=SAMPLE_RATE, axis=-1)
        m_s = band_mask(f_s, BAND)
        med_s = np.median(Pxx_s[:, m_s], axis=0)
        lo_s = np.percentile(Pxx_s[:, m_s], 5, axis=0)
        hi_s = np.percentile(Pxx_s[:, m_s], 95, axis=0)
        ax_p.fill_between(f_s[m_s], lo_s, hi_s, color="tab:red", alpha=0.25, label="Signal 90% CI")
        ax_p.loglog(f_s[m_s], med_s, color="tab:red", lw=1.2, label="Signal median")
    if len(posterior_gli_ts) > 0:
        f_g, Pxx_g = periodogram(posterior_gli_ts, fs=SAMPLE_RATE, axis=-1)
        m_g = band_mask(f_g, BAND)
        med_g = np.median(Pxx_g[:, m_g], axis=0)
        lo_g = np.percentile(Pxx_g[:, m_g], 5, axis=0)
        hi_g = np.percentile(Pxx_g[:, m_g], 95, axis=0)
        ax_p.fill_between(f_g[m_g], lo_g, hi_g, color="tab:green", alpha=0.25, label="Glitch 90% CI")
        ax_p.loglog(f_g[m_g], med_g, color="tab:green", lw=1.2, label="Glitch median")
    ax_p.set_xlim(*BAND)
    ax_p.set_xlabel("Frequency [Hz]")
    ax_p.set_ylabel("PSD [1/Hz]")
    ax_p.legend(fontsize=9)
    ax_p.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Main flow
# ----------------------------------------------------------------------
def main(detector=DEFAULT_DETECTOR, trigger_time=DEFAULT_TRIGGER_TIME, inject_kind=INJECT_KIND, seed=0):
    outdir = OUTROOT / f"{int(trigger_time)}_{inject_kind}"
    outdir.mkdir(parents=True, exist_ok=True)

    rng_master = jax.random.PRNGKey(seed)

    strain = fetch_strain(detector, trigger_time, DURATION, SAMPLE_RATE)
    data = strain.value
    time = strain.times.value
    fs = strain.sample_rate.value

    sos = butter(4, BAND, btype="band", fs=fs, output="sos")
    data_vis = sosfilt(sos, data)
    initial_plot(time, data_vis, fs, detector, outdir)

    if inject_kind == "noise":
        data_used = data_vis.copy()
        injected_wf = np.zeros_like(data_used)
    else:
        data_used, injected_wf = inject_signal_into_data(data_vis, inject_kind, SNR_INJECTION, fs, sos)

    ctx = build_whittle_context(data_used, injected_wf, fs)

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

    snr_mf = matched_filter_snr(res_sig, ll_sig, ctx)
    print(f"Matched-filter SNR (data|MAP signal) ≈ {snr_mf:.2f}")

    ts_sig = posterior_draw_time_series(res_sig, ctx, "signal", n_draws=20)
    ts_gli = posterior_draw_time_series(res_gli, ctx, "glitch", n_draws=20)

    fig_pre = plot_summary(
        ctx["time_used"],
        ctx["data_win"],
        ctx["injected_wf"],
        ctx["freqs_plot"],
        ctx["psd_plot"],
        [],
        [],
        title="Before analysis (time domain)",
        injected_raw=ctx["injected_wf"] if inject_kind != "noise" else None,
    )
    fig_pre.savefig(outdir / "before_analysis.png", dpi=150)
    plt.close(fig_pre)

    H_inj_scaled = np.fft.rfft(ctx["injected_wf"] * ctx["win"]) / ctx["C"]
    snr_est = np.sqrt(np.sum(np.abs(H_inj_scaled[ctx["mask_r"]]) ** 2 / ctx["S_k"])) if inject_kind != "noise" else None
    lnBF_sig_noise = lnz_sig_morph - lnz_noise
    lnBF_sig_glitch = lnz_sig_morph - lnz_gli_morph

    fig_post = plot_summary(
        ctx["time_used"],
        ctx["data_win"],
        ctx["injected_wf"],
        ctx["freqs_plot"],
        ctx["psd_plot"],
        ts_sig,
        ts_gli,
        title="Posterior predictive (time domain)",
        snr=snr_est,
        lnBF_sig_noise=lnBF_sig_noise,
        lnBF_sig_glitch=lnBF_sig_glitch,
        injected_raw=ctx["injected_wf"] if inject_kind != "noise" else None,
    )
    fig_post.savefig(outdir / "posterior_predictive.png", dpi=150)
    plt.close(fig_post)

    sig_ci = _psd_ci(ts_sig)
    gli_ci = _psd_ci(ts_gli)
    out_freq_plot = outdir / "frequency_summary.png"
    plot_frequency_summary(
        ctx["f_data"],
        ctx["Pxx_data"],
        (ctx["freqs_plot"], ctx["psd_plot"]),
        ctx["f_sig"] if inject_kind != "noise" else None,
        ctx["Pxx_sig"] if inject_kind != "noise" else None,
        sig_ci,
        gli_ci,
        out_freq_plot,
    )

    # Save metrics
    metrics_path = outdir / "metrics.csv"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("detector,gps,inject,seed,lnz_noise,lnz_signal,lnz_signal_err,lnz_glitch,lnz_glitch_err,logBF_sig_alt,snr_mf\n")
        f.write(
            f"{detector},{trigger_time},{inject_kind},{seed},"
            f"{lnz_noise:.6f},{lnz_sig_morph:.6f},{lnz_sig_err:.6f},"
            f"{lnz_gli_morph:.6f},{lnz_gli_err:.6f},"
            f"{float(logBF_sig_alt):.6f},{snr_mf:.6f}\n"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="One-detector Starccato analysis")
    parser.add_argument("--detector", default=DEFAULT_DETECTOR, help="Detector (e.g., H1)")
    parser.add_argument("--gps", type=float, default=DEFAULT_TRIGGER_TIME, help="GPS start time")
    parser.add_argument("--inject", choices=["signal", "glitch", "noise"], default=INJECT_KIND, help="Injection kind")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    main(
        detector=args.detector,
        trigger_time=args.gps,
        inject_kind=args.inject,
        seed=args.seed,
    )
