#!/usr/bin/env python
"""
One-detector analysis using a known PSD:
- Load a design PSD (default: bilby aLIGO O4 ASD)
- Simulate Gaussian noise from that PSD
- Optionally inject a Starccato signal/glitch with target SNR (using the same PSD)
- Run NumPyro Whittle likelihood with the known PSD
- Report morphZ evidences, BFs, matched-filter SNRs, and excess-power SNR
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram
from scipy.signal.windows import tukey

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, log_likelihood
from starccato_jax.waveforms import StarccatoCCSNe, StarccatoBlip
from morphZ import evidence as morph_evidence

jax.config.update("jax_enable_x64", True)

# -------------------------
# Config
# -------------------------
SAMPLE_RATE = 4096.0
DURATION = 4.0
BAND = (100.0, 1024.0)
N_LATENTS = 32
REFERENCE_DIST = 10.0
REFERENCE_SCALE = 1e-21
BASE_WF_LEN = len(StarccatoCCSNe().generate(z=np.zeros((1, N_LATENTS)))[0])
AMP_LOGMEAN = -30.756
AMP_LOGSIGMA = 0.932

STARCCATO_SIGNAL = StarccatoCCSNe()
STARCCATO_GLITCH = StarccatoBlip()


# -------------------------
# Helpers
# -------------------------
def band_mask(f, band):
    return (f >= band[0]) & (f <= band[1])


def load_design_psd(detector: str, asd_file: str | None = None):
    if asd_file:
        path = Path(asd_file).expanduser()
    else:
        import site
        candidates = []
        for p in site.getsitepackages():
            candidates.append(Path(p) / "bilby" / "gw" / "detector" / "noise_curves" / "aLIGO_O4_high_asd.txt")
        path = next((c for c in candidates if c.exists()), None)
        if path is None:
            raise FileNotFoundError("Could not find aLIGO_O4_high_asd.txt; pass --asd-file.")
    data = np.loadtxt(path)
    freqs = data[:, 0]
    asd = data[:, 1]
    psd = asd**2
    return freqs, psd


def simulate_noise(psd_f, psd_vals, n, fs, rng):
    dt = 1.0 / fs
    df = fs / n
    freqs = np.fft.rfftfreq(n, dt)
    psd_interp = np.interp(freqs, psd_f, psd_vals, left=psd_vals[0], right=psd_vals[-1])
    amp = np.sqrt(0.5 * psd_interp * df)
    re = rng.normal(size=freqs.shape) * amp
    im = rng.normal(size=freqs.shape) * amp
    im[0] = 0.0
    if n % 2 == 0:
        im[-1] = 0.0
    ntilde = re + 1j * im
    noise = np.fft.irfft(ntilde, n=n)
    return noise


def rfft_model_waveform_scaled(z, amp, which, ctx):
    model = STARCCATO_SIGNAL if which == "signal" else STARCCATO_GLITCH
    wf = model.generate(z=z[None, :])[0]
    wf = wf * amp * REFERENCE_SCALE * (REFERENCE_DIST / 7.0)
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
        numpyro.sample("y", dist.Normal(mu, jnp.expand_dims(jnp.array(ctx["sigma_comp"]), -1)), obs=y)

    return model


def log_prior_samples(samples):
    amp = jnp.asarray(samples["amp"])
    z = jnp.asarray(samples["z"])
    lp_amp = dist.LogNormal(AMP_LOGMEAN, AMP_LOGSIGMA).log_prob(amp)
    lp_z = jnp.sum(dist.Normal(0, 1).log_prob(z), axis=1)
    return np.array(lp_amp + lp_z)


def log_posterior_whittle(theta_vec, which, ctx):
    amp = jnp.asarray(theta_vec[0])
    if amp <= 0.0:
        return -jnp.inf
    z = jnp.asarray(theta_vec[1:])
    lp_amp = dist.LogNormal(AMP_LOGMEAN, AMP_LOGSIGMA).log_prob(amp)
    lp_z = jnp.sum(dist.Normal(0, 1).log_prob(z))
    lp = lp_amp + lp_z
    Hk = rfft_model_waveform_scaled(z, amp, which, ctx)
    mu = jnp.stack([jnp.real(Hk), jnp.imag(Hk)], axis=-1)
    sigma = jnp.asarray(ctx["sigma_comp"])
    resid = jnp.asarray(ctx["y_obs"]) - mu
    ll = -0.5 * jnp.sum((resid / sigma[:, None]) ** 2 + 2.0 * jnp.log(sigma[:, None]) + jnp.log(2.0 * jnp.pi))
    return float(lp + ll)


def estimate_evidence_with_morph(samples, ll_total, which, ctx, outdir):
    theta_samples = np.concatenate([np.array(samples["amp"])[..., None], np.array(samples["z"])], axis=1)
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


def matched_filter_snr(samples, ll_total, ctx, which="signal"):
    log_post_vals = log_prior_samples(samples) + np.array(ll_total)
    idx = int(np.argmax(log_post_vals))
    amp = float(samples["amp"][idx])
    z = np.array(samples["z"][idx])
    Hk = np.array(rfft_model_waveform_scaled(z, amp, which, ctx))
    num = np.sum(np.conj(Hk) * ctx["D_k"] / ctx["S_k"])
    den = np.sum(np.abs(Hk) ** 2 / ctx["S_k"])
    snr = float(np.real(num) / np.sqrt(max(den, 1e-30)))
    return snr


def excess_power_snr(ctx):
    return float(np.sqrt(np.sum(np.abs(ctx["D_k"]) ** 2 / ctx["S_k"])))


def inject_signal(psd_f, psd_vals, kind, target_snr, fs, rng):
    z = np.zeros(N_LATENTS)
    model = STARCCATO_SIGNAL if kind == "signal" else STARCCATO_GLITCH
    wf0 = np.array(model.generate(z=z[None, :])[0], dtype=np.float64)
    wf0 = wf0[:BASE_WF_LEN]
    win = tukey(len(wf0), 0.1)
    wf_win = wf0 * win
    hf = np.fft.rfft(wf_win) * (1.0 / fs)
    freqs = np.fft.rfftfreq(len(wf0), 1.0 / fs)
    psd_interp = np.interp(freqs, psd_f, psd_vals, left=psd_vals[0], right=psd_vals[-1])
    df = freqs[1] - freqs[0]
    rho2 = 4.0 * np.sum(np.abs(hf) ** 2 / np.maximum(psd_interp, 1e-30)) * df
    rho_ref = np.sqrt(max(rho2, 1e-30))
    scale = target_snr / rho_ref
    wf_scaled = wf0 * scale
    return wf_scaled, target_snr


def build_context(data_ts, injected_wf, psd_f, psd_vals, fs, analysis_len):
    data_seg = data_ts[-analysis_len:]
    inj_seg = injected_wf[-analysis_len:]
    win = tukey(analysis_len, 0.1)
    data_win = data_seg * win

    freqs_full = psd_f
    psd_full = psd_vals
    mask_welch = band_mask(freqs_full, BAND)
    freqs_plot, psd_plot = freqs_full[mask_welch], psd_full[mask_welch]

    f_data, Pxx_data = periodogram(data_win, fs=fs)
    mask_data = band_mask(f_data, BAND)

    f_sig, Pxx_sig = periodogram(inj_seg, fs=fs)
    mask_sig = band_mask(f_sig, BAND)

    U = np.mean(win**2)
    Neff = analysis_len * U
    C = np.sqrt(fs * Neff / 2.0)

    D_full = np.fft.rfft(data_win) / C
    f_rfft = np.fft.rfftfreq(analysis_len, 1.0 / fs)
    mask_r = band_mask(f_rfft, BAND)
    f_k = f_rfft[mask_r]
    psd_interp = np.interp(f_k, freqs_full, psd_full, left=psd_full[0], right=psd_full[-1])
    D_k = D_full[mask_r]
    S_k = psd_interp

    y_obs = np.stack([D_k.real, D_k.imag], axis=-1)
    sigma_comp = np.sqrt(S_k / 2.0)

    return {
        "N": analysis_len,
        "win": win,
        "C": C,
        "mask_r": mask_r,
        "sigma_comp": sigma_comp,
        "y_obs": y_obs,
        "D_k": D_k,
        "S_k": S_k,
        "freqs_plot": freqs_plot,
        "psd_plot": psd_plot,
        "f_data": f_data[mask_data],
        "Pxx_data": Pxx_data[mask_data],
        "f_sig": f_sig[mask_sig],
        "Pxx_sig": Pxx_sig[mask_sig],
        "time_used": np.arange(analysis_len) / fs,
        "data_ts": data_seg,
        "injected_wf": inj_seg,
    }


def run_model(model_fn, y, rng_key, num_warmup=500, num_samples=500):
    nuts = NUTS(model_fn)
    mcmc = MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key, y=y)
    samples = mcmc.get_samples()
    ll = log_likelihood(model_fn, samples, y=y)["y"]
    ll_total = jnp.sum(ll, axis=(1, 2))
    return samples, np.array(ll_total)


def plot_frequency_summary(f_data, P_data, f_inj, P_inj, sig_ci, gli_ci, outpath):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(f_data, P_data, color="k", lw=1.2, label="Data periodogram")
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


def _psd_ci(draws):
    if len(draws) == 0:
        return None
    f, Pxx = periodogram(draws, fs=SAMPLE_RATE, axis=-1)
    m = band_mask(f, BAND)
    med = np.median(Pxx[:, m], axis=0)
    lo = np.percentile(Pxx[:, m], 5, axis=0)
    hi = np.percentile(Pxx[:, m], 95, axis=0)
    return f[m], med, lo, hi


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


# -------------------------
# Main
# -------------------------
def main(detector="H1", inject_kind="signal", seed=0, snr=20.0, snr_min=None, snr_max=None, asd_file=None):
    outdir = Path(__file__).parent / "out_known_psd"
    outdir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    psd_f, psd_vals = load_design_psd(detector, asd_file)

    N_total = int(DURATION * SAMPLE_RATE)
    noise = simulate_noise(psd_f, psd_vals, N_total, SAMPLE_RATE, rng)

    if inject_kind == "noise":
        injected_wf = np.zeros(BASE_WF_LEN)
        snr_target = None
    else:
        if snr_min is not None and snr_max is not None and snr_max > snr_min:
            snr_target = float(rng.uniform(snr_min, snr_max))
        else:
            snr_target = snr
        wf_scaled, _ = inject_signal(psd_f, psd_vals, inject_kind, snr_target, SAMPLE_RATE, rng)
        injected_wf = _center_crop_or_pad(wf_scaled, BASE_WF_LEN)
        # place at end
        start = N_total - BASE_WF_LEN
        noise[start : start + BASE_WF_LEN] += injected_wf

    analysis_len = BASE_WF_LEN
    ctx = build_context(noise, injected_wf, psd_f, psd_vals, SAMPLE_RATE, analysis_len)

    rng_sig, rng_gli = jax.random.split(jax.random.PRNGKey(seed))
    model_sig = make_whittle_model("signal", ctx)
    model_gli = make_whittle_model("glitch", ctx)

    res_sig, ll_sig = run_model(model_sig, ctx["y_obs"], rng_sig)
    res_gli, ll_gli = run_model(model_gli, ctx["y_obs"], rng_gli)

    lnz_sig_morph, lnz_sig_err = estimate_evidence_with_morph(res_sig, ll_sig, "signal", ctx, outdir)
    lnz_gli_morph, lnz_gli_err = estimate_evidence_with_morph(res_gli, ll_gli, "glitch", ctx, outdir)
    lnz_noise = -np.sum((np.abs(ctx["D_k"]) ** 2) / ctx["S_k"] + np.log(np.pi * ctx["S_k"]))

    print(f"LnZ[noise] (analytic)  = {lnz_noise:.2f}")
    print(f"morphZ LnZ[signal]     = {lnz_sig_morph:.2f} ± {lnz_sig_err:.2f}")
    print(f"morphZ LnZ[glitch]     = {lnz_gli_morph:.2f} ± {lnz_gli_err:.2f}")
    logZ_alt = jax.scipy.special.logsumexp(jnp.array([lnz_gli_morph, lnz_noise]))
    logBF_sig_alt = lnz_sig_morph - logZ_alt
    print(f"log BF(signal | glitch/noise) ≈ {float(logBF_sig_alt):.2f}")

    snr_mf_sig = matched_filter_snr(res_sig, ll_sig, ctx, "signal")
    snr_mf_gli = matched_filter_snr(res_gli, ll_gli, ctx, "glitch")
    snr_excess = excess_power_snr(ctx)
    print(f"Matched-filter SNR (signal) ≈ {snr_mf_sig:.2f}")
    print(f"Matched-filter SNR (glitch) ≈ {snr_mf_gli:.2f}")
    print(f"Excess-power SNR ≈ {snr_excess:.2f}")
    if inject_kind != "noise":
        print(f"Target injection SNR ≈ {snr_target:.2f}")

    ts_sig = posterior_draw_time_series(res_sig, ctx, "signal", n_draws=20)
    ts_gli = posterior_draw_time_series(res_gli, ctx, "glitch", n_draws=20)

    sig_ci = _psd_ci(ts_sig)
    gli_ci = _psd_ci(ts_gli)
    out_freq_plot = outdir / f"frequency_summary_{inject_kind}.png"
    plot_frequency_summary(
        ctx["f_data"],
        ctx["Pxx_data"],
        ctx["f_sig"] if inject_kind != "noise" else None,
        ctx["Pxx_sig"] if inject_kind != "noise" else None,
        sig_ci,
        gli_ci,
        out_freq_plot,
    )

    metrics_path = outdir / f"metrics_{inject_kind}.csv"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("detector,inject,seed,target_snr,lnz_noise,lnz_signal,lnz_signal_err,lnz_glitch,lnz_glitch_err,logBF_sig_alt,snr_mf_sig,snr_mf_glitch,snr_excess\n")
        f.write(
            f"{detector},{inject_kind},{seed},"
            f"{snr_target if inject_kind!='noise' else ''},"
            f"{lnz_noise:.6f},{lnz_sig_morph:.6f},{lnz_sig_err:.6f},"
            f"{lnz_gli_morph:.6f},{lnz_gli_err:.6f},"
            f"{float(logBF_sig_alt):.6f},{snr_mf_sig:.6f},{snr_mf_gli:.6f},{snr_excess:.6f}\n"
        )
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="One-detector analysis with known PSD and simulated noise.")
    parser.add_argument("--detector", default="H1", help="Detector name (H1/L1/V1)")
    parser.add_argument("--inject", choices=["signal", "glitch", "noise"], default="signal", help="Injection kind")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--snr", type=float, default=20.0, help="Target injection SNR (used if no range)")
    parser.add_argument("--snr-min", type=float, default=None, help="Min SNR for uniform draw")
    parser.add_argument("--snr-max", type=float, default=None, help="Max SNR for uniform draw")
    parser.add_argument("--asd-file", type=str, default=None, help="Path to ASD file (f, ASD); defaults to bilby aLIGO O4")
    args = parser.parse_args()

    main(
        detector=args.detector,
        inject_kind=args.inject,
        seed=args.seed,
        snr=args.snr,
        snr_min=args.snr_min,
        snr_max=args.snr_max,
        asd_file=args.asd_file,
    )
