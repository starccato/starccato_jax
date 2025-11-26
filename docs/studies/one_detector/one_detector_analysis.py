# %% [markdown]
# Starccato + NumPyro with Whittle likelihood, unified plots, and SNR/Bayes factor annotation.

# %%
# !pip install gwpy numpyro starccato-jax matplotlib morphZ scipy -q

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, periodogram, butter, sosfilt
from scipy.signal.windows import tukey
from gwpy.timeseries import TimeSeries

import jax, jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, log_likelihood
from starccato_jax.waveforms import StarccatoCCSNe, StarccatoBlip
from morphZ import evidence as morph_evidence
import os
from pathlib import Path

HERE = Path(__file__).parent.resolve()
OUTDIR = HERE / "out"

# -------------------------------
# Configuration
# -------------------------------
jax.config.update("jax_enable_x64", True)
rng_key = jax.random.PRNGKey(0)

trigger_time = 1186741733      # quiet segment (noise)
duration = 4.0
detector = "H1"
sample_rate = 4096.0
band = (100.0, 1024.0)

N_LATENTS = 32
REFERENCE_DIST = 10.0
REFERENCE_SCALE = 1e-21

STARCCATO_SIGNAL = StarccatoCCSNe()
STARCCATO_GLITCH = StarccatoBlip()

def band_mask(f, band):
    return (f >= band[0]) & (f <= band[1])

# -------------------------------
# Fetch data (GWPy) with caching
# -------------------------------
# Create cache directory
cache_dir = HERE / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)
cache_file = cache_dir / f"{detector}_{trigger_time}_{duration}s_{int(sample_rate)}Hz.hdf5"

if cache_file.exists():
    print(f"Loading cached data from {cache_file}...")
    strain = TimeSeries.read(cache_file)
else:
    print(f"Fetching {detector} data around GPS {trigger_time}...")
    strain = TimeSeries.fetch_open_data(
        detector, trigger_time - 2, trigger_time + duration + 2, sample_rate=sample_rate
    ).crop(trigger_time, trigger_time + duration)
    # Save to cache
    print(f"Saving to cache: {cache_file}")
    strain.write(cache_file)

data = strain.value
time = strain.times.value
FS = strain.sample_rate.value

# Band-pass for visualization only
sos = butter(4, band, btype="band", fs=FS, output="sos")
data_vis = sosfilt(sos, data)

plt.figure(figsize=(10, 3))
plt.plot(time - time[0], data_vis, lw=0.6)
plt.title(f"{detector} strain (band-limited {band[0]}–{band[1]} Hz)")
plt.xlabel("Time [s]"); plt.ylabel("Strain"); plt.tight_layout(); plt.show()

# -------------------------------
# Injection setup
# -------------------------------
def gen_waveform(latents, amp=1.0, distance=7.0, model=None):
    if model is None:
        model = STARCCATO_SIGNAL
    wf = model.generate(z=latents[None, :])[0]
    wf = np.array(wf, dtype=np.float64, copy=True)
    return wf * amp * REFERENCE_SCALE * (REFERENCE_DIST / distance)

def inject_signal_into_data(raw_data, kind="signal", snr=8.0):
    """Inject a Starccato signal/glitch with *target optimal SNR* in the analysis band.
    SNR is computed with the same PSD-consistent scaling used in the Whittle likelihood:
        rho^2 ≈ sum_k |H_k|^2 / S_k  (one-sided, rfft, masked bins)
    """
    # Base waveform (centered crop to match data length later)
    z = np.zeros(N_LATENTS)
    model = STARCCATO_SIGNAL if kind == "signal" else STARCCATO_GLITCH
    wf0 = gen_waveform(z, model=model)

    # Crop data to waveform length (centered)
    N_wf = min(len(wf0), len(raw_data))
    wf0 = wf0[:N_wf]
    start = (len(raw_data) - N_wf) // 2
    data_crop = raw_data[start:start + N_wf]

    # Window used for analysis (same shape as later)
    win_local = tukey(N_wf, 0.1)

    # PSD for this segment (Welch on windowed noise-only data)
    data_crop_win = data_crop * win_local
    freqs_full_local, psd_full_local = welch(data_crop_win, fs=FS, nperseg=256)

    # PSD-consistent scaling factor C for this length
    U_local = np.mean(win_local**2)
    C_local = np.sqrt(FS * (N_wf * U_local) / 2.0)

    # RFFT bins and band mask
    f_rfft_local = np.fft.rfftfreq(N_wf, 1.0/FS)
    mask_local = band_mask(f_rfft_local, band)

    # Compute current SNR of the (windowed) waveform in PSD units
    wf_win = sosfilt(sos, wf0) * win_local
    Hk_scaled = np.fft.rfft(wf_win) / C_local
    S_k_local = np.interp(f_rfft_local[mask_local], freqs_full_local, psd_full_local,
                          left=psd_full_local[0], right=psd_full_local[-1])
    rho2 = np.sum(np.abs(Hk_scaled[mask_local])**2 / S_k_local)
    rho = np.sqrt(max(rho2, 1e-30))

    # Scale waveform to achieve target SNR in the analysis band
    scale = snr / rho
    wf_scaled = sosfilt(sos, wf0) * scale

    # Inject into the *uncwindowed* data crop (windowing happens later in analysis)
    injected = data_crop + wf_scaled
    return injected, wf_scaled
    model = STARCCATO_SIGNAL if kind == "signal" else STARCCATO_GLITCH
    wf = gen_waveform(z, model=model)
    wf = sosfilt(sos, wf)
    wf *= tukey(len(wf), 0.1)
    wf /= np.sqrt(np.mean(wf**2))
    wf *= snr * np.std(raw_data)
    N_wf = len(wf)
    start = (len(raw_data) - N_wf) // 2
    injected = raw_data[start:start + N_wf] + wf
    return injected, wf

inject_kind = "signal"  # "noise", "signal", or "glitch"
snr_injection = 100.0

if inject_kind == "noise":
    data_used = data_vis.copy()
    injected_wf = np.zeros_like(data_used)
else:
    data_used, injected_wf = inject_signal_into_data(data_vis, inject_kind, snr_injection)

N = len(data_used)
win = tukey(N, 0.1)
data_win = data_used * win
time_used = np.arange(N) / FS

# -------------------------------
# PSD diagnostics (Welch/Periodogram)
# -------------------------------
freqs_full, psd_full = welch(data_win, fs=FS, nperseg=256)
mask_welch = band_mask(freqs_full, band)
freqs_plot, psd_plot = freqs_full[mask_welch], psd_full[mask_welch]

f_data, Pxx_data = periodogram(data_win, fs=FS)
mask_data = band_mask(f_data, band)
f_sig, Pxx_sig = periodogram(injected_wf, fs=FS)
mask_sig = band_mask(f_sig, band)

# -------------------------------
# Whittle likelihood prep
# -------------------------------
U = np.mean(win**2)
Neff = N * U
C = np.sqrt(FS * Neff / 2.0)  # scale factor -> PSD-consistent units

D_full = np.fft.rfft(data_win) / C
f_rfft = np.fft.rfftfreq(N, 1.0 / FS)
mask_r = band_mask(f_rfft, band)
f_k = f_rfft[mask_r]
D_k = D_full[mask_r]
S_k = np.interp(f_k, freqs_full, psd_full, left=psd_full[0], right=psd_full[-1])

y_obs = np.stack([D_k.real, D_k.imag], axis=-1)
sigma_comp = np.sqrt(S_k / 2.0)

# -------------------------------
# Models (Whittle, PSD-consistent)
# -------------------------------
def rfft_model_waveform_scaled(z, amp, which="signal"):
    if which == "signal":
        wf = STARCCATO_SIGNAL.generate(z=z[None, :])[0]
    else:
        wf = STARCCATO_GLITCH.generate(z=z[None, :])[0]
    wf = wf * amp * REFERENCE_SCALE * (REFERENCE_DIST / 7.0)
    wf = wf * jnp.array(win)
    Hf = jnp.fft.rfft(wf) / C
    return Hf[mask_r]

def model_whittle_signal(y=None):
    amp = numpyro.sample("amp", dist.LogNormal(0.0, 0.5))
    z = numpyro.sample("z", dist.Normal(0, 1).expand([N_LATENTS]))
    Hk = rfft_model_waveform_scaled(z, amp, "signal")
    mu = jnp.stack([jnp.real(Hk), jnp.imag(Hk)], axis=-1)
    numpyro.sample("y", dist.Normal(mu, jnp.expand_dims(jnp.array(sigma_comp), -1)), obs=y)

def model_whittle_glitch(y=None):
    amp = numpyro.sample("amp", dist.LogNormal(0.0, 0.5))
    z = numpyro.sample("z", dist.Normal(0, 1).expand([N_LATENTS]))
    Hk = rfft_model_waveform_scaled(z, amp, "glitch")
    mu = jnp.stack([jnp.real(Hk), jnp.imag(Hk)], axis=-1)
    numpyro.sample("y", dist.Normal(mu, jnp.expand_dims(jnp.array(sigma_comp), -1)), obs=y)


def _log_prior(amp, z):
    """Log prior for shared parameterization (amp + z vector)."""
    lp_amp = dist.LogNormal(0.0, 0.5).log_prob(amp)
    lp_z = jnp.sum(dist.Normal(0, 1).log_prob(z))
    return lp_amp + lp_z


def log_posterior_whittle(theta_vec, which="signal"):
    """Unnormalized log-posterior used by morphZ."""
    amp = jnp.asarray(theta_vec[0])
    if amp <= 0.0:  # outside LogNormal support
        return -jnp.inf
    z = jnp.asarray(theta_vec[1:])
    lp = _log_prior(amp, z)
    Hk = rfft_model_waveform_scaled(z, amp, which)
    mu = jnp.stack([jnp.real(Hk), jnp.imag(Hk)], axis=-1)
    sigma = jnp.asarray(sigma_comp)
    resid = jnp.asarray(y_obs) - mu
    ll = -0.5 * jnp.sum(
        (resid / sigma[:, None]) ** 2
        + 2.0 * jnp.log(sigma[:, None])
        + jnp.log(2.0 * jnp.pi)
    )
    return float(lp + ll)


def log_prior_samples(samples):
    """Vectorized log prior for MCMC samples dict."""
    amp = jnp.asarray(samples["amp"])
    z = jnp.asarray(samples["z"])
    lp_amp = dist.LogNormal(0.0, 0.5).log_prob(amp)
    lp_z = jnp.sum(dist.Normal(0, 1).log_prob(z), axis=1)
    return np.array(lp_amp + lp_z)


def estimate_evidence_with_morph(samples, ll_total, which):
    """Estimate LnZ with morphZ using posterior samples and log-posterior callback."""
    theta_samples = np.concatenate(
        [np.array(samples["amp"])[..., None], np.array(samples["z"])], axis=1
    )
    log_post_vals = log_prior_samples(samples) + np.array(ll_total)
    param_names = ["amp"] + [f"z{i}" for i in range(N_LATENTS)]
    results = morph_evidence(
        post_samples=theta_samples,
        log_posterior_values=log_post_vals,
        log_posterior_function=lambda th: log_posterior_whittle(th, which),
        n_resamples=2000,
        morph_type="tree",
        kde_bw="isj",
        param_names=param_names,
        output_path=f"package/docs/studies/one_detector/morph_{which}",
        n_estimations=2,
        verbose=True,
    )
    results = np.array(results)
    logz_mean = float(np.mean(results[:, 0]))
    logz_err = float(np.mean(results[:, 1]))
    print(f"morphZ LnZ[{which}] ≈ {logz_mean:.2f} ± {logz_err:.2f}")
    return logz_mean, logz_err


# -------------------------------
# Inference
# -------------------------------
def run_model(model, y, name):
    nuts = NUTS(model)
    mcmc = MCMC(nuts, num_warmup=400, num_samples=800)
    mcmc.run(rng_key, y=y)
    samples = mcmc.get_samples()
    ll = log_likelihood(model, samples, y=y)["y"]
    ll_total = jnp.sum(ll, axis=(1, 2))
    lnz = jax.scipy.special.logsumexp(ll_total) - jnp.log(ll_total.shape[0])
    print(f"Whittle LnZ[{name}] ≈ {float(lnz):.2f}")
    return samples, np.array(ll_total), float(lnz)

res_sig, ll_sig, lnz_sig = run_model(model_whittle_signal, y_obs, "signal")
res_gli, ll_gli, lnz_gli = run_model(model_whittle_glitch, y_obs, "glitch")

lnz_sig_morph, lnz_sig_err = estimate_evidence_with_morph(res_sig, ll_sig, "signal")
lnz_gli_morph, lnz_gli_err = estimate_evidence_with_morph(res_gli, ll_gli, "glitch")

lnz_noise = -np.sum((np.abs(D_k)**2) / S_k + np.log(np.pi * S_k))
print(f"Whittle LnZ[noise]   = {lnz_noise:.2f}")
print(f"ΔLnZ(signal–noise)   = {lnz_sig - lnz_noise:.2f}")
print(f"ΔLnZ(signal–glitch)  = {lnz_sig - lnz_gli:.2f}")
print(f"morphZ LnZ[noise]    = {lnz_noise:.2f} (analytic)")
print(f"morphZ ΔLnZ(sig–noise)  ≈ {lnz_sig_morph - lnz_noise:.2f} ± {lnz_sig_err:.2f}")
print(f"morphZ ΔLnZ(sig–glitch) ≈ {lnz_sig_morph - lnz_gli_morph:.2f} ± {np.hypot(lnz_sig_err, lnz_gli_err):.2f}")

# -------------------------------
# Posterior predictive
# -------------------------------
def posterior_draw_time_series(samples, which="signal", n_draws=20):
    draws = []
    n = min(n_draws, samples["amp"].shape[0])
    for i in range(n):
        amp = samples["amp"][i]
        z = samples["z"][i]
        Hk_scaled = rfft_model_waveform_scaled(z, amp, which)  # Hk is scaled by 1/C
        Hfull_scaled = jnp.zeros_like(jnp.fft.rfft(jnp.zeros(N)))
        Hfull_scaled = Hfull_scaled.at[mask_r].set(Hk_scaled)
        # Convert back to unscaled FFT domain for time-series prediction (match injected/data units)
        Hfull_unscaled = Hfull_scaled * C
        ht = jnp.fft.irfft(Hfull_unscaled, n=N)
        draws.append(np.asarray(ht))
    return np.array(draws)

ts_sig = posterior_draw_time_series(res_sig, "signal", n_draws=20)
ts_gli = posterior_draw_time_series(res_gli, "glitch", n_draws=20)

# -------------------------------
# Plot helper: time + PSD combined
# -------------------------------
def plot_summary(time, data, injected, freqs, psd, posterior_sig, posterior_gli, title, snr=None, lnBF_sig_noise=None, lnBF_sig_glitch=None):
    fig, (ax_t, ax_p) = plt.subplots(2, 1, figsize=(9,6), gridspec_kw={'height_ratios':[1,1.2]})
    # --- time ---
    ax_t.plot(time, data, color='k', lw=0.8, label='Data')
    if injected is not None:
        ax_t.plot(time, injected, color='tab:orange', lw=1, label='Injected')
    for draw in posterior_sig:
        ax_t.plot(time, draw, color='tab:red', alpha=0.25)
    for draw in posterior_gli:
        ax_t.plot(time, draw, color='tab:green', alpha=0.20)
    ax_t.set_ylabel("Strain"); ax_t.legend(fontsize=9)
    txt = []
    if snr is not None: txt.append(f"SNR ≈ {snr:.1f}")
    if lnBF_sig_noise is not None: txt.append(f"ΔLnZ(sig–noise) ≈ {lnBF_sig_noise:.1f}")
    if lnBF_sig_glitch is not None: txt.append(f"ΔLnZ(sig–gli) ≈ {lnBF_sig_glitch:.1f}")
    if txt:
        ax_t.text(0.02, 0.9, "".join(txt), transform=ax_t.transAxes,
                  fontsize=10, va='top', ha='left',
                  bbox=dict(boxstyle="round,pad=0.3", fc='w', alpha=0.5))
    ax_t.set_title(title)
    # --- PSD ---
    ax_p.loglog(freqs, psd, 'k', lw=1.2, label='Noise PSD')
    if injected is not None:
        f_sig_inj, Pxx_sig_inj = periodogram(injected, fs=FS)
        m_sig = band_mask(f_sig_inj, band)
        ax_p.loglog(f_sig_inj[m_sig], Pxx_sig_inj[m_sig], 'tab:orange', alpha=0.7, label='Injected')
    # signal posterior
    if len(posterior_sig) > 0:
        f_s, Pxx_s = periodogram(posterior_sig, fs=FS, axis=-1)
        m_s = band_mask(f_s, band)
        med_s = np.median(Pxx_s[:, m_s], axis=0)
        lo_s = np.percentile(Pxx_s[:, m_s], 5, axis=0)
        hi_s = np.percentile(Pxx_s[:, m_s], 95, axis=0)
        ax_p.fill_between(f_s[m_s], lo_s, hi_s, color='tab:red', alpha=0.25, label='Signal 90% CI')
        ax_p.loglog(f_s[m_s], med_s, color='tab:red', lw=1.2, label='Signal median')
    # glitch posterior
    if len(posterior_gli) > 0:
        f_g, Pxx_g = periodogram(posterior_gli, fs=FS, axis=-1)
        m_g = band_mask(f_g, band)
        med_g = np.median(Pxx_g[:, m_g], axis=0)
        lo_g = np.percentile(Pxx_g[:, m_g], 5, axis=0)
        hi_g = np.percentile(Pxx_g[:, m_g], 95, axis=0)
        ax_p.fill_between(f_g[m_g], lo_g, hi_g, color='tab:green', alpha=0.25, label='Glitch 90% CI')
        ax_p.loglog(f_g[m_g], med_g, color='tab:green', lw=1.2, label='Glitch median')
    ax_p.set_xlim(*band); ax_p.set_xlabel("Frequency [Hz]"); ax_p.set_ylabel("PSD [1/Hz]")
    ax_p.legend(fontsize=9); ax_p.grid(alpha=0.3)
    fig.tight_layout()
    return fig

# -------------------------------
# Plot: before analysis (no posteriors)
# -------------------------------
plot_summary(
    time_used, data_win, injected_wf, freqs_plot, psd_plot,
    [], [], title="Before analysis"
)


# -------------------------------
# Plot: after analysis (posterior)
# -------------------------------
# Use PSD-consistent scaling for SNR: window + C, restricted to masked bins
H_inj_scaled = np.fft.rfft(injected_wf * win) / C
snr_est = np.sqrt(np.sum(np.abs(H_inj_scaled[mask_r])**2 / S_k))
lnBF_sig_noise = lnz_sig - lnz_noise
lnBF_sig_glitch = lnz_sig - lnz_gli

plot_summary(
    time_used, data_win, injected_wf, freqs_plot, psd_plot,
    ts_sig, ts_gli, title="Posterior predictive (signal & glitch)",
    snr=snr_est, lnBF_sig_noise=lnBF_sig_noise, lnBF_sig_glitch=lnBF_sig_glitch
)
