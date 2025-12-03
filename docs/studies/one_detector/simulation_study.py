"""
Simple simulation study:
 1) Train/load 6-D VAEs for CCSNe (signal) and blip (glitch) once and cache them.
 2) Simulate Gaussian noise with flat PSD and optionally inject signal/glitch.
 3) Run NUTS with the Whittle likelihood.
 4) Estimate evidence with morphZ.
 5) Plot prior/posterior predictive (freq + time).
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

import arviz as az
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.signal.windows import tukey
from starccato_jax import Config, StarccatoVAE
from morphZ import evidence as morph_evidence

jax.config.update("jax_enable_x64", True)

HERE = Path(__file__).parent
OUTROOT = HERE / "sim_out"
OUTROOT.mkdir(parents=True, exist_ok=True)

# Simulation / analysis settings
FS = 4096.0
N = 512
BAND = (100.0, 1024.0)
WINDOW = jnp.array(tukey(N, 0.25))
U = jnp.mean(WINDOW ** 2)
Neff = N * U
C = jnp.sqrt(FS * Neff / 2.0)
WINDOW_NP = np.asarray(WINDOW)
FREQ = np.fft.rfftfreq(N, 1.0 / FS)
FMASK = np.array((FREQ >= BAND[0]) & (FREQ <= BAND[1]))
FMASK_JAX = jnp.array(FMASK, dtype=bool)

# Priors
AMP_LOGMEAN = -2.0
AMP_LOGSIGMA = 2.0
REFERENCE_SCALE = 1e-20

# Model cache dirs
MODEL_DIR_SIG = HERE / "models" / "ccsne_z6"
MODEL_DIR_GLI = HERE / "models" / "blip_z6"


def snr_from_excess_power(x: np.ndarray, psd_vals: np.ndarray, fs=FS) -> float:
    """Compute excess-power SNR of a time series given a PSD."""
    Xf = np.abs(np.fft.rfft(x * WINDOW_NP) / float(C))[FMASK]
    psd_arr = np.asarray(psd_vals)
    if psd_arr.shape[0] == FREQ.shape[0]:
        Sf = psd_arr[FMASK]
    else:  # assume already masked
        Sf = psd_arr
    return float(np.sqrt(np.sum(Xf**2 / Sf)))


def _psd_consistent(series: np.ndarray):
    X = np.fft.rfft(series * WINDOW_NP, axis=-1) / float(C)
    Pxx = np.abs(X) ** 2 * 2.0
    return FREQ[FMASK], Pxx[..., FMASK]


def load_or_train(model_dir: Path, dataset: str) -> StarccatoVAE:
    model_dir = Path(model_dir)
    # Avoid retraining if weights already exist (config.json is not persisted)
    if (model_dir / "model.h5").exists():
        return StarccatoVAE(str(model_dir))
    model_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config(latent_dim=6, epochs=300, dataset=dataset)
    StarccatoVAE.train(model_dir=str(model_dir), config=cfg, plot_every=np.inf)
    return StarccatoVAE(str(model_dir))


def simulate_noise(psd_level: float, key) -> np.ndarray:
    rng = np.random.default_rng(int(key[0]))
    sigma = np.sqrt(psd_level)
    return rng.normal(0.0, sigma, size=N)


def sample_injection(model: StarccatoVAE, key) -> np.ndarray:
    rng_amp, rng_z = jax.random.split(key)
    amp = float(dist.LogNormal(AMP_LOGMEAN, AMP_LOGSIGMA).sample(rng_amp))
    z = dist.Normal(0, 1).sample(rng_z, (6,))
    wf = model.generate(z=z[None, :])[0]
    wf = np.array(wf, dtype=np.float64)
    return wf * amp * REFERENCE_SCALE


def build_context(data: np.ndarray, injection: np.ndarray, psd_level: float):
    data_win = data * WINDOW_NP
    inj_win = injection * WINDOW_NP
    D_full = np.fft.rfft(data_win) / float(C)
    PSD_k = np.full_like(FREQ[FMASK], psd_level)
    y_obs = jnp.stack([jnp.real(jnp.asarray(D_full[FMASK])), jnp.imag(jnp.asarray(D_full[FMASK]))], axis=-1)
    sigma_comp = jnp.sqrt(jnp.asarray(PSD_k) / 2.0)
    lnz_noise = float(-jnp.sum((jnp.abs(jnp.asarray(D_full[FMASK])) ** 2) / PSD_k + jnp.log(jnp.pi * PSD_k)))
    _, Pxx_data = _psd_consistent(data_win)
    _, Pxx_inj = _psd_consistent(inj_win)
    return {
        "y_obs": y_obs,
        "sigma_comp": sigma_comp,
        "PSD_k": jnp.asarray(PSD_k),
        "f_k": jnp.asarray(FREQ[FMASK]),
        "Pxx_data": Pxx_data,
        "Pxx_inj": Pxx_inj,
        "lnz_noise": lnz_noise,
    }


def rfft_model_waveform(model: StarccatoVAE, z, amp):
    wf = model.generate(z=z[None, :])[0]
    wf = wf * amp * REFERENCE_SCALE
    return (jnp.fft.rfft(wf * WINDOW) / C)[FMASK_JAX]


def log_likelihood_whittle(model, z, amp, ctx):
    Hk_scaled = rfft_model_waveform(model, z, amp)
    Hk = jnp.stack([jnp.real(Hk_scaled), jnp.imag(Hk_scaled)], axis=-1)
    residual = ctx["y_obs"] - Hk
    sigma = jnp.expand_dims(ctx["sigma_comp"], -1)
    return -0.5 * jnp.sum((residual / sigma) ** 2 + 2.0 * jnp.log(sigma) + jnp.log(2.0 * jnp.pi))


def make_model(model_obj, ctx):
    def model():
        amp = numpyro.sample("amp", dist.LogNormal(AMP_LOGMEAN, AMP_LOGSIGMA))
        z = numpyro.sample("z", dist.Normal(0, 1).expand([6]))
        ll = log_likelihood_whittle(model_obj, z, amp, ctx)
        numpyro.factor("whittle_ll", ll)

    return model


def run_nuts(model_fn, ctx, rng_key, num_warmup, num_samples):
    nuts = NUTS(model_fn, dense_mass=True)
    mcmc = MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples)
    mcmc.run(rng_key)
    samples = mcmc.get_samples()
    loglike_fn = jax.vmap(lambda a, z: log_likelihood_whittle(model_fn.model_obj, z, a, ctx))
    ll_total = np.array(loglike_fn(samples["amp"], samples["z"]))
    ess = az.ess(az.from_numpyro(mcmc), method="bulk").to_array().values
    print(f"ESS min={ess.min():.1f}, max={ess.max():.1f}")
    return samples, ll_total


def estimate_evidence(samples, ll_total, which, ctx, outdir, n_resamples):
    theta_samples = np.concatenate([np.array(samples["amp"])[..., None], np.array(samples["z"])], axis=1)

    def log_post_single(th):
        th = jnp.asarray(th)
        amp = th[0]
        z = th[1:]
        lp = dist.LogNormal(AMP_LOGMEAN, AMP_LOGSIGMA).log_prob(amp) + jnp.sum(dist.Normal(0, 1).log_prob(z))
        return lp + log_likelihood_whittle(which.model_obj, z, amp, ctx)

    log_post_vals = np.array(jax.vmap(log_post_single)(jnp.array(theta_samples)))
    param_names = ["amp"] + [f"z{i}" for i in range(6)]
    out = f"{outdir}/morph_{which.name}"
    results = morph_evidence(
        post_samples=theta_samples,
        log_posterior_values=log_post_vals,
        log_posterior_function=lambda th: float(log_post_single(th)),
        n_resamples=n_resamples,
        morph_type="pair",
        kde_bw="silverman",
        param_names=param_names,
        output_path=out,
        n_estimations=1,
        verbose=False,
    )
    results = np.array(results)
    logz_mean = float(np.mean(results[:, 0]))
    logz_err = float(np.mean(results[:, 1]))
    print(f"morphZ LnZ[{which.name}] ≈ {logz_mean:.2f} ± {logz_err:.2f}")
    return logz_mean, logz_err


def plot_summary(ctx, injection, post_sig, post_gli, inj_type, logBF, outdir):
    f_k = ctx["f_k"]
    plt.figure(figsize=(8, 5))
    plt.loglog(f_k, ctx["PSD_k"], color="k", lw=1.2, label="Noise PSD (flat)")
    plt.loglog(f_k, ctx["Pxx_data"], color="0.6", alpha=0.4, label="Data periodogram")
    if inj_type != "noise":
        plt.loglog(f_k, ctx["Pxx_inj"], color="tab:orange", label="Injection")

    def _plot(label, color, series):
        if series is None or len(series) == 0:
            return
        _, Pxx = _psd_consistent(series)
        med = np.median(Pxx, axis=0)
        lo = np.percentile(Pxx, 10, axis=0)
        hi = np.percentile(Pxx, 90, axis=0)
        plt.fill_between(f_k, lo, hi, color=color, alpha=0.2)
        plt.loglog(f_k, med, color=color, lw=1.1, label=label)

    _plot("Posterior signal", "tab:red", post_sig)
    _plot("Posterior glitch", "tab:green", post_gli)
    plt.xlim(*BAND)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD [1/Hz]")
    title = f"SNR≈{(np.sqrt(np.sum(ctx['Pxx_inj'])) if inj_type != 'noise' else 0):.1f} | inj={inj_type}"
    if logBF is not None and not np.isnan(logBF):
        title += f" | logBF≈{logBF:.1f}"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "frequency_summary.png", dpi=150)
    plt.close()


def main(args):
    outdir = OUTROOT / f"seed{args.seed}_{args.inject}"
    outdir.mkdir(parents=True, exist_ok=True)

    rng = jax.random.PRNGKey(args.seed)
    rng_inj, rng_sig, rng_gli = jax.random.split(rng, 3)

    sig_model = load_or_train(MODEL_DIR_SIG, "ccsne")
    gli_model = load_or_train(MODEL_DIR_GLI, "blip")

    noise = simulate_noise(psd_level=args.psd_level, key=rng)
    if args.inject == "signal":
        injection = sample_injection(sig_model, rng_inj)
    elif args.inject == "glitch":
        injection = sample_injection(gli_model, rng_inj)
    else:
        injection = np.zeros_like(noise)
    data = noise + injection

    ctx = build_context(data, injection, psd_level=args.psd_level)
    inj_snr = snr_from_excess_power(injection, ctx["PSD_k"])
    data_snr = snr_from_excess_power(data, ctx["PSD_k"])

    model_sig = make_model(sig_model, ctx)
    model_sig.model_obj = sig_model
    model_sig.name = "signal"
    model_gli = make_model(gli_model, ctx)
    model_gli.model_obj = gli_model
    model_gli.name = "glitch"

    res_sig, ll_sig = run_nuts(model_sig, ctx, rng_sig, args.num_warmup, args.num_samples)
    res_gli, ll_gli = run_nuts(model_gli, ctx, rng_gli, args.num_warmup, args.num_samples)

    lnz_sig, lnz_sig_err = estimate_evidence(res_sig, ll_sig, model_sig, ctx, outdir, args.morph_resamples)
    lnz_gli, lnz_gli_err = estimate_evidence(res_gli, ll_gli, model_gli, ctx, outdir, args.morph_resamples)
    lnz_noise = float(ctx["lnz_noise"])
    logZ_alt = jax.scipy.special.logsumexp(jnp.array([lnz_gli, lnz_noise]))
    logBF = lnz_sig - logZ_alt
    print(f"logBF(signal|gli/noise) ≈ {logBF:.2f}")

    # Draw posterior waveforms
    def posterior_draws(samples, model_obj, n_draws=50):
        draws = []
        n = min(n_draws, samples["amp"].shape[0])
        for i in range(n):
            amp = samples["amp"][i]
            z = samples["z"][i]
            Hk_scaled = rfft_model_waveform(model_obj, z, amp)
            Hfull = jnp.zeros_like(jnp.fft.rfft(jnp.zeros(N)))
            Hfull = Hfull.at[FMASK_JAX].set(Hk_scaled)
            ht = jnp.fft.irfft(Hfull * C, n=N)
            draws.append(np.asarray(ht))
        return np.array(draws)

    post_sig_ts = posterior_draws(res_sig, sig_model)
    post_gli_ts = posterior_draws(res_gli, gli_model)
    plot_summary(ctx, injection, post_sig_ts, post_gli_ts, args.inject, float(logBF), outdir)

    metrics = {
        "inject": args.inject,
        "seed": args.seed,
        "excess_snr_injection": inj_snr,
        "excess_snr_data": data_snr,
        "lnz_noise": lnz_noise,
        "lnz_signal": lnz_sig,
        "lnz_signal_err": lnz_sig_err,
        "lnz_glitch": lnz_gli,
        "lnz_glitch_err": lnz_gli_err,
        "logBF": float(logBF),
    }
    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulation study (noise + optional injection)")
    parser.add_argument("--inject", choices=["signal", "glitch", "noise"], default="signal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--psd-level", type=float, default=1e-46, help="Flat PSD level (1/Hz)")
    parser.add_argument("--num-warmup", type=int, default=300)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--morph-resamples", type=int, default=75)
    args = parser.parse_args()
    # main(args)

    # do the above for 100 seeds, injecting either signal, glitch, or noise
    for seed in range(100):
        for inject_type in ["signal", "glitch", "noise"]:
            args.seed = seed
            args.inject = inject_type
            main(args)
