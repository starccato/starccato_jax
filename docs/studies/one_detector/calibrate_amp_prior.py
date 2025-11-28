#!/usr/bin/env python
"""
Estimate the base SNR distribution of Starccato waveforms against a design PSD,
then derive a log-normal amplitude prior to span a target SNR range.

Usage:
  python calibrate_amp_prior.py --n-draws 500 --snr-min 3 --snr-max 100
Optional:
  --asd-file /path/to/asd.txt   # two-column f, ASD; defaults to bilby aLIGO_O4_high_asd.txt
  --sample-rate 4096            # Hz
  --n-samples 512               # waveform length used for SNR calc
Outputs:
  - Prints base SNR stats and suggested LogNormal(mu, sigma) for amplitude
  - Saves base SNR draws to out/base_snr_samples.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal.windows import tukey

from starccato_jax.waveforms import StarccatoCCSNe


def find_default_asd():
    """Locate bilby aLIGO O4 ASD shipped with the environment."""
    import site
    candidates = []
    for p in site.getsitepackages():
        candidates.append(Path(p) / "bilby" / "gw" / "detector" / "noise_curves" / "aLIGO_O4_high_asd.txt")
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Could not find default aLIGO_O4_high_asd.txt; please pass --asd-file.")


def load_psd(asd_path: Path):
    data = np.loadtxt(asd_path)
    freqs = data[:, 0]
    psd = data[:, 1] ** 2
    return freqs, psd


def waveform_samples(model, latent_dim, n_samples, n_draws, seed):
    rng = np.random.default_rng(seed)
    zs = rng.normal(size=(n_draws, latent_dim))
    wfs = []
    for z in zs:
        wf = np.array(model.generate(z=z[None, :])[0], dtype=np.float64)
        if len(wf) > n_samples:
            start = (len(wf) - n_samples) // 2
            wf = wf[start : start + n_samples]
        elif len(wf) < n_samples:
            pad = np.zeros(n_samples, dtype=wf.dtype)
            start = (n_samples - len(wf)) // 2
            pad[start : start + len(wf)] = wf
            wf = pad
        wfs.append(wf)
    return np.array(wfs)


def snr_optimal(wf, psd_f, psd_vals, fs):
    """One-sided optimal SNR for real wf using standard GW convention."""
    n = len(wf)
    dt = 1.0 / fs
    win = tukey(n, 0.1)
    wf_win = wf * win
    hf = np.fft.rfft(wf_win) * dt
    freqs = np.fft.rfftfreq(n, dt)
    psd_interp = np.interp(freqs, psd_f, psd_vals, left=psd_vals[0], right=psd_vals[-1])
    df = freqs[1] - freqs[0]
    rho2 = 4.0 * np.sum(np.abs(hf) ** 2 / np.maximum(psd_interp, 1e-30)) * df
    return np.sqrt(max(rho2, 0.0))


def suggest_lognormal(q5, q95, target_low, target_high):
    """Solve LogNormal(mu, sigma) so that q5 -> target_low and q95 -> target_high."""
    lnq5 = np.log(target_low / q5)
    lnq95 = np.log(target_high / q95)
    z = 1.6448536269514722  # 95th percentile of standard normal
    sigma = (lnq95 - lnq5) / (2 * z)
    mu = lnq5 + z * sigma
    return mu, sigma


def main():
    parser = argparse.ArgumentParser(description="Calibrate amplitude prior via design PSD SNRs.")
    parser.add_argument("--n-draws", type=int, default=500, help="Number of waveform draws")
    parser.add_argument("--snr-min", type=float, default=3.0, help="Target 5th percentile SNR")
    parser.add_argument("--snr-max", type=float, default=100.0, help="Target 95th percentile SNR")
    parser.add_argument("--sample-rate", type=float, default=4096.0, help="Sample rate [Hz]")
    parser.add_argument("--n-samples", type=int, default=512, help="Number of samples used for SNR calc")
    parser.add_argument("--asd-file", type=str, default=None, help="Path to ASD file (f, ASD)")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--out-dir", type=str, default="out", help="Directory to write outputs")
    args = parser.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    asd_path = Path(args.asd_file) if args.asd_file else find_default_asd()
    psd_f, psd_vals = load_psd(asd_path)

    model = StarccatoCCSNe()
    latent_dim = model.latent_dim
    wfs = waveform_samples(model, latent_dim, args.n_samples, args.n_draws, args.seed)

    snrs = np.array([snr_optimal(wf, psd_f, psd_vals, args.sample_rate) for wf in wfs])
    q5, q50, q95 = np.percentile(snrs, [5, 50, 95])
    mu, sigma = suggest_lognormal(q5, q95, args.snr_min, args.snr_max)

    print(f"Base SNR stats (amp=1): q5={q5:.2f}, median={q50:.2f}, q95={q95:.2f}")
    print(f"To map q5->{args.snr_min} and q95->{args.snr_max}, use LogNormal(mu={mu:.3f}, sigma={sigma:.3f}) on amplitude.")
    print("Check: implied SNR 5/50/95% ~ "
          f"{q5*np.exp(mu-1.645*sigma):.1f}/"
          f"{q50*np.exp(mu):.1f}/"
          f"{q95*np.exp(mu+1.645*sigma):.1f}")

    np.savetxt(outdir / "base_snr_samples.csv", snrs, delimiter=",")
    print(f"Saved base SNR samples to {outdir / 'base_snr_samples.csv'}")


if __name__ == "__main__":
    main()
