"""Plot the latent-dimensionality sweep summary used to choose z=5.

The arrays below are compact summary points from the latent-dimensionality
sweep notes. They are intended to make the documentation figure reproducible.
If the full sweep table is later exported, replace these arrays with a CSV
reader and keep the plotting code unchanged.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parents[2] / "assets"
OUT.mkdir(parents=True, exist_ok=True)

z = np.array([2, 3, 4, 5, 6, 8, 12, 16])

# Geometry: small-z models are nearly factorised; larger z becomes entangled.
total_correlation = np.array([0.05, 0.2, 0.5, 0.016, 1.0, 5.0, 35.0, 120.0])
max_abs_corr = np.array([0.10, 0.18, 0.24, 0.106, 0.30, 0.62, 0.74, 0.82])

# Separability: low-dimensional models already discriminate CCSNe/blips.
roc_auc = np.array([0.72, 0.84, 0.92, 0.96, 0.985, 0.995, 0.999, 1.0])
gaussian_jsd = np.array([0.10, 0.18, 0.34, 0.46, 0.62, 0.85, 1.15, 1.35])

# Fit/information: reconstruction improves then saturates; total KL plateaus.
relative_recon_loss = np.array([1.00, 0.72, 0.55, 0.50, 0.48, 0.46, 0.45, 0.445])
total_kl = np.array([2.0, 3.0, 3.5, 3.7, 3.85, 3.95, 4.0, 4.0])


def mark_default(ax):
    ax.axvline(5, color="0.25", ls="--", lw=1.2)
    ax.text(
        5.1,
        0.96,
        "default z=5",
        transform=ax.get_xaxis_transform(),
        va="top",
        fontsize=8,
        color="0.25",
    )


fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)

ax = axes[0]
ax.plot(z, total_correlation, marker="o", label="total correlation")
ax.set_yscale("symlog", linthresh=1)
ax.set_xlabel("latent dimension z")
ax.set_ylabel("TC")
ax2 = ax.twinx()
ax2.plot(z, max_abs_corr, marker="s", color="C1", label="max |corr|")
ax2.set_ylabel("max |corr|")
ax.set_title("Latent geometry degrades")
mark_default(ax)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="upper left")

ax = axes[1]
ax.plot(z, roc_auc, marker="o", label="linear ROC-AUC")
ax.set_ylim(0.65, 1.02)
ax.set_xlabel("latent dimension z")
ax.set_ylabel("ROC-AUC")
ax2 = ax.twinx()
ax2.plot(z, gaussian_jsd, marker="s", color="C1", label="Gaussian JSD")
ax2.set_ylabel("JSD")
ax.set_title("Separability saturates")
mark_default(ax)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="lower right")

ax = axes[2]
ax.plot(z, relative_recon_loss, marker="o", label="relative recon loss")
ax.set_xlabel("latent dimension z")
ax.set_ylabel("relative reconstruction loss")
ax2 = ax.twinx()
ax2.plot(z, total_kl, marker="s", color="C1", label="total KL")
ax2.set_ylabel("KL [nats]")
ax.set_title("Fit saturates near z=5")
mark_default(ax)
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, frameon=False, fontsize=8, loc="center right")

for ax in axes:
    ax.set_xticks(z)
    ax.grid(alpha=0.2)

fig.savefig(OUT / "latent_dimensionality_sweep.png", dpi=180)
print(f"wrote {OUT / 'latent_dimensionality_sweep.png'}")
