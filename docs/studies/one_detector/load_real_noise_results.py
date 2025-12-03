import os
import json
import numpy as np
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt

files = glob("out/*/metrics.csv")

LOGBF = "logBF_sig_alt"
SNR = "snr_excess"


# load each JSON, append to list, to later merge into a DataFrame
data = []
for f in files:
    d = pd.read_csv(f)
    data.append(list(d.values[0]))

header = pd.read_csv(files[0])
header = list(header.columns)

df = pd.DataFrame(data, columns=header)
df.to_csv("out/results.csv", index=False)




# make a scatter plot of SNR (y) logBF (x) colored by type
plt.figure(figsize=(8, 6))
types = df["inject"].unique()
colors = plt.cm.viridis(np.linspace(0, 1, len(types)))
for t, c in zip(types, colors):
    subset = df[df["inject"] == t]
    plt.scatter(subset[LOGBF], subset[SNR], label=t, color=c, alpha=0.7)

plt.legend()
plt.xlabel("LnZsig - Log( Zglitch + Znoise )")
plt.ylabel("Signal-to-Noise Ratio (SNR)")
plt.savefig("out/results_scatter.png")


#### PDF plots
plt.close('all')
# Define a consistent linear threshold for symlog
LIN_THRESH = 10.0

# Define a common plotting range for logBF for consistent visual comparison
# Using a fixed range (e.g., -10 to 10) is often better than quantiles
# if the physical interpretation of the logBF is known.
# However, sticking to your quantile approach requires filtering the data.

fig, ax = plt.subplots(2, 1, figsize=(8, 6))

# plot PDFs of SNR for glitches and signals on top axes, logBF on bottom axes
types = ["signal", "glitch"]
colors = ["tab:orange", "tab:blue"]


min_lim = df[LOGBF].quantile(0.1) # Assuming 'df' is available
max_lim = df[LOGBF].quantile(0.9) # Assuming 'df' is availab

for t, c in zip(types, colors):
    subset = df[df["inject"] == t].copy()  # Using .copy() to avoid SettingWithCopyWarning

    # --- TOP PLOT (SNR) ---
    snr_data = subset[SNR]
    q90_snr = snr_data.quantile(0.9)
    # Ensure start of geomspace is positive and reasonable
    snr_min = snr_data[snr_data > 0].min() if snr_data[snr_data > 0].min() < q90_snr else 1.0
    bins_snr = np.geomspace(snr_min, q90_snr, 30)

    ax[0].hist(snr_data, bins=bins_snr, density=True, histtype='step', lw=2, color=c, label=t)

    # --- BOTTOM PLOT (logBF) ---
    logbf_data = subset[LOGBF]
    sign = np.sign(logbf_data.values)[-1]
    logbf_data = np.abs(logbf_data)

    # 1. Define the plot range using quantiles
    q90_logbf = logbf_data.quantile(0.9)
    q10_logbf = logbf_data.quantile(0.01)

    # 2. Filter the data to match the bin range
    logbf_data_filtered = logbf_data[
        (logbf_data >= q10_logbf) &
        (logbf_data <= q90_logbf)
    ]

    # 3. Calculate linear bins based on the filtered data range (matching the filter)
    # Note: Using the actual filtered min/max is safer than using q10/q90 directly
    bins_logbf = np.geomspace(logbf_data_filtered.min(), logbf_data_filtered.max(), 20)
    bins_logbf = np.sort(bins_logbf * sign)  # Restore original sign for bin edges
    print(bins_logbf)

    print(f"Type: {t}, logBF bins from {bins_logbf[0]} to {bins_logbf[-1]}")

    # 4. Pass the FILTERED data to the histogram
    ax[1].hist(logbf_data_filtered*sign, bins=bins_logbf, density=True, histtype='step', lw=2, color=c, label=t)

# --- SET AXES PROPERTIES ---

ax[0].set_xscale('log')  # Log scale for SNR
ax[0].set_xlabel("Signal-to-Noise Ratio (SNR)")
ax[0].set_ylabel("PDF")
ax[0].legend()  # Add legend for SNR plot

ax[1].set_xlabel("LnZsig - Log( Zglitch + Znoise )")
ax[1].set_ylabel("PDF")
# Set x-limits to match the bin range for a clean plot
# ax[1].set_xlim(q10_logbf, q90_logbf)
ax[1].set_xscale('symlog', linthresh=LIN_THRESH)  # Use consistent linthresh
ax[1].legend()  # Add legend for logBF plot
ax[1].set_xlim(min_lim, max_lim)

fig.tight_layout()  # Improves spacing between subplots
fig.savefig("out/pdf.png")