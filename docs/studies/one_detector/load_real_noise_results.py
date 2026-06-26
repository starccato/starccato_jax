import os
import numpy as np
from glob import glob
import pandas as pd
import matplotlib.pyplot as plt
import sys

files = glob("out/*/metrics.csv")
os.makedirs("out", exist_ok=True)

# canonical column names we will emit
STD_COLS = [
    "source",
    "path",
    "inject",
    "seed",
    "snr_injection",
    "snr_data",
    "logBF",
    "lnz_signal",
    "lnz_glitch",
    "lnz_noise",
    "classification",
]


def first_present(raw: pd.Series, *keys):
    for k in keys:
        if k in raw and pd.notna(raw[k]):
            return raw[k]
    return None


def standardize_record(raw: pd.Series, path: str, source: str) -> dict:
    rec = {k: None for k in STD_COLS}
    rec["source"] = source
    rec["path"] = path
    rec["inject"] = raw.get("inject")
    rec["seed"] = raw.get("seed")
    rec["snr_injection"] = first_present(raw, "excess_snr_injection", "snr_excess", "snr_injection")
    rec["snr_data"] = first_present(raw, "excess_snr_data", "snr_data")
    rec["logBF"] = first_present(raw, "logBF", "logBF_sig_alt")
    rec["lnz_signal"] = first_present(raw, "lnz_signal")
    rec["lnz_glitch"] = first_present(raw, "lnz_glitch")
    rec["lnz_noise"] = first_present(raw, "lnz_noise")
    rec["classification"] = first_present(raw, "classification")
    return rec


def confusion_counts(df: pd.DataFrame, snr_threshold: float | None = None):
    if df.empty or "inject" not in df or "logBF" not in df:
        return None
    subset = df[df["inject"].isin(["signal", "glitch"])].copy()
    if snr_threshold is not None and "snr_injection" in subset:
        subset = subset[subset["snr_injection"] >= snr_threshold]
    if subset.empty:
        return None
    subset["pred"] = np.where(subset["logBF"] > 0, "signal", "glitch")
    tp_sig = int(((subset["inject"] == "signal") & (subset["pred"] == "signal")).sum())
    fn_sig = int(((subset["inject"] == "signal") & (subset["pred"] != "signal")).sum())
    tp_gli = int(((subset["inject"] == "glitch") & (subset["pred"] == "glitch")).sum())
    fn_gli = int(((subset["inject"] == "glitch") & (subset["pred"] != "glitch")).sum())
    total = len(subset)
    correct = tp_sig + tp_gli
    wrong = total - correct
    acc = correct / total if total else np.nan
    return {
        "snr_threshold": snr_threshold if snr_threshold is not None else 0.0,
        "tp_signal": tp_sig,
        "fn_signal": fn_sig,
        "tp_glitch": tp_gli,
        "fn_glitch": fn_gli,
        "correct": correct,
        "wrong": wrong,
        "total": total,
        "accuracy": acc,
    }


def misclassified(df: pd.DataFrame, snr_threshold: float | None = None) -> pd.DataFrame | None:
    if df.empty or "inject" not in df or "logBF" not in df:
        return None
    subset = df[df["inject"].isin(["signal", "glitch"])].copy()
    if snr_threshold is not None and "snr_injection" in subset:
        subset = subset[subset["snr_injection"] >= snr_threshold]
    if subset.empty:
        return None
    subset["pred"] = np.where(subset["logBF"] > 0, "signal", "glitch")
    wrong = subset[subset["pred"] != subset["inject"]].copy()
    return wrong[["inject", "pred", "snr_injection", "logBF", "path"]]


def plot_confusion(df: pd.DataFrame, snr_threshold: float | None, out_path: str):
    subset = df[df["inject"].isin(["signal", "glitch"])].copy()
    if snr_threshold is not None and "snr_injection" in subset:
        subset = subset[subset["snr_injection"] >= snr_threshold]
    if subset.empty:
        return
    subset["pred"] = np.where(subset["logBF"] > 0, "signal", "glitch")
    tp_sig = ((subset["inject"] == "signal") & (subset["pred"] == "signal")).sum()
    fn_sig = ((subset["inject"] == "signal") & (subset["pred"] == "glitch")).sum()
    fp_sig = ((subset["inject"] == "glitch") & (subset["pred"] == "signal")).sum()
    tp_gli = ((subset["inject"] == "glitch") & (subset["pred"] == "glitch")).sum()
    mat = np.array([[tp_sig, fn_sig], [fp_sig, tp_gli]], dtype=int)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(mat, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["signal", "glitch"])
    ax.set_yticklabels(["signal", "glitch"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    title = "Confusion"
    if snr_threshold is not None:
        title += f" (SNR≥{snr_threshold})"
    ax.set_title(title)
    total = mat.sum() if mat.sum() else 1
    acc = (mat[0, 0] + mat[1, 1]) / total if total else np.nan
    for i in range(2):
        for j in range(2):
            val = mat[i, j]
            pct = 100.0 * val / total
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color="black")
    ax.text(
        0.5,
        -0.2,
        f"Accuracy: {acc*100:.1f}%",
        transform=ax.transAxes,
        ha="center",
        va="center",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# load each JSON, append to list, to later merge into a DataFrame
data = []
for f in files:
    d = pd.read_csv(f)
    if d.empty:
        continue
    # assume single-row metrics CSV
    rec = standardize_record(d.iloc[0], path=f, source="real_noise")
    data.append(rec)

df = pd.DataFrame(data, columns=STD_COLS)
if df.empty:
    print("No metrics found under out/*/metrics.csv. Nothing to plot.")
    sys.exit(0)
df.to_csv("out/results.csv", index=False)

# keep full data for metrics/confusion; make a filtered copy for plots
df_plot = df.copy()
if "snr_injection" in df_plot.columns:
    df_plot = df_plot[df_plot["snr_injection"] >= 5]
else:
    print("Warning: 'snr_injection' missing; skipping SNR filter")

# percentage of signals with logBF > 0 and glitches with logBF < 0
num_signals = len(df_plot[df_plot["inject"] == "signal"])
num_signals_pos_logbf = len(df_plot[(df_plot["inject"] == "signal") & (df_plot["logBF"] > 0)])
if num_signals:
    perc_signals_pos_logbf = num_signals_pos_logbf / num_signals * 100
    print(f"Percentage of signals with logBF > 0: {perc_signals_pos_logbf:.2f}%")
else:
    print("No signal rows in plot subset.")
num_glitches = len(df_plot[df_plot["inject"] == "glitch"])
num_glitches_neg_logbf = len(df_plot[(df_plot["inject"] == "glitch") & (df_plot["logBF"] < 0)])
if num_glitches:
    perc_glitches_neg_logbf = num_glitches_neg_logbf / num_glitches * 100
    print(f"Percentage of glitches with logBF < 0: {perc_glitches_neg_logbf:.2f}%")
else:
    print("No glitch rows in plot subset.")

# confusion matrices (full set and SNR>=8)
conf_rows = []
for thr in [None, 8.0]:
    res = confusion_counts(df, snr_threshold=thr)
    if res:
        conf_rows.append(res)
        label = "all SNR" if thr is None else f"SNR≥{thr}"
        print(f"Confusion ({label}):", res)
        wrong = misclassified(df, snr_threshold=thr)
        if wrong is not None and not wrong.empty:
            csv_name = "misclassified_all.csv" if thr is None else "misclassified_snr8.csv"
            wrong.to_csv(f"out/{csv_name}", index=False)
            print(f"Saved {len(wrong)} misclassified rows ({label}) to out/{csv_name}")
        png_name = "confusion_matrix.png" if thr is None else "confusion_matrix_snr8.png"
        plot_confusion(df, snr_threshold=thr, out_path=f"out/{png_name}")
if conf_rows:
    conf_df = pd.DataFrame(conf_rows)
    conf_df.to_csv("out/confusion_matrix.csv", index=False)



# make a scatter plot of SNR (y) logBF (x) colored by type
if df_plot.empty:
    print("No rows after SNR filter; skipping plots.")
else:
    plt.figure(figsize=(8, 6))
    types = df_plot["inject"].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(types)))
    for t, c in zip(types, colors):
        subset = df_plot[df_plot["inject"] == t]
        plt.scatter(subset["logBF"], subset["snr_injection"], label=t, color=c, alpha=0.7)

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

if df_plot.empty:
    print("No rows after SNR filter; skipping PDF plots.")
else:
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))

    # plot PDFs of SNR for glitches and signals on top axes, logBF on bottom axes
    types = ["signal", "glitch"]
    colors = ["tab:orange", "tab:blue"]

    min_lim = df_plot["logBF"].quantile(0.1)
    max_lim = df_plot["logBF"].quantile(0.9)

    for t, c in zip(types, colors):
        subset = df_plot[df_plot["inject"] == t].copy()
        if subset.empty:
            continue

        # --- TOP PLOT (SNR) ---
        snr_data = subset["snr_injection"]
        q90_snr = snr_data.quantile(0.9)
        snr_min = snr_data[snr_data > 0].min() if snr_data[snr_data > 0].min() < q90_snr else 1.0
        bins_snr = np.geomspace(snr_min, q90_snr, 30)

        ax[0].hist(snr_data, bins=bins_snr, density=True, histtype='step', lw=2, color=c, label=t)

        # --- BOTTOM PLOT (logBF) ---
        logbf_data = subset["logBF"]
        sign = np.sign(logbf_data.values)[-1]
        logbf_data = np.abs(logbf_data)

        q90_logbf = logbf_data.quantile(0.9)
        q10_logbf = logbf_data.quantile(0.01)

        logbf_data_filtered = logbf_data[(logbf_data >= q10_logbf) & (logbf_data <= q90_logbf)]

        bins_logbf = np.geomspace(logbf_data_filtered.min(), logbf_data_filtered.max(), 20)
        bins_logbf = np.sort(bins_logbf * sign)  # Restore original sign for bin edges
        print(f"Type: {t}, logBF bins from {bins_logbf[0]} to {bins_logbf[-1]}")

        ax[1].hist(logbf_data_filtered * sign, bins=bins_logbf, density=True, histtype='step', lw=2, color=c, label=t)

    ax[0].set_xscale('log')  # Log scale for SNR
    ax[0].set_xlabel("Signal-to-Noise Ratio (SNR)")
    ax[0].set_ylabel("PDF")
    ax[0].legend()  # Add legend for SNR plot

    ax[1].set_xlabel("LnZsig - Log( Zglitch + Znoise )")
    ax[1].set_ylabel("PDF")
    ax[1].set_xscale('symlog', linthresh=LIN_THRESH)  # Use consistent linthresh
    ax[1].legend()  # Add legend for logBF plot
    ax[1].set_xlim(min_lim, max_lim)

    fig.tight_layout()  # Improves spacing between subplots
    fig.savefig("out/pdf.png")
