import gengli
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal.windows import tukey

from starccato_jax.credible_intervals import pointwise_ci
from starccato_jax.data.urls import CCSNE_PARAMETERS_URL, CCSNE_SIGNALS_URL

srate = 4096.0  # Hz
GLITCH_FNAME = "glitches.dat"


def load_richers_dataset(seed: int = 0) -> np.ndarray:
    parameters = pd.read_csv(CCSNE_PARAMETERS_URL)
    data = pd.read_csv(CCSNE_SIGNALS_URL).astype("float32")[
        parameters["beta1_IC_b"] > 0
    ]
    data = data.values.T

    # shuffle data
    np.random.seed(seed)
    np.random.shuffle(data)

    # standardise using max value from entire dataset, and zero mean for each row
    data = data / np.max(np.abs(data))
    data = data - np.mean(data, axis=1, keepdims=True)

    # # Standardize each sample (row) to have zero mean and unit variance.
    # mus = np.mean(data, axis=1, keepdims=True)
    # sigmas = np.std(data, axis=1, keepdims=True)
    # data = (data - mus) / sigmas
    return data


def generate_glitches(N_glitches: int):
    g = gengli.glitch_generator("L1")
    # (whithened) glitch @ 4096Hz
    glitches = g.get_glitch(
        N_glitches, srate=srate, fhigh=250, alpha=0.2, seed=0
    )
    np.savetxt(GLITCH_FNAME, glitches)


def load_glitches():
    glitches = np.loadtxt(GLITCH_FNAME)
    t_glitches = (
        np.arange(-glitches.shape[1] // 2, glitches.shape[1] // 2) / srate
    )
    return glitches, t_glitches


def load_signals():
    ccsne_data = load_richers_dataset(seed=0)
    # normalize the data (so between -1 and 1)
    ccsne_data = ccsne_data / np.max(np.abs(ccsne_data))
    t_ccsne = np.arange(0, ccsne_data.shape[1]) / srate
    t_ccsne = t_ccsne - t_ccsne[58]  # index of the peak
    return ccsne_data, t_ccsne


def plot_signals(signals, time, ax, color="black", label=None):
    ci = pointwise_ci(signals, ci=0.9)
    ax.fill_between(
        time, ci[0], ci[-1], color=color, alpha=0.5, lw=0, label=label
    )
    ax.plot(time, ci[1], color=color, lw=3)
    ax.set_title(label)


def pad_signal(signal, target_length):
    """Pad the signal symmetrically to reach target_length."""
    current_length = signal.shape[1]
    pad_total = target_length - current_length
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    return np.pad(signal, ((0, 0), (pad_left, pad_right)), mode="constant")


def align_signal(signal, srate):
    """Roll the signal so that its maximum absolute value is at time=0.

    Assumes that the time vector is defined as from -N/2 to N/2.
    """
    N = signal.shape[1]
    peak_index = np.argmax(np.abs(signal), axis=1)[
        0
    ]  # assuming one signal per batch (first row)
    center_index = N // 2
    shift = center_index - peak_index
    # Use np.roll after padding; the wrapped part will be zeros if padding was applied
    signal_aligned = np.roll(signal, shift, axis=1)
    t_signal = get_time_vector(N, srate)
    return signal_aligned, t_signal


def get_time_vector(N, srate):
    return np.arange(-N // 2, N // 2) / srate


def truncate_signals_and_window(data, time, t_start, t_end, tukey_alpha=0.1):
    """Truncate the signals to the given time window and apply a Tukey window."""
    # 1. Truncate the signals to the given time window
    mask = (time >= t_start) & (time <= t_end)
    truncated_data = data[:, mask]

    # 2. Apply a Tukey window
    window = tukey(truncated_data.shape[1], alpha=tukey_alpha)
    window = window.reshape(1, -1)  # Reshape to match the data shape
    truncated_data = truncated_data * window

    # zero pad the beginning and end of the signal so we have data-len power of 2
    N = 2 ** int(np.ceil(np.log2(truncated_data.shape[1])))
    truncated_data = pad_signal(truncated_data, N)
    truncated_time = get_time_vector(N, srate)
    print(f"Cutting {data.shape} to {truncated_data.shape}")

    return truncated_data, truncated_time


def main():
    generate_glitches(2000)
    glitches, t_glitches = load_glitches()
    ccsne_data, t_ccsne = load_signals()
    print("Glitches shape:", glitches.shape)
    print("CCSNe shape:", ccsne_data.shape)

    # Determine the maximum length between the signals
    target_length = max(glitches.shape[1], ccsne_data.shape[1])

    # Pad both signals to have the same length
    glitches_padded = pad_signal(glitches, target_length)
    ccsne_padded = pad_signal(ccsne_data, target_length)

    # Now roll (shift) the padded signals so that the maximum absolute value is at time = 0
    glitches_aligned, t_glitches_new = align_signal(glitches_padded, srate)
    ccsne_aligned, t_ccsne_new = align_signal(ccsne_padded, srate)

    # and finally, truncate + window the signals
    glitches_trunc, t_glitches_new = truncate_signals_and_window(
        glitches_aligned,
        t_glitches_new,
        t_start=-0.05,
        t_end=0.05,
        tukey_alpha=0.1,
    )
    ccsne_trunc, t_ccsne_new = truncate_signals_and_window(
        ccsne_aligned, t_ccsne_new, t_start=-0.05, t_end=0.05, tukey_alpha=0.1
    )

    # Plotting the results
    fig, ax = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
    plot_signals(
        glitches_trunc,
        t_glitches_new,
        ax[0],
        color="tab:purple",
        label="Glitches",
    )
    plot_signals(
        ccsne_trunc, t_ccsne_new, ax[1], color="tab:orange", label="CCSNe"
    )
    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig("glitches_and_ccsne_trunc.png", dpi=300)
    plt.show()

    # Save the processed data
    with h5py.File("aligned_data.h5", "w") as f:
        f.create_dataset("blip", data=glitches_trunc)
        f.create_dataset("ccsne", data=ccsne_trunc)


if __name__ == "__main__":
    main()
