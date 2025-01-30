import pytest
from absl import flags
import os
from scipy.signal import chirp, windows
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from starccato_jax.vae import Batch
from typing import Iterator, Sequence
from starccato_jax.vae import run_training, Config
import itertools


np.random.seed(0)

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.join(HERE, "data")
os.makedirs(DATA_DIR, exist_ok=True)

FS = 100
DURATION = 10.0
T = np.linspace(0, DURATION, int(FS * DURATION))
F1_RANGE = (-3, 3)
F2_RANGE = (-5, 5)


def gen_signals(n: int):
    f1s = np.random.uniform(*F1_RANGE, n)
    f2s = np.random.uniform(*F2_RANGE, n)
    ys = jnp.array([
        (f1 * T + f2)
        for f1, f2 in zip(f1s, f2s)])
    # standardize ys
    # ys = (ys - ys.mean()) / ys.std()
    return ys


def generate_iterator(
        num_draws: int,
        batch_size: int,
) -> Iterator[Batch]:
    signals = gen_signals(num_draws*batch_size)
    signals = signals.reshape(num_draws, batch_size, *signals.shape[1:])
    return itertools.cycle(iter(signals))  # Wrap in cycle to loop indefinitely


def plot_signals(signals, color="tab:blue", ax=None):
    if ax is None:
        plt.figure()
        ax = plt.gca()
    fig = ax.figure
    for s in signals:
        # ax.psd(s, Fs=FS, NFFT=128, noverlap=64, scale_by_freq=False, detrend=None, alpha=0.1, color=color)
        ax.plot(s, color=color, alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Amp')
    return fig


if __name__ == '__main__':

    print("Loading data")

    # plot 3 batches of data (blue is training, orange is validation)
    batch_size = 32
    train_iter, val_iter = generate_iterator(batch_size*8, batch_size), generate_iterator(batch_size*2, batch_size)

    fig, axs = plt.subplots(3, 1, figsize=(6, 12))
    for i in range(3):
        plot_signals(next(train_iter), color="tab:blue", ax=axs[i])
        plot_signals(next(val_iter), color="tab:orange", ax=axs[i])
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/data.png")

    config = Config(
        batch_size=128,
        training_steps=2000
    )
    plots_dir = os.path.join(DATA_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    run_training(
        train_dataset=train_iter,
        eval_dataset=val_iter,
        config=config,
        outdir=plots_dir
    )