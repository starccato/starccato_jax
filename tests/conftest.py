import logging
import os
import subprocess

import h5py
import numpy as np
import pytest
from utils import BRANCH

from starccato_jax.data.training_data import load_richers_dataset

HERE = os.path.abspath(os.path.dirname(__file__))
N_RICHERS = 1764


logger = logging.getLogger("starccato_jax")
logger.setLevel(logging.DEBUG)


@pytest.fixture
def outdir() -> str:
    dir = os.path.join(HERE, f"test_output-{BRANCH}")
    os.makedirs(dir, exist_ok=True)
    return dir


@pytest.fixture
def gan_signals() -> np.ndarray:
    """Load GAN generated signals from cache (shape = (1000, 256))"""
    cache = os.path.join(HERE, "gan_signals.h5")
    if not os.path.exists(cache):
        data_url = "https://github.com/starccato/data/blob/main/generated_signals/gan_signals.h5?raw=true"
        subprocess.run(["wget", "-O", cache, data_url])

    with h5py.File(cache, "r") as f:
        signals = f["signals"][:]

    return signals[:N_RICHERS]


@pytest.fixture
def richers_signals() -> np.ndarray:
    data = load_richers_dataset()
    assert data.shape == (N_RICHERS, 256)
    return data


@pytest.fixture
def cached_vae_signals() -> np.ndarray:
    cache = os.path.join(HERE, "test_data/vae_signals[main].h5")
    with h5py.File(cache, "r") as f:
        signals = f["signals"][:]
        return signals
