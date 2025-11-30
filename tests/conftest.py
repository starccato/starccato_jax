import logging
import os
import subprocess

import h5py
import numpy as np
import pytest
from utils import BRANCH

from starccato_jax.data.training_data import TrainValData

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
    """Load GAN generated signals from cache (shape = (1000, 512))"""
    cache = os.path.join(HERE, "gan_signals.h5")
    if not os.path.exists(cache):
        data_url = "https://github.com/starccato/data/blob/main/generated_signals/gan_signals.h5?raw=true"
        # prefer urllib to avoid requiring external wget binary
        try:
            from urllib.request import urlretrieve
            urlretrieve(data_url, cache)
        except Exception:
            # fall back to requests if available
            try:
                import requests
                resp = requests.get(data_url, stream=True)
                resp.raise_for_status()
                with open(cache, "wb") as fh:
                    for chunk in resp.iter_content(8192):
                        if chunk:
                            fh.write(chunk)
            except Exception:
                # last resort: try wget if present
                subprocess.run(["wget", "-O", cache, data_url], check=False)

    with h5py.File(cache, "r") as f:
        ds = f["signals"]
        if isinstance(ds, h5py.Dataset):
            signals = ds[()]
        else:
            raise RuntimeError(f"Expected 'signals' dataset in {cache}, found {type(ds)}")

    return signals[:N_RICHERS]


@pytest.fixture
def richers_signals() -> np.ndarray:
    data = TrainValData.load()
    assert data.combined.shape == (N_RICHERS, 512)
    return data.combined


@pytest.fixture
def cached_vae_signals() -> np.ndarray:
    cache = os.path.join(HERE, "test_data/vae_signals[main].h5")
    with h5py.File(cache, "r") as f:
        signals = f["signals"][:]
        return signals
