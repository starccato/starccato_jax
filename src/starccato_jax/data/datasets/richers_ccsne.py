import os

import h5py
import numpy as np
import pandas as pd

from ..downloader import download_with_progress
from ..urls import ALIGNED_DATA_URL, CCSNE_PARAMETERS_URL, CCSNE_SIGNALS_URL

HERE = os.path.dirname(__file__)
CCSNE_CACHE = f"{HERE}/ccsne_data.npz"


def _get_raw_data():
    """Unzip the CCSNE data file."""
    parameters = pd.read_csv(CCSNE_PARAMETERS_URL)
    data = pd.read_csv(CCSNE_SIGNALS_URL).astype("float32")[
        parameters["beta1_IC_b"] > 0
    ]
    data = data.values.T[:, 140:]  # cut the first few datapoints
    return data


def load_richers_ccsne_data(clean: bool = False) -> np.ndarray:
    if not os.path.exists(CCSNE_CACHE) or clean:
        # Download and unzip the raw data
        # signals = _get_raw_data()

        local_cache = os.path.join(HERE, ALIGNED_DATA_URL.split("/")[-1])
        download_with_progress(ALIGNED_DATA_URL, local_cache)

        with h5py.File(local_cache, "r") as f:
            data = f["ccsne"][:]

        np.savez(CCSNE_CACHE, data=data)
    data = np.load(CCSNE_CACHE)["data"]
    return data
