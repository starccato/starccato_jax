import os

import h5py
import numpy as np

from ..downloader import download_with_progress
from ..urls import ALIGNED_DATA_URL

HERE = os.path.dirname(__file__)
CCSNE_CACHE = f"{HERE}/ccsne_data.npz"


def load_richers_ccsne_data(clean: bool = False) -> np.ndarray:
    if not os.path.exists(CCSNE_CACHE) or clean:
        local_cache = os.path.join(HERE, ALIGNED_DATA_URL.split("/")[-1])
        download_with_progress(ALIGNED_DATA_URL, local_cache)

        with h5py.File(local_cache, "r") as f:
            data = f["ccsne"][:]

        np.savez(CCSNE_CACHE, data=data)
    data = np.load(CCSNE_CACHE)["data"]
    return data
