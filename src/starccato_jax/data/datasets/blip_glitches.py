import os

import h5py
import numpy as np

from ..downloader import download_with_progress
from ..urls import ALIGNED_DATA_URL

HERE = os.path.dirname(__file__)
BLIP_CACHE = f"{HERE}/blip_data.npz"


def load_blip_data(clean: bool = False) -> np.ndarray:
    if not os.path.exists(BLIP_CACHE) or clean:
        # Load the data from the h5 file
        local_cache = os.path.join(HERE, ALIGNED_DATA_URL.split("/")[-1])
        download_with_progress(ALIGNED_DATA_URL, local_cache)

        with h5py.File(local_cache, "r") as f:
            signals = f["blip"][:]

        np.savez(BLIP_CACHE, data=signals)
    data = np.load(BLIP_CACHE)["data"]
    return data
