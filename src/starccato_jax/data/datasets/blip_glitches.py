import os
import shutil
import zipfile

import h5py
import numpy as np

from ..downloader import download_with_progress
from ..urls import ALIGNED_DATA_URL, BLIP_SIGNALS_URL

HERE = os.path.dirname(__file__)
BLIP_CACHE = f"{HERE}/blip_data.npz"
DAT_FNAME = "gwspy_glitches.dat"


def _get_raw_data():
    """Unzip the BLIP data file."""
    zip_path = BLIP_SIGNALS_URL.split("/")[-1]
    zip_path = os.path.join(HERE, zip_path)
    download_with_progress(BLIP_SIGNALS_URL, zip_path)

    tmp_dir = os.path.join(HERE, "tmp")

    # Unzip the file
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(tmp_dir)

    # Remove the zip file after extraction
    os.remove(zip_path)
    # Load the data
    data_path = f"{tmp_dir}/{DAT_FNAME}"

    # np read the dat file
    data = np.loadtxt(data_path, delimiter=None, dtype="float32")

    # Remove the temporary directory
    shutil.rmtree(tmp_dir)

    return data


def load_blip_data(clean: bool = False) -> np.ndarray:
    if not os.path.exists(BLIP_CACHE) or clean:
        # Download and unzip the raw data
        # signals = _get_raw_data()

        # Load the data from the h5 file

        local_cache = os.path.join(HERE, ALIGNED_DATA_URL.split("/")[-1])
        download_with_progress(ALIGNED_DATA_URL, local_cache)

        with h5py.File(local_cache, "r") as f:
            signals = f["blip"][:]

        np.savez(BLIP_CACHE, data=signals)
    data = np.load(BLIP_CACHE)["data"]
    return data
