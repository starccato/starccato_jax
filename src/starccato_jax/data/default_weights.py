import os

from .downloader import download_with_progress
from .urls import BLIP_WEIGHTS_URL, CCSNE_WEIGHTS_URL
import time

DATA_DIR = f"{os.path.dirname(__file__)}/default_weights/"


def get_default_weights_dir(
    dataset: str = "ccsne", clean: bool = False
) -> str:

    model_dir = os.path.join(DATA_DIR, dataset)
    os.makedirs(model_dir, exist_ok=True)
    fpath = os.path.join(model_dir, "model.h5")

    should_download = clean or (not os.path.exists(fpath))
    if not should_download:
        try:
            age_seconds = time.time() - os.path.getmtime(fpath)
            if age_seconds > 24 * 3600:
                should_download = True
        except OSError:
            should_download = True

    if should_download:
        url = BLIP_WEIGHTS_URL if "blip" in dataset else CCSNE_WEIGHTS_URL
        try:
            download_with_progress(url, fpath)
        except OSError:
            # Offline (e.g. HPC compute node with no internet): fall back to
            # the existing cached weights rather than crashing on a routine
            # 24h freshness check.
            if not os.path.exists(fpath):
                raise

    return model_dir
