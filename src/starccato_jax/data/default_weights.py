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
        if "blip" in dataset:
            download_with_progress(BLIP_WEIGHTS_URL, fpath)
        elif "ccsne" in dataset:
            download_with_progress(CCSNE_WEIGHTS_URL, fpath)

    return model_dir
