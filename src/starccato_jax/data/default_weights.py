import os
from typing import Tuple

from .downloader import download_with_progress
from .urls import DEFAULT_WEIGHTS_URL

DATA_DIR = f"{os.path.dirname(__file__)}/default_weights/"
DEFAULT_WEIGHTS_FNAME = "model.h5"


def get_default_weights(clean: bool = False) -> str:
    fpath = os.path.join(DATA_DIR, DEFAULT_WEIGHTS_FNAME)
    if clean or not os.path.exists(fpath):
        os.makedirs(DATA_DIR, exist_ok=True)
        download_with_progress(DEFAULT_WEIGHTS_URL, fpath)
    return DATA_DIR
