import os

from .downloader import download_with_progress
from .urls import BLIP_WEIGHTS_URL, CCSNE_WEIGHTS_URL

DATA_DIR = f"{os.path.dirname(__file__)}/default_weights/"


def get_default_weights_dir(
    dataset: str = "ccsne", clean: bool = False
) -> str:
    fpath = f"{DATA_DIR}/{dataset}/model.h5"
    model_dir = os.path.dirname(fpath)
    if clean or not os.path.exists(fpath):
        os.makedirs(model_dir, exist_ok=True)
        if "blip" in dataset:
            download_with_progress(BLIP_WEIGHTS_URL, fpath)
        elif "ccsne" in dataset:
            download_with_progress(CCSNE_WEIGHTS_URL, fpath)
    return model_dir
