import hashlib
import os
import tempfile

from .downloader import download_with_progress
from .urls import (
    BLIP_WEIGHTS_SHA256,
    BLIP_WEIGHTS_URL,
    CCSNE_WEIGHTS_SHA256,
    CCSNE_WEIGHTS_URL,
)

DATA_DIR = f"{os.path.dirname(__file__)}/default_weights/"

WEIGHT_SPECS = {
    "blip": (BLIP_WEIGHTS_URL, BLIP_WEIGHTS_SHA256),
    "ccsne": (CCSNE_WEIGHTS_URL, CCSNE_WEIGHTS_SHA256),
}


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download_verified(url: str, expected_sha256: str, destination: str):
    directory = os.path.dirname(destination)
    temporary = tempfile.NamedTemporaryFile(
        prefix=".model.", suffix=".download", dir=directory, delete=False
    )
    temporary_path = temporary.name
    temporary.close()
    try:
        download_with_progress(url, temporary_path)
        actual_sha256 = _file_sha256(temporary_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                "Default-weight checksum mismatch for "
                f"{url}: expected {expected_sha256}, got {actual_sha256}"
            )
        os.replace(temporary_path, destination)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def get_default_weights_dir(
    dataset: str = "ccsne", clean: bool = False
) -> str:
    if dataset not in WEIGHT_SPECS:
        valid = ", ".join(sorted(WEIGHT_SPECS))
        raise ValueError(
            f"Unknown default-weight dataset '{dataset}'; use {valid}"
        )
    model_dir = os.path.join(DATA_DIR, dataset)
    os.makedirs(model_dir, exist_ok=True)
    fpath = os.path.join(model_dir, "model.h5")
    url, expected_sha256 = WEIGHT_SPECS[dataset]

    should_download = clean or (not os.path.exists(fpath))
    if not should_download:
        should_download = _file_sha256(fpath) != expected_sha256

    if should_download:
        _download_verified(url, expected_sha256, fpath)

    return model_dir
