import os

import numpy as np

from .blip_glitches import load_blip_data
from .richers_ccsne import load_richers_ccsne_data


def load_dataset(source: str, clean: bool = False) -> np.ndarray:
    data = None

    # check known sources
    if source == "ccsne":
        data = load_richers_ccsne_data(clean=clean)
    elif source == "blip":
        data = load_blip_data(clean=clean)

    # check if source is a path to npz file
    if os.path.exists(source):
        if source.endswith(".npz"):
            data = np.load(source)["data"]

        elif source.endswith(".csv"):
            data = np.loadtxt(source, delimiter=",")

    # if data is still None, raise an error
    if data is None:
        raise ValueError(
            f"Unknown source: {source}. Please provide a valid source or a path to a .npz or .csv file."
        )

    return data
