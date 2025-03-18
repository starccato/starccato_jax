import argparse
import os
from typing import Tuple

import jax
import numpy as np
from jax.random import PRNGKey
from starccato_sampler.sampler import sample

from starccato_jax import StarccatoVAE

rng = PRNGKey(0)

INJECTION_FN = "injections.npz"
N = 300


def generate_injection_set():
    model = StarccatoVAE()
    z = jax.random.normal(rng, shape=(N, model.latent_dim))
    signals = model.generate(z, rng)
    # save z and signals to disk
    np.savez(INJECTION_FN, z=z, signals=signals)


def load_injections() -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(INJECTION_FN):
        generate_injection_set()
    data = np.load(INJECTION_FN)
    return data["signals"], data["z"]
