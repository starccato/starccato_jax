_ROOT_URL = "https://starccato.github.io/data/"

CCSNE_SIGNALS_URL = f"{_ROOT_URL}/training/richers_1764.csv"
CCSNE_PARAMETERS_URL = f"{_ROOT_URL}/training/richers_1764_parameters.csv"

BLIP_SIGNALS_URL = f"{_ROOT_URL}/training/gwspy_glitches.dat.zip"
ALIGNED_DATA_URL = f"{_ROOT_URL}/training/aligned_data.h5"

_WEIGHTS_VERSION = "v0.4.0"
_WEIGHTS_ROOT = f"{_ROOT_URL}/weights/starcatto_jax/{_WEIGHTS_VERSION}"

BLIP_WEIGHTS_URL = f"{_WEIGHTS_ROOT}/blip_vae.h5"
CCSNE_WEIGHTS_URL = f"{_WEIGHTS_ROOT}/ccsne_vae.h5"

# These hashes are part of the immutable default-weight contract.  A new
# artifact must use a new versioned URL and hash rather than replacing a file
# in place.
BLIP_WEIGHTS_SHA256 = (
    "4f7d590bfe5f23f07785fa6b16b68ed8115a8ad1e8333c87341b7731c27c9879"
)
CCSNE_WEIGHTS_SHA256 = (
    "edf65676b43f0b015dfff4a5613b9189b1f2bc19159e0a599d5db5595abea0be"
)
