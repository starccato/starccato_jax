_ROOT_URL = "https://starccato.github.io/data/"

CCSNE_SIGNALS_URL = f"{_ROOT_URL}/training/richers_1764.csv"
CCSNE_PARAMETERS_URL = f"{_ROOT_URL}/training/richers_1764_parameters.csv"

BLIP_SIGNALS_URL = f"{_ROOT_URL}/training/gwspy_glitches.dat.zip"
ALIGNED_DATA_URL = f"{_ROOT_URL}/training/aligned_data.h5"

_WEIGHTS_VERSION = "v0.3.0"
_WEIGHTS_ROOT = f"{_ROOT_URL}/weights/starcatto_jax/{_WEIGHTS_VERSION}"

BLIP_WEIGHTS_URL = f"{_WEIGHTS_ROOT}/blip_vae.h5"
CCSNE_WEIGHTS_URL = f"{_WEIGHTS_ROOT}/ccsne_vae.h5"
