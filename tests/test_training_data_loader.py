import os.path

import jax
import pytest

from starccato_jax.data import TrainValData
from starccato_jax.data.datasets.blip_glitches import BLIP_CACHE
from starccato_jax.data.datasets.richers_ccsne import CCSNE_CACHE


def test_training_data():
    data = TrainValData.load(clean=True)
    assert data.train.shape == (1411, 512)
    assert data.train.shape[0] + data.val.shape[0] == 1411 + 353
    data2 = TrainValData.load()

    # check that the data is the same (it should be shuffled the same way)
    assert (data.train == data2.train).all()
    assert (data.val == data2.val).all()


def test_batch_generation():
    data = TrainValData.load()
    rng = jax.random.PRNGKey(42)
    batch_size = 64
    n_batches = 1411 // batch_size
    batches = data.generate_training_batches(batch_size, rng)
    assert batches.shape == (n_batches, batch_size, 512)


@pytest.mark.parametrize(
    "source, cache",
    [
        ("blip", BLIP_CACHE),
        ("ccsne", CCSNE_CACHE),
    ],
)
def test_cache_generation(source, cache):
    data = TrainValData.load(source=source, clean=True)
    # assert cache file exists
    assert os.path.exists(cache)
    print(f"{source} -> {cache} [{data.train.shape, data.val.shape}]")

