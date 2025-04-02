import jax

from starccato_jax.data import TrainValData


def test_training_data():
    data = TrainValData.load(clean=True)
    assert data.train.shape == (1411, 256)
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
    assert batches.shape == (n_batches, batch_size, 256)
