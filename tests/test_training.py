from starccato_jax.vae import run_training, Config, Batch
from typing import Tuple, Iterator



def test_training(outdir, data_iterators):
    train_itr, val_itr = data_iterators

    config = Config(
        batch_size=32,
        training_steps=1000
    )
    run_training(
        train_dataset=train_itr,
        eval_dataset=val_itr,
        config=config
    )


