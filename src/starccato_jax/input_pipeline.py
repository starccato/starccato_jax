"""Input pipeline for VAE dataset."""

import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds


def build_train_set(batch_size, ds_builder):
  """Builds train dataset."""

  train_ds = ds_builder.as_dataset(split=tfds.Split.TRAIN)
  train_ds = train_ds.map(prepare_image)
  train_ds = train_ds.cache()
  train_ds = train_ds.repeat()
  train_ds = train_ds.shuffle(50000)
  train_ds = train_ds.batch(batch_size)
  train_ds = iter(tfds.as_numpy(train_ds))
  return train_ds


def build_test_set(ds_builder):
  """Builds train dataset."""
  test_ds = ds_builder.as_dataset(split=tfds.Split.TEST)
  test_ds = test_ds.map(prepare_image).batch(10000)
  test_ds = jnp.array(list(test_ds)[0])
  test_ds = jax.device_put(test_ds)
  return test_ds


def prepare_image(x):
  x = tf.cast(x['image'], tf.float32)
  x = tf.reshape(x, (-1,))
  return x
