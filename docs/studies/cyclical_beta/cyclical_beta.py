import os

from starccato_sampler.sampler import sample

from starccato_jax import Config, StarccatoVAE
from starccato_jax.data import load_training_data

HERE = os.path.dirname(__file__)
OUT_BETA_RAMP = os.path.join(HERE, "out_models/beta_monotonic")
OUT_BETA_1 = os.path.join(HERE, "out_models/beta_1")
OUT_BETA_0 = os.path.join(HERE, "out_models/beta_0")
OUT_BETA_CYCLICAL = os.path.join(HERE, "out_models/beta_cyclical")

Z_SIZE = 24

config_args = dict(latent_dim=Z_SIZE, epochs=1000)

kwargs = [
    dict(
        model_dir=OUT_BETA_RAMP,
        config=Config(
            beta_start=0,
            beta_end=1,
            cyclical_annealing_cycles=1,
            **config_args,
        ),
    ),
    dict(
        model_dir=OUT_BETA_0,
        config=Config(
            beta_start=0,
            beta_end=0,
            cyclical_annealing_cycles=0,
            **config_args,
        ),
    ),
    dict(
        model_dir=OUT_BETA_CYCLICAL,
        config=Config(
            beta_start=0,
            beta_end=1,
            cyclical_annealing_cycles=3,
            **config_args,
        ),
    ),
    dict(
        model_dir=OUT_BETA_1,
        config=Config(
            beta_start=1,
            beta_end=1,
            cyclical_annealing_cycles=0,
            **config_args,
        ),
    ),
]


def main():
    train_data, val_data = load_training_data()
    for kw in kwargs:
        model = StarccatoVAE.train(**kw)
        sample(val_data[0], model_path=kw["model_dir"], outdir=kw["model_dir"])


if __name__ == "__main__":
    main()
