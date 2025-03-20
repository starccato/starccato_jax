import os

from starccato_jax import Config, StarccatoVAE
from starccato_jax.data import load_training_data

HERE = os.path.dirname(__file__)


def main():
    train_data, val_data = load_training_data()
    config = Config(latent_dim=16, epochs=1000, cyclical_annealing_cycles=3)
    starccato_vae = StarccatoVAE.train(
        model_dir="model_out",
        config=config,
        # plot_every=50,
        # print_every=50,
        track_gradients=True,
    )


if __name__ == "__main__":
    main()
