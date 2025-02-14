from tqdm.auto import trange

from starccato_jax.data import load_data
from starccato_jax.sampler import sample_latent_vars_given_data
from starccato_jax.trainer import Config, train_vae
import os

MODEL_DIR = 'vae_outdir'

# train model

train_data, val_data = load_data()




if not os.path.exists(MODEL_DIR):
    train_vae(
        train_data, val_data, save_dir=MODEL_DIR, print_every=500,
        config=Config(
            latent_dim=32,
            epochs=2000,
            learning_rate=1e-4,
        )
    )


i = 42

sample_latent_vars_given_data(
        val_data[i], model_path=MODEL_DIR, outdir=f"out_mcmc/val{i}", verbose=False,
        num_warmup=1500, num_samples=2000
    )



# sample latent variables
# for i in trange(len(val_data), desc="Sampling latent variables"):
#     sample_latent_vars_given_data(
#         val_data[i], model_path=MODEL_DIR, outdir=f"out_mcmc/val{i}", verbose=False
#     )
