import jax.numpy as jnp
import matplotlib.pyplot as plt
import tsnex

from starccato_jax import StarccatoVAE
from starccato_jax.data import load_training_data

_, val_data = load_training_data(train_fraction=0.8)

vae = StarccatoVAE()

# get validataion encodings
val_encodings = vae.encode(val_data)

# compute correlation matrix of the encodings
correlation_matrix = jnp.corrcoef(val_encodings.T)

# ensure bewteen -1 and 1
correlation_matrix = jnp.clip(correlation_matrix)

# plot the correlation matrix
plt.matshow(correlation_matrix)
plt.colorbar()
plt.title("Correlation matrix of the encodings")
plt.savefig("correlation_matrix.png")


z_2dim = tsnex.transform(val_encodings, n_components=3)
plt.figure()
plt.scatter(z_2dim[:, 0], z_2dim[:, 1], c=z_2dim[:, 2])
plt.colorbar()
plt.title("t-SNE of the encodings")
plt.savefig("tsne.png")
