import jax
import matplotlib.pyplot as plt
import numpy as np

def plot_loss(train_losses, val_losses, fname=None):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    if fname is not None:
        plt.savefig(fname)


def plot_reconstructions(state, model, val_data, nrows=3, fname=None):
    ncols = nrows
    nsamples = nrows * ncols
    rng = jax.random.PRNGKey(0)
    idx = np.random.choice(val_data.shape[0], nsamples, replace=False)
    orig = val_data[idx]
    reconstructed, _, _ = model.apply({'params': state.params}, orig, rng)
    reconstructed = np.array(reconstructed)

    fig, axes = plt.subplots(nrows,ncols, figsize=(2.5*ncols, 2.5*nrows))
    axes = axes.flatten()
    for i in range(nsamples):
        axes[i].plot(orig[i], 'k', label='Original')
        axes[i].plot(reconstructed[i], 'tab:orange', label='Reconstruction')
        axes[i].set_axis_off()
    axes[-1].legend(framon=False, loc='lower right')
    plt.subplots_adjust(hspace=0, wspace=0)
    if fname is not None:
        plt.savefig(fname)


