import matplotlib.pyplot as plt

from starccato_jax.data.default_weights import get_default_weights_dir
from starccato_jax.waveforms import StarccatoBlip, StarccatoCCSNe
import numpy as np


def test_waveforms(outdir):
    """
    Test the StarccatoWaveform class.
    """
    # Initialize the StarccatoBlip and StarccatoCCSNe classes
    blip = StarccatoBlip()
    ccsne = StarccatoCCSNe()

    # ensure that the paths of the model are correct
    assert blip.model_dir == get_default_weights_dir("blip")
    assert ccsne.model_dir == get_default_weights_dir("ccsne")

    sig = ccsne.generate(n=1)
    glitch = blip.generate(n=1)

    # Plot the generated signals
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(sig[0])
    plt.title("CCSNe Signal")
    plt.subplot(1, 2, 2)
    plt.plot(glitch[0])
    plt.title("Blip Signal")
    plt.savefig(f"{outdir}/waveforms.png")
    plt.close()


def test_waveform_default(outdir):
    """
    Test the default waveforms.
    """
    ccsne = StarccatoCCSNe()
    blip = StarccatoBlip()

    # Generate default waveforms
    ccsne_signal = ccsne.generate(n=1)[0]
    blip_signal = blip.generate(n=1)[0]

    # Check the shapes of the generated signals
    assert ccsne_signal.shape == (512,)
    assert blip_signal.shape == (512,)

    # Save the generated signals
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(ccsne_signal)
    plt.title("Default CCSNe Signal")
    plt.subplot(1, 2, 2)
    plt.plot(blip_signal)
    plt.title("Default Blip Signal")
    plt.savefig(f"{outdir}/default_waveforms.png")
    plt.close()


def test_outliers(outdir):
    """
    Test the outlier generation of the StarccatoWaveform class.
    """
    blip_vae = StarccatoBlip()
    ccsne_vae = StarccatoCCSNe()

    # Generate outlier signals
    blip_signals = blip_vae.generate(n=10000)
    ccsen_signals = ccsne_vae.generate(n=10000)

    # encode the signals in their own latent space
    blip_in_blip_latent = blip_vae.encode(blip_signals)
    ccsne_in_ccsne_latent = ccsne_vae.encode(ccsen_signals)

    # encode the signals in latent space of the other model
    blip_in_ccsne_latent = ccsne_vae.encode(blip_signals)
    ccsne_in_blip_latent = blip_vae.encode(ccsen_signals)

    # compute norms/umap to see how different the encodings are
    blip_in_blip_norms = np.linalg.norm(blip_in_blip_latent, axis=1)
    ccsne_in_ccsne_norms = np.linalg.norm(ccsne_in_ccsne_latent, axis=1)
    blip_in_ccsne_norms = np.linalg.norm(blip_in_ccsne_latent, axis=1)
    ccsne_in_blip_norms = np.linalg.norm(ccsne_in_blip_latent, axis=1)

    blip_col = 'C0'
    ccsne_col = 'C1'

    def make_bins(data, data2, num_bins=50):
        # Use logarithmic spacing for bins but exclude outliers based on percentiles
        d = np.concatenate([data, data2])
        min_val = np.percentile(d, 1)
        max_val = np.percentile(d, 99)
        return np.logspace(np.log10(min_val), np.log10(max_val), num_bins)

    # Plot the norms using explicit Axes and improved logspace binning
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Left subplot: Blip latent norms - use quantile_log_bins_primary to
    # place bins according to Blip density
    bins = make_bins(blip_in_blip_norms, ccsne_in_blip_norms)
    ax = axes[0]
    ax.hist(ccsne_in_blip_norms, bins=bins, alpha=0.5, lw=2, label='CCSNe in Blip Latent', color=ccsne_col, histtype='step')
    ax.hist(blip_in_blip_norms, bins=bins, alpha=0.5, lw=2, label='Blip in Blip Latent', color=blip_col, histtype='step')
    ax.set_xscale('log')
    ax.set_title("Blip Latent Space Norms")
    ax.legend()
    ax.set_xlim(1, 100)
    ax.set_xlabel("Latent Space Norm")

    # Right subplot: CCSNe latent norms (keep existing binning)
    bins = make_bins(blip_in_ccsne_norms, ccsne_in_ccsne_norms)
    ax = axes[1]
    ax.hist(ccsne_in_ccsne_norms, bins=bins, alpha=0.5, lw=2, label='CCSNe in CCSNe Latent', color=ccsne_col, histtype='step')
    ax.hist(blip_in_ccsne_norms, bins=bins, alpha=0.5, lw=2,  label='Blip in CCSNe Latent', color=blip_col, histtype='step')
    ax.set_xscale('log')
    ax.set_title("CCSNe Latent Space Norms")
    ax.legend()
    ax.set_xlim(1, 100)
    ax.set_xlabel("Latent Space Norm")

    fig.tight_layout()
    fig.savefig(f"{outdir}/outlier_norms.png")
    plt.close(fig)


    # print means, std, max, min for the generated waveforms (not latent spaces) (4SF)
    print("\n")
    print("Blip Signals Statistics : max, min, mean, std: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
        np.max(blip_signals), np.min(blip_signals), np.mean(blip_signals), np.std(blip_signals)
    ))
    print("CCSNe Signals Statistics: max, min, mean, std: {:.4f}, {:.4f}, {:.4f}, {:.4f}".format(
        np.max(ccsen_signals), np.min(ccsen_signals), np.mean(ccsen_signals), np.std(ccsen_signals)
    ))
