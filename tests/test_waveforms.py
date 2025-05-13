import matplotlib.pyplot as plt

from starccato_jax.data.default_weights import get_default_weights_dir
from starccato_jax.waveforms import StarccatoBlip, StarccatoCCSNe


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
