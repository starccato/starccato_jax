import matplotlib.pyplot as plt

from starccato_jax.waveforms import StarccatoBlip, StarccatoCCSNe


def test_waveforms(outdir):
    """
    Test the StarccatoWaveform class.
    """
    # Initialize the StarccatoBlip and StarccatoCCSNe classes
    blip = StarccatoBlip()
    ccsne = StarccatoCCSNe()

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
