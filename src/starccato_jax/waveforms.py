from .vae import StarccatoVAE


class StarccatoCCSNe(StarccatoVAE):
    """
    StarccatoWaveform class that inherits from StarccatoVAE.
    This class is used to generate and reconstruct waveforms.
    """

    def __init__(self):
        super().__init__(model_dir="default_ccsne")


class StarccatoBlip(StarccatoVAE):
    """
    StarccatoWaveform class that inherits from StarccatoVAE.
    This class is used to generate and reconstruct waveforms.
    """

    def __init__(self):
        super().__init__(model_dir="default_blip")
