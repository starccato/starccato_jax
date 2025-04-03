from .vae import StarccatoVAE


class StarccatoCCSNe(StarccatoVAE):
    """
    StarccatoWaveform class that inherits from StarccatoVAE.
    This class is used to generate and reconstruct waveforms.
    """

    def __init__(self, model_dir: str = None):
        super().__init__(model_dir=model_dir)


class StarccatoBlip(StarccatoVAE):
    """
    StarccatoWaveform class that inherits from StarccatoVAE.
    This class is used to generate and reconstruct waveforms.
    """

    def __init__(self, model_dir: str = None):
        super().__init__(model_dir=model_dir)
