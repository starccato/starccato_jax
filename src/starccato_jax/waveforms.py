from .vae import StarccatoVAE

MODELS = ["ccsne", "blip"]


def get_model(model_type)-> StarccatoVAE:
    if model_type.lower() == 'ccsne':
        return StarccatoCCSNe()
    elif model_type.lower() == 'blip':
        return StarccatoBlip()
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'ccsne' or 'blip'")


class StarccatoCCSNe(StarccatoVAE):
    """
    StarccatoWaveform class that inherits from StarccatoVAE.
    This class is used to generate and reconstruct waveforms.
    """

    def __init__(self):
        super().__init__(model_dir="default_ccsne")
        self.model_name = "ccsne"


class StarccatoBlip(StarccatoVAE):
    """
    StarccatoWaveform class that inherits from StarccatoVAE.
    This class is used to generate and reconstruct waveforms.
    """

    def __init__(self):
        super().__init__(model_dir="default_blip")
        self.model_name = "blip"

