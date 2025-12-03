from dataclasses import dataclass

import numpy as np

from ..data import TrainValData
from ..logging import logger


@dataclass
class Config:
    latent_dim: int = 32
    learning_rate: float = 3e-4
    epochs: int = 1000
    batch_size: int = 64
    cyclical_annealing_cycles: int = 3  # 0 for no annealing
    beta_start: float = 0.0
    beta_end: float = 0.5
    beta_ratio: float = 0.3  # fraction of each cycle spent ramping beta
    kl_free_bits: float = 0.05  # minimum KL per batch (0 disables free-bits)
    gradient_clip_value: float | None = 1.0
    learning_rate_final_mult: float = 0.1  # final lr fraction for decay schedule
    learning_rate_decay_steps: int | None = None  # None -> computed from data
    train_fraction: float = 0.8
    dataset: str = "ccsne"
    data_dim: int | None = None

    def __repr__(self):
        return (
            "Config("
            f"latent_dim={self.latent_dim}, "
            f"learning_rate={self.learning_rate}, "
            f"epochs={self.epochs}, "
            f"batch_size={self.batch_size}, "
            f"cyclical_annealing_cycles={self.cyclical_annealing_cycles}, "
            f"beta_ratio={self.beta_ratio}, "
            f"kl_free_bits={self.kl_free_bits}, "
            f"gradient_clip_value={self.gradient_clip_value}, "
            f"learning_rate_final_mult={self.learning_rate_final_mult}, "
            f"learning_rate_decay_steps={self.learning_rate_decay_steps}, "
            f"source={self.dataset}, "
            f"data_dim={self.data_dim}"
            ")"
        )

    def __post_init__(self):
        self.beta_schedule = cyclical_annealing_beta(
            n_epoch=self.epochs,
            start=self.beta_start,
            stop=self.beta_end,
            n_cycle=self.cyclical_annealing_cycles,
            ratio=self.beta_ratio,
        )

        self.data = TrainValData.load(
            train_fraction=self.train_fraction, source=self.dataset
        )

        # capture the input/output dimensionality for downstream use
        inferred_dim = self.data.train.shape[-1]
        if self.data_dim is not None and self.data_dim != inferred_dim:
            raise ValueError(
                f"Provided data_dim={self.data_dim} does not match dataset "
                f"dimension {inferred_dim}."
            )
        self.data_dim = inferred_dim

        self._batch_size_check()

    def _batch_size_check(self):
        # check fraction that will be discarded due to batch size
        n = self.data.train.shape[0]
        n_batches = self.data.train.shape[0] // self.batch_size
        n_discarded = self.data.train.shape[0] - n_batches * self.batch_size



def cyclical_annealing_beta(
    n_epoch: int,
    start: float = 0.0,
    stop: float = 1.0,
    n_cycle: int = 4,
    ratio: float = 0.5,
) -> np.ndarray:
    """
    Computes a cyclical annealing schedule for the beta parameter in a VAE.

    Parameters:
        start (float): Initial beta value (e.g., 0.0).
        stop (float): Maximum beta value (e.g., 1.0).
        n_epoch (int): Total number of epochs.
        n_cycle (int): Number of cycles for annealing. Set to 0 for no annealing.
        ratio (float): Ratio of the increasing phase within each cycle.

    Returns:
        np.ndarray: A list of beta values for each epoch.
    """
    beta_schedule = np.ones(n_epoch) * stop  # Default to max beta

    if n_cycle > 0:
        period = n_epoch / n_cycle  # Length of each cycle
        step = (stop - start) / (period * ratio)

        for c in range(n_cycle):
            v = start
            for i in range(int(period * ratio)):  # Annealing phase
                idx = int(i + c * period)
                if idx < n_epoch:
                    beta_schedule[idx] = 1.0 / (
                        1.0 + np.exp(-(v * 12.0 - 6.0))
                    )
                v += step

    return beta_schedule
