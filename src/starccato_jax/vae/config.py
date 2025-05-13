from dataclasses import dataclass

import numpy as np

from ..data import BlipDataset, CCSNeDataset, TrainValData
from ..logging import logger


@dataclass
class Config:
    latent_dim: int = 20
    learning_rate: float = 1e-3
    epochs: int = 1000
    batch_size: int = 64
    cyclical_annealing_cycles: int = 1  # 0 for no annealing
    beta_start: float = 0.0
    beta_end: float = 1.0
    train_fraction: float = 0.8
    dataset: str = "richers_ccsne"  # or "gan"

    def __repr__(self):
        return (
            "Config("
            f"latent_dim={self.latent_dim}, "
            f"learning_rate={self.learning_rate}, "
            f"epochs={self.epochs}, "
            f"batch_size={self.batch_size}, "
            f"cyclical_annealing_cycles={self.cyclical_annealing_cycles}"
            ")"
        )

    def __post_init__(self):
        self.beta_schedule = cyclical_annealing_beta(
            n_epoch=self.epochs,
            start=self.beta_start,
            stop=self.beta_end,
            n_cycle=self.cyclical_annealing_cycles,
        )

        if self.dataset == "richers_ccsne":
            self.data = CCSNeDataset.load(train_fraction=self.train_fraction)
        elif self.dataset == "blip":
            self.data = BlipDataset.load(train_fraction=self.train_fraction)
        self._batch_size_check()

    def _batch_size_check(self):
        # check fraction that will be discarded due to batch size
        n = self.data.train.shape[0]
        n_batches = self.data.train.shape[0] // self.batch_size
        n_discarded = self.data.train.shape[0] - n_batches * self.batch_size
        if n_discarded > 0:
            logger.warning(
                f"Every epoch will discard {n_discarded}/{n} training samples due to batch size {self.batch_size}"
            )


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
