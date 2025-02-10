from dataclasses import dataclass


@dataclass
class Config:
    latent_dim: int = 20
    learning_rate: float = 1e-3
    epochs: int = 1000
    batch_size: int = 64
    cyclical_annealing_cycles: int = 4 # 0 for no annealing

