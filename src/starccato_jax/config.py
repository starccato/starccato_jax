from dataclasses import dataclass


@dataclass
class Config:
    latent_dim: int = 20
    learning_rate: float = 1e-3
    epochs: int = 1000
    batch_size: int = 64
    cyclical_annealing_cycles: int = 1  # 0 for no annealing
    beta_start: float = 0.0
    beta_end: float = 1.0

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
