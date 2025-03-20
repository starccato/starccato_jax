from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Union

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


@dataclass
class ModelData:
    params: Dict
    latent_dim: int


@dataclass
class Losses:
    reconstruction_loss: Union[float, List[float]]
    kl_divergence: Union[float, List[float]]
    loss: Union[float, List[float]]
    beta: Union[float, List[float]]


@dataclass
class Gradients:
    def __init__(self, n: int = 0):
        self.data = {}
        self.n = n

    def append(self, i: int, grads: List[Dict[str, Any]]):
        avg_grad_norm_per_layer = jax.tree_util.tree_map(
            lambda *x: jnp.mean(jnp.stack(x)), *grads
        )
        global_grad = jnp.sqrt(
            sum(jnp.sum(g**2) for g in jax.tree_util.tree_leaves(grads))
        )
        flattened = flatten_params(avg_grad_norm_per_layer)
        flattened["global_grad_norm"] = global_grad
        for k, v in flattened.items():
            self.data.setdefault(k, np.zeros(self.n))
            self.data[k][i] = float(v)


@dataclass
class TrainValMetrics:
    train_metrics: Losses
    val_metrics: Losses
    gradient_norms: Gradients

    @classmethod
    def for_epochs(cls, n: int) -> "TrainValMetrics":
        return cls(
            train_metrics=Losses(
                reconstruction_loss=np.zeros(n),
                kl_divergence=np.zeros(n),
                loss=np.zeros(n),
                beta=np.zeros(n),
            ),
            val_metrics=Losses(
                reconstruction_loss=np.zeros(n),
                kl_divergence=np.zeros(n),
                loss=np.zeros(n),
                beta=np.zeros(n),
            ),
            gradient_norms=Gradients(n),
        )

    def __str__(self) -> str:
        if not hasattr(self, "_i"):
            self._i = 0
        tl, vl = (
            self.train_metrics.loss[self._i],
            self.val_metrics.loss[self._i],
        )
        return f"Train Loss: {tl:.3e}, Val Loss: {vl:.3e}"

    def append(
        self,
        i: int,
        train_loss: Losses,
        val_loss: Losses,
        gradient_norms: List[Dict[str, Any]],
    ):
        self._i = i
        for metric in self.train_metrics.__annotations__.keys():
            getattr(self.train_metrics, metric)[i] = getattr(
                train_loss, metric
            )
            getattr(self.val_metrics, metric)[i] = getattr(val_loss, metric)

        if len(gradient_norms) > 0:
            self.gradient_norms.append(i, gradient_norms)

    @property
    def n(self):
        return len(self.train_metrics.loss)

    def __dict__(self):
        return {
            "train_metrics": asdict(self.train_metrics),
            "val_metrics": asdict(self.val_metrics),
            "gradient_norms": self.gradient_norms.data,
        }


def flatten_params(nested_dict: Dict[str, Any]) -> Dict[str, Array]:
    """Flatten nested parameter dictionary into single-level keys.

    {key1: {key2: value}} -> {"key1_key2": value}

    """
    flat = {}

    def _recurse(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                _recurse(v, prefix=f"{prefix}{k}_")
            else:
                flat_key = f"{prefix}{k}"
                flat[flat_key] = v

    _recurse(nested_dict)
    return flat


def extract_gradient_numbers(grad_dict: Dict[str, Any]) -> np.ndarray:
    """
    Extract all numerical values from a nested gradient dictionary.

    Args:
        grad_dict: Nested dictionary containing gradient values

    Returns:
        JAX array of all extracted numerical values
    """
    numbers = []

    def _recursive_extract(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                _recursive_extract(v)
        elif isinstance(obj, jax.Array):
            numbers.append(jnp.asarray(obj).item())  # Convert to float

    _recursive_extract(grad_dict)
    return np.array(numbers)
