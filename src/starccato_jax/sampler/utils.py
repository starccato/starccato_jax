import jax.numpy as jnp
import numpy as np
from scipy.stats import beta as BetaDist


def beta_spaced_samples(
    n: int, alpha: float, beta: float, min: float = 1e-8, max: float = 1
):
    """Generate N samples spaced according to a Beta(a, b) distribution."""
    linspace = jnp.linspace(min, max, n)  # Uniformly spaced values
    return jnp.array(BetaDist.ppf(linspace, alpha, beta))
