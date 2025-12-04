from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict

from .model import VAE


def compute_latent_stats(
    params,
    model: VAE,
    data: jnp.ndarray,
    rng: jax.random.PRNGKey,
    kl_threshold: float = 0.1,
):
    """
    Compute KL per dim, active counts, dims for 80/90/100% KL, correlation stats, and total correlation.

    Returns a dict with keys:
        kl_per_dim, active, n80, n90, n100, mean_abs_corr, max_abs_corr, total_corr
    """
    # Handle FrozenDict params
    if isinstance(params, FrozenDict):
        params = params.unfreeze()

    _, mean, logvar = model.apply({"params": params}, data, rng, True, method=model.__call__)
    kl_per_dim = 0.5 * (jnp.exp(logvar) + mean**2 - 1.0 - logvar).mean(axis=0)

    kl_sorted = jnp.sort(kl_per_dim)[::-1]
    kl_cumsum = jnp.cumsum(kl_sorted)
    kl_frac = kl_cumsum / (kl_cumsum[-1] + 1e-8)
    n80 = int(jnp.argmax(kl_frac >= 0.8)) + 1
    n90 = int(jnp.argmax(kl_frac >= 0.9)) + 1
    n100 = int(len(kl_per_dim))
    active = int(jnp.sum(kl_per_dim >= kl_threshold))

    z = model.apply({"params": params}, data, rng, method=model.encode)
    z_centered = z - jnp.mean(z, axis=0, keepdims=True)
    cov = (z_centered.T @ z_centered) / (z.shape[0] - 1)
    var = jnp.diag(cov)
    std = jnp.sqrt(var + 1e-8)
    corr = cov / (std[:, None] * std[None, :] + 1e-8)
    off_diag = corr - jnp.eye(corr.shape[0])
    mean_abs_corr = float(jnp.mean(jnp.abs(off_diag)))
    max_abs_corr = float(jnp.max(jnp.abs(off_diag)))
    sign_full, logdet_full = jnp.linalg.slogdet(cov + 1e-6 * jnp.eye(cov.shape[0]))
    sign_diag, logdet_diag = jnp.linalg.slogdet(jnp.diag(var + 1e-6))
    total_corr = 0.5 * float(logdet_diag - logdet_full)

    return dict(
        kl_per_dim=np.array(kl_per_dim),
        active=active,
        n80=n80,
        n90=n90,
        n100=n100,
        mean_abs_corr=mean_abs_corr,
        max_abs_corr=max_abs_corr,
        total_corr=total_corr,
    )
