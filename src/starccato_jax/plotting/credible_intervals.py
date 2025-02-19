import jax.numpy as jnp


def pointwise_ci(post_pred, ci=0.9):
    """
    Compute pointwise credible intervals using quantiles with JAX.

    Parameters:
      - post_pred: jnp.ndarray of shape (n_samples, n_xpoints)
      - ci: float, credible interval (default: 0.9)

    Returns:
      - lower: jnp.ndarray of shape (n_xpoints,)
      - upper: jnp.ndarray of shape (n_xpoints,)
    """
    lower_quantile = (1 - ci) / 2
    upper_quantile = 1 - lower_quantile

    qtiles = jnp.quantile(
        post_pred, [lower_quantile, 0.5, upper_quantile], axis=0
    )
    return qtiles


def uniform_ci(post_pred, ci=0.9):
    """
    Compute uniform credible intervals (simultaneous credible bands) using a simulation-based approach with JAX.
    The band is centered around the pointwise median and expanded by the threshold derived from the maximum deviation.

    Parameters:
      - post_pred: jnp.ndarray of shape (n_samples, n_xpoints)
      - ci: float, credible interval (default: 0.9)

    Returns:
      - lower: jnp.ndarray of shape (n_xpoints,)
      - upper: jnp.ndarray of shape (n_xpoints,)
    """
    # Compute the pointwise median as the central curve
    median = jnp.median(post_pred, axis=0)

    # Compute the absolute deviations from the median for each sample and each x point
    deviations = jnp.abs(post_pred - median)

    # For each sample, find the maximum deviation across x points
    max_devs = jnp.max(deviations, axis=1)

    # The threshold is the ci quantile of these maximum deviations
    threshold = jnp.quantile(max_devs, ci)

    lower = median - threshold
    upper = median + threshold

    return jnp.array([lower, median, upper])
