"""
Credible Intervals Overview:

1. Pointwise Credible Intervals:
   - These intervals are computed independently at each x-point.
   - A (1 - α) * 100% pointwise credible interval means that, for each fixed x,
     the true function value lies within the interval with (1 - α) probability.
   - However, when considering the entire function across all x-points,
     the probability that the entire function stays within the pointwise bands is lower than (1 - α).

2. Uniform (Simultaneous) Credible Intervals:
   - These intervals ensure that the entire function remains within the band with a probability of (1 - α).
   - Instead of computing quantiles at each x separately, the band width is determined
     by the maximum deviation from the median across all x.
   - This results in wider intervals than pointwise CIs, accounting for the joint variability
     of the function across all x-points.
   - More robust when making global statements about the function rather than just local estimates.

Summary:
- Pointwise CIs give uncertainty at individual x-points.
- Uniform CIs provide an overall confidence band for the full function, making them more conservative.

"""

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
        post_pred, jnp.array([lower_quantile, 0.5, upper_quantile]), axis=0
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


def coverage_probability(
    quantiles: jnp.array, true_signal: jnp.array
) -> float:
    """
    Compute the coverage probability of a credible interval given a true signal.

    For each element in the true signal, if it falls within the interval [lower, upper],
    it is considered "covered." The function returns the proportion of the true signal that
    falls within the interval.

    Parameters:
      - lower: jnp.ndarray, the lower bound of the credible interval (e.g., shape (n_xpoints,))
      - upper: jnp.ndarray, the upper bound of the credible interval (e.g., shape (n_xpoints,))
      - true_signal: jnp.ndarray, the true values (e.g., shape (n_xpoints,))

    Returns:
      - coverage: float, the proportion of true signal values that are within the credible interval.
    """
    lower, _, upper = quantiles
    covered = (lower <= true_signal) & (true_signal <= upper)
    return float(jnp.mean(covered))
