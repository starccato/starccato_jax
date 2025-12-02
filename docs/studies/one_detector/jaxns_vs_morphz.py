"""
Quick comparison of evidence estimates on a toy logistic-regression problem.

We generate synthetic Bernoulli data, then:
1) run numpyro's NestedSampler (JAXNS backend) and read its logZ,
2) run a *separate* NUTS chain + morphZ to estimate logZ from those posterior samples.

The goal is a cheap sanity check near `one_detector_analysis.py` that also reports
runtime for both approaches.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from jax import random
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.contrib.nested_sampling import NestedSampler
from numpyro.infer import MCMC, NUTS, log_likelihood
from morphZ import evidence as morph_evidence

jax.config.update("jax_enable_x64", True)

TRUE_COEFS = jnp.array([1.0, 2.0, 3.0])
TRUE_INTERCEPT = 0.0
HERE = Path(__file__).parent
DEFAULT_OUTDIR = HERE / "out" / "jaxns_vs_morphz"


def logistic_model(data, labels):
    coefs = numpyro.sample("coefs", dist.Normal(0, 1).expand([TRUE_COEFS.shape[0]]))
    intercept = numpyro.sample("intercept", dist.Normal(0.0, 10.0))
    logits = (coefs * data).sum(-1) + intercept
    numpyro.sample("y", dist.Bernoulli(logits=logits), obs=labels)


def make_synthetic_data(seed: int, n_obs: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    k_data, k_label = random.split(random.PRNGKey(seed))
    data = random.normal(k_data, (n_obs, TRUE_COEFS.shape[0]))
    logits = (TRUE_COEFS * data).sum(-1) + TRUE_INTERCEPT
    labels = dist.Bernoulli(logits=logits).sample(k_label)
    return data, labels


def make_log_posterior_fn(data: jnp.ndarray, labels: jnp.ndarray) -> Callable[[np.ndarray], float]:
    def _fn(theta_vec: np.ndarray) -> float:
        intercept = jnp.asarray(theta_vec[0])
        coefs = jnp.asarray(theta_vec[1:])
        lp = dist.Normal(0.0, 10.0).log_prob(intercept) + jnp.sum(dist.Normal(0, 1).log_prob(coefs))
        logits = (coefs * data).sum(-1) + intercept
        ll = jnp.sum(dist.Bernoulli(logits=logits).log_prob(labels))
        return float(lp + ll)

    return _fn


def run_jaxns(data: jnp.ndarray, labels: jnp.ndarray, num_live: int, max_samples: int, num_draws: int, seed: int):
    rng = random.PRNGKey(seed)
    ns = NestedSampler(
        logistic_model,
        constructor_kwargs={"num_live_points": num_live, "max_samples": float(max_samples), "verbose": False},
        termination_kwargs={"max_samples": float(max_samples), "dlogZ": 0.1},
    )
    t0 = time.perf_counter()
    ns.run(rng, data, labels)
    results = ns._results
    samples = ns.get_samples(random.split(rng)[1], num_samples=num_draws)
    elapsed = time.perf_counter() - t0
    logz = float(results.log_Z_mean)
    logz_err = float(results.log_Z_uncert)
    return logz, logz_err, elapsed, samples


def _log_prior(samples):
    lp_intercept = dist.Normal(0.0, 10.0).log_prob(samples["intercept"])
    lp_coefs = jnp.sum(dist.Normal(0, 1).log_prob(samples["coefs"]), axis=1)
    return lp_intercept + lp_coefs


def run_morphz_with_nuts(
    data: jnp.ndarray,
    labels: jnp.ndarray,
    log_post_fn: Callable[[np.ndarray], float],
    num_warmup: int,
    num_samples: int,
    n_resamples: int,
    seed: int,
    outdir: Path,
):
    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.PRNGKey(seed)
    mcmc = MCMC(NUTS(logistic_model), num_warmup=num_warmup, num_samples=num_samples, progress_bar=False)
    # Elapsed time covers NUTS sampling + morphZ evidence estimation.
    t0 = time.perf_counter()
    mcmc.run(rng, data, labels)
    samples = mcmc.get_samples()
    ll = log_likelihood(logistic_model, samples, data=data, labels=labels)["y"]
    ll_total = jnp.sum(ll, axis=1)
    log_post_vals = np.array(_log_prior(samples) + ll_total)
    theta = np.concatenate(
        [np.array(samples["intercept"])[..., None], np.array(samples["coefs"])],
        axis=1,
    )
    logz = morph_evidence(
        post_samples=theta,
        log_posterior_values=log_post_vals,
        log_posterior_function=log_post_fn,
        n_resamples=n_resamples,
        morph_type="pair",
        n_estimations=1,
        kde_bw="silverman",
        verbose=False,
        output_path=str(outdir),
    )
    logz = np.asarray(logz)
    elapsed = time.perf_counter() - t0
    logz_mean = float(np.mean(logz[:, 0]))
    logz_err = float(np.mean(logz[:, 1]))
    return logz_mean, logz_err, elapsed


def main():
    parser = argparse.ArgumentParser(description="Compare JAXNS nested sampling vs morphZ LnZ on a toy problem.")
    parser.add_argument("--seed", type=int, default=0, help="Base PRNG seed.")
    parser.add_argument("--n-obs", type=int, default=800, help="Number of synthetic observations.")
    parser.add_argument("--num-live", type=int, default=200, help="Live points for nested sampling.")
    parser.add_argument("--max-samples", type=int, default=4000, help="Max nested-sampling likelihood evals.")
    parser.add_argument("--num-jaxns-draws", type=int, default=800, help="Posterior draws to request from JAXNS.")
    parser.add_argument("--num-warmup", type=int, default=300, help="NUTS warmup steps for morphZ path.")
    parser.add_argument("--num-samples", type=int, default=600, help="NUTS samples for morphZ path.")
    parser.add_argument("--n-resamples", type=int, default=400, help="morphZ resamples.")
    args = parser.parse_args()

    data, labels = make_synthetic_data(args.seed, args.n_obs)
    log_post_fn = make_log_posterior_fn(data, labels)

    logz_ns, logz_ns_err, t_ns, ns_samples = run_jaxns(
        data,
        labels,
        num_live=args.num_live,
        max_samples=args.max_samples,
        num_draws=args.num_jaxns_draws,
        seed=args.seed + 2,
    )

    logz_morph, logz_morph_err, t_morph = run_morphz_with_nuts(
        data,
        labels,
        log_post_fn=log_post_fn,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        n_resamples=args.n_resamples,
        seed=args.seed + 50,
        outdir=DEFAULT_OUTDIR,
    )

    coefs_mean = np.mean(np.array(ns_samples["coefs"]), axis=0)
    intercept_mean = float(np.mean(np.array(ns_samples["intercept"])))

    print("==== Synthetic logistic regression ====")
    print(f"True coefs:      {np.asarray(TRUE_COEFS)}")
    print(f"JAXNS coefs mean {coefs_mean}")
    print(f"JAXNS intercept  {intercept_mean:.4f}")
    print(f"JAXNS logZ       {logz_ns:.2f} ± {logz_ns_err:.2f} (time {t_ns:.1f}s)")
    print(
        f"morphZ logZ      {logz_morph:.2f} ± {logz_morph_err:.2f} "
        f"(time {t_morph:.1f}s incl. NUTS + morphZ, posterior from NUTS)"
    )
    if t_morph > 0:
        print(f"Speed ratio (JAXNS/morphZ): {t_ns / t_morph:.2f}x")


if __name__ == "__main__":
    main()
