# VAE retraining contract and NUTS validation

This study fixes two inference-facing issues in the legacy CCSNE and blip VAEs,
selects replacement training settings, and tests the resulting models with
four-chain NUTS on representative LVK data.

## Changes

1. Training waveforms were already standardized independently to zero mean and
   unit standard deviation. Newly trained decoders now enforce the same
   zero-mean, unit-RMS contract on every generated waveform. Downstream
   ``exp(log_amp)`` therefore controls amplitude, while the latent variables
   control morphology. Legacy artifacts that lack the new metadata continue to
   use their original raw decoder output.
2. The old KL implementation averaged over both the batch and latent axes. A
   nominal capacity of 4 nats in a five-dimensional model therefore admitted
   approximately 20 total nats. The corrected implementation sums over latent
   dimensions and averages only over the batch. Capacity is now expressed in
   total nats per waveform.
3. Training initialization and the train/validation split have separate
   reproducible seeds (``seed`` and ``data_seed``).
4. ``StarccatoVAE.encode_mean`` exposes the deterministic encoder mean for use
   as an optional inexpensive initializer.

The capacity loss is an upper-budget hinge,
``reconstruction + beta_capacity * max(KL - capacity, 0)``. It is not an exact
constraint forcing KL to equal the requested capacity.

## Selection study

The pilot compared corrected total-KL capacities 4, 8, and 12 at latent
dimension 5. Capacities 4 and 8 noticeably underfit. Capacity 12 restored
held-out reconstruction quality while keeping all five dimensions active and
weakly correlated. Three training seeds were then compared on the same
``data_seed=0`` validation split.

The selected artifacts are:

| family | epochs | training seed | saved/best epoch | deterministic validation MSE | total KL (nats) | active dims | max abs latent corr | total corr |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CCSNE | 2000 | 1 | 1483 | 0.18917 | 11.464 | 5/5 | 0.159 | 0.033 |
| blip | 1000 | 0 | 978 | 0.03266 | 11.921 | 5/5 | 0.107 | 0.015 |

The 2000-epoch blip comparison early-stopped after 1408 recorded epochs and had
worse deterministic validation MSE (0.03586), so the 1000-epoch artifact was
retained. These are focused validation results on the fixed project split, not
claims about population-level waveform fidelity.

Run both selected trainings from the repository root with:

```bash
./scripts/run_training_for_defaults.sh
```

The local selected artifacts are ``out_ccsne/model.h5`` and
``out_blip/model.h5``. Their SHA-256 hashes for the 2026-07-18 run are:

```text
bb29a84f913e8dde92e3e5f855606fcc4fbe036cf74bd11dadc18a157b1800aa  out_ccsne/model.h5
2ce2b017da79a6497c622617a8cf12a176a001979631ffec012c8515907a6314  out_blip/model.h5
```

The weight files are ignored by Git. Uploading/replacing the public default
weights is a separate release action; code should be merged and both
``starccato_jax`` and ``starccato_lvk`` tested before publishing them.

For v0.3.0 the weights are published under versioned URLs at
``weights/starcatto_jax/v0.3.0/``. The unversioned v0.2.x files remain in place
so older installations never attempt to load artifacts with the new metadata
and decoder contract.

## Cheap NUTS initialization

``starccato_lvk.analysis.jim_likelihood.find_multistart_map`` performs bounded
L-BFGS optimization over the five latent variables and ``log_amp``. It starts
from the zero latent vector plus prior draws and cycles through a small
``log_amp`` grid. Twenty starts cost far less than nested sampling in this
six-dimensional problem. The returned values can be passed to NumPyro with
``init_to_value``.

```python
import jax
import jax.numpy as jnp
from numpyro.infer import init_to_value

from starccato_lvk.analysis.jim_likelihood import (
    find_multistart_map,
    run_numpyro_sampling,
)

map_result = find_multistart_map(
    likelihood,
    latent_names=waveform.latent_names,
    fixed_params=fixed_params,
    rng_key=jax.random.PRNGKey(10),
    log_amp_sigma=5.0,
    num_starts=20,
    noise_scale_marginal=True,
)
result = run_numpyro_sampling(
    likelihood,
    latent_names=waveform.latent_names,
    fixed_params=fixed_params,
    rng_key=jax.random.PRNGKey(11),
    log_amp_sigma=5.0,
    num_chains=4,
    target_accept_prob=0.95,
    init_strategy=init_to_value(
        values={k: jnp.asarray(v) for k, v in map_result.values.items()}
    ),
    noise_scale_marginal=True,
)
```

The deterministic encoder mean can be included as an additional start when a
clean, centered, standardized estimate of the transient morphology is
available. For a noisy event, multistart MAP is safer than trusting the encoder
alone because the encoder was trained on clean waveforms.

This initialization finds high-density basins; it does **not** measure their
posterior probability. If the attempts converge to separated solutions with
similar log density, run NUTS from each basin and report the ambiguity. Ordinary
NUTS cannot determine relative weights of disconnected modes.

## Real-data NUTS checks

Both selected artifacts were checked with the noise-scale-marginalized
likelihood, 20-start MAP initialization, four chains, 800 warmup steps, 800
samples per chain, target acceptance 0.95, and maximum tree depth 12.

| case | divergences | max R-hat | minimum ESS | E-BFMI range | max leapfrog steps | tree-depth saturation |
|---|---:|---:|---:|---:|---:|---:|
| CCSNE injection in real L1 noise | 0 | 1.0025 | 107.9 | 0.915-0.997 | 191 | 0 |
| real L1 blip at GPS 1260776833 | 0 | 1.0132 | 185.2 | 0.641-0.930 | 127 | 0 |

The blip posterior still has other lower-density local optima. The MAP start
places all chains in the best basin found and gives a clean local posterior.
This is enough for waveform reconstruction and conditional parameter
uncertainty, but not evidence estimation or global mode weighting.
