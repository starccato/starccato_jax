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

## 2026-07-18 hardening pass

The follow-up inference audit made the following changes before the next
training campaign:

1. Validation and early stopping now decode the encoder mean. Previously,
   ``deterministic=True`` disabled decoder dropout but still drew a random
   latent sample, making checkpoint selection noisy.
2. Inference-only model loading no longer constructs ``Config`` or loads the
   training dataset.
3. New artifacts record the artifact schema, best and recorded epochs,
   validation reconstruction loss, training/data seeds, and checksums of the
   standardized training and validation arrays.
4. Default weights are immutable versioned downloads with pinned SHA-256
   checksums and atomic cache replacement. The 24-hour refresh was removed.
5. Training rejects non-finite or zero-variance waveform rows before
   standardization.
6. The latent sweep now supports repeated training seeds and reports
   deterministic time-domain error, 300--800 Hz mismatch, log-spectral
   distance, and peak-time error.
7. ``run_decoder_diagnostics.py`` records local decoder singular values,
   condition numbers, effective rank, latent round-trip error, and distant
   latent points with nearly identical decoded waveforms.

Focused baseline diagnostics using the currently published v0.3.0 weights are
stored in ``results/2026-07-18-*-decoder-diagnostics.json``. At 128 prior,
validation, posterior, or MAP points per reported group:

| family | prior median Jacobian condition | posterior median condition | minimum effective rank | prior near-collisions |
|---|---:|---:|---:|---:|
| CCSNE | 6.1 | 3.7 | 5/5 | 0/8076 |
| blip | 141.5 | 51.2 | 5/5 | 0/8076 |

The blip decoder is full rank but has one much less responsive latent
direction. This is a conditioning warning, not by itself evidence that NUTS is
invalid: the existing MAP-initialized four-chain blip run had no divergences
and acceptable local convergence. The multistart search's two highest blip
solutions are only 0.08 apart in latent Euclidean distance and represent the
same basin. Other optima are at least 18 log-density units below the best. We
therefore defer a cycle-consistency regularizer until the z=4/5/6/8 repeated-
seed study tests whether the blip model is simply over-parameterized.

The fidelity results also show why selection cannot use time-domain MSE alone.
The median 300--800 Hz mismatch was 0.040 for CCSNE and 0.241 for blips, while
the 90th percentiles were 0.683 and 0.816. These are unweighted, sign-invariant
band overlaps on the fixed validation split, not detector-PSD-weighted
population claims. The high tails require inspection in the latent sweep and
should be compared with PSD-weighted mismatch before final model selection.

The current v0.3.0 weights predate deterministic checkpoint selection and do
not contain the new provenance block. They remain valid NUTS-tested baselines;
the final replacement weights should be trained only after the focused latent
study selects a configuration.

### Hardened candidate selection

The one-seed 1000-epoch pilot compared z=4, 5, 6, and 8 for both families.
Increasing beyond five dimensions produced little or no improvement in the
300--800 Hz mismatch while sharply worsening decoder conditioning. At z=8,
the median prior Jacobian condition numbers were 166 for CCSNE and 424 for
blips, and only 7/8 and 6/8 dimensions were active. z=4 underfit both families.
The study therefore retains z=5.

Three z=5 training seeds were then run under deterministic checkpoint
selection. All five dimensions remained active. The mean validation MSE was
0.2010 +/- 0.0025 for CCSNE and 0.03391 +/- 0.00122 for blips.

For CCSNE, extending seed 1 to 2000 requested epochs reduced validation MSE to
0.1890, but its multistart screen found a distinct solution only 4.41
log-density units below the best. Four-chain NUTS did not converge even after
2000 warmup and 2000 retained samples per chain: max R-hat was 1.16 and the
minimum ESS was 13.7. The 1000-epoch seed-2 model instead placed the next
distinct MAP solution 14.69 log-density units below the best and passed the
800/800 NUTS check with zero divergences, max R-hat 1.0104, minimum ESS 367.7,
E-BFMI 0.85--0.97, maximum 95 leapfrog steps, and no tree-depth saturation.
It is selected despite its higher validation MSE of 0.2037.

For blips, reconstruction alone favored seed 0, but its real-glitch multistart
screen found two distinct solutions within 2.3 log-density units. Four
MAP-initialized NUTS chains split between these basins: max R-hat was 1.71 and
the minimum ESS was 3.1 despite zero divergences.

The separated seed-0 basins decode to distinct shapes (roughly 0.18--0.23
normalized mismatch), rather than distant latent points producing the same
waveform. This is genuine likelihood/model multimodality, not decoder folding,
so a cycle-consistency penalty is not justified by this event.

Blip seed 2 had no distinct MAP solution within 30.9 log-density units and
passed the full real-L1-glitch NUTS check: zero divergences, max R-hat 1.0012,
minimum ESS 573, E-BFMI 0.73--0.85, maximum 127 leapfrog steps, and no
tree-depth saturation. It is selected over seed 0 despite a validation-MSE
increase from 0.03252 to 0.03476.

The hardened candidates are:

| family | seed | best / recorded epoch | deterministic validation MSE | SHA-256 |
|---|---:|---:|---:|---|
| CCSNE | 2 | 998 / 1000 | 0.20373 | ``edf65676b43f0b015dfff4a5613b9189b1f2bc19159e0a599d5db5595abea0be`` |
| blip | 2 | 969 / 1000 | 0.03476 | ``4f7d590bfe5f23f07785fa6b16b68ed8115a8ad1e8333c87341b7731c27c9879`` |

The compact machine-readable decision record is
``results/2026-07-18-hardened-selection-summary.json``. Large model and NUTS
sample files remain ignored; the result JSONs store hashes, source paths, and
diagnostics.

The selected files are published immutably at
``weights/starcatto_jax/v0.4.0/``. ``src/starccato_jax/data/urls.py`` pins both
URLs and SHA-256 hashes. The publication helper accepts explicit candidate
paths and refuses to overwrite an existing version directory.

The capacity loss is an upper-budget hinge,
``reconstruction + beta_capacity * max(KL - capacity, 0)``. It is not an exact
constraint forcing KL to equal the requested capacity.

## v0.3.0 baseline selection study

The pilot compared corrected total-KL capacities 4, 8, and 12 at latent
dimension 5. Capacities 4 and 8 noticeably underfit. Capacity 12 restored
held-out reconstruction quality while keeping all five dimensions active and
weakly correlated. Three training seeds were then compared on the same
``data_seed=0`` validation split.

The v0.3.0 artifacts were:

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

The weight files are ignored by Git. For v0.3.0 they were published under
versioned URLs at
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

Both v0.4.0 artifacts were checked with the noise-scale-marginalized
likelihood, 20-start MAP initialization, four chains, 800 warmup steps, 800
samples per chain, target acceptance 0.95, and maximum tree depth 12.

| case | divergences | max R-hat | minimum ESS | E-BFMI range | max leapfrog steps | tree-depth saturation |
|---|---:|---:|---:|---:|---:|---:|
| CCSNE injection in real L1 noise | 0 | 1.0104 | 367.7 | 0.849-0.975 | 95 | 0 |
| real L1 blip at GPS 1260776833 | 0 | 1.0012 | 573.4 | 0.732-0.852 | 127 | 0 |

The blip posterior still has other lower-density local optima. The MAP start
places all chains in the best basin found and gives a clean local posterior.
This is enough for waveform reconstruction and conditional parameter
uncertainty, but not evidence estimation or global mode weighting.
