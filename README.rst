Starccato-JAX
=============

``starccato_jax`` trains and serves compact VAE waveform surrogates for two
classes used in Starccato analyses:

* ``ccsne``: the signal VAE for core-collapse supernova waveforms.
* ``blip``: the glitch VAE for short blip-like detector transients.

The two practical jobs are:

1. Train a VAE for one waveform family.
2. Compare the signal and glitch VAEs in latent space to see how much their
   encoded populations overlap.

The documentation below shows three complementary diagnostics:

* **Latent norm overlap** asks whether the two waveform families occupy
  similar radii after encoding.
* **Cross-reconstruction score** asks which VAE reconstructs a waveform better.
  This is the most useful simple class-separation score.
* **Latent PCA projection** asks whether the full latent vectors separate in a
  low-dimensional view.

What the Models Produce
-----------------------

The shipped default models can generate both signal-like waveforms and
glitch-like waveforms. These samples are standardized strain time series with
512 points.

.. figure:: assets/waveform_samples.png
   :alt: Generated CCSNe and blip waveform samples
   :width: 95%
   :align: center

   Example waveforms generated from the default signal and glitch VAEs.

Install
-------

.. code-block:: bash

   pip install starccato-jax

For development, clone the repository and use ``uv``:

.. code-block:: bash

   uv sync
   uv run train-vae --help

Train a VAE
-----------

The training CLI downloads/loads the requested dataset, standardizes each
waveform, trains the VAE, and writes weights plus diagnostic plots into the
output directory.

Training runs write model weights and diagnostics under the requested
``--outdir``. A typical reconstruction diagnostic looks like this:

.. figure:: model_out/reconstructions.png
   :alt: Example VAE reconstruction diagnostic
   :width: 80%
   :align: center

   Example reconstruction plot produced by the VAE diagnostics.

Train the signal VAE:

.. code-block:: bash

   uv run train-vae \
     --dataset ccsne \
     --outdir runs/ccsne_vae \
     --latent-dim 5 \
     --epochs 1000 \
     --batch-size 64

Train the glitch VAE:

.. code-block:: bash

   uv run train-vae \
     --dataset blip \
     --outdir runs/blip_vae \
     --latent-dim 5 \
     --epochs 1000 \
     --batch-size 64

Useful knobs:

* ``--latent-dim`` controls the dimension of the learned latent space. The
  default is ``5``.
* ``--cycles`` controls cyclical KL annealing.
* ``--use-capacity`` / ``--no-use-capacity`` switches the capacity-controlled
  KL objective.
* ``--capacity-end`` and ``--capacity-warmup-epochs`` tune the target KL ramp.

Why the Default Latent Dimension is 5
-------------------------------------

The VAE training default is ``latent_dim=5``. This was selected from a
dimensionality sweep over small latent sizes, using three families of
diagnostics:

* latent geometry: total correlation and maximum off-diagonal correlation;
* cross-model separability: linear discriminant ROC-AUC and Gaussian
  Jensen-Shannon divergence;
* model fit: train/validation reconstruction loss and final total KL
  divergence.

The sweep showed that the effective intrinsic dimensionality of the waveform
morphology is low, around ``4-6`` latent dimensions. Reconstruction loss
decreases rapidly up to ``z ~= 4-5`` and then saturates. Cross-model
separability is already strong by ``z ~= 4-5`` and is essentially saturated by
``z >= 6``. Larger latent spaces do not add much independent descriptive
power; instead they introduce correlated and sampler-hostile geometry.

Observed geometry:

* For small ``z ~= 4-6``, total correlation is near zero to order unity and the
  largest pairwise latent correlation is typically weak.
* For larger ``z ~= 8-16``, total correlation rises quickly and the largest
  pairwise correlations can become large.
* The total KL plateaus near the capacity target, about ``4`` nats, so larger
  latent dimensions mostly spread the same information across more correlated
  coordinates.

The chosen default, ``z=5``, sits in the stable part of the sweep:

* reconstruction loss decreases smoothly with train and validation curves close
  together;
* train/validation KL climbs smoothly and sits just below the capacity target;
* all five dimensions are active by the ``KL >= 0.1`` criterion;
* roughly ``80%`` of the KL is carried by four dimensions and ``90%`` by all
  five;
* mean absolute latent correlation is about ``0.039``;
* maximum absolute latent correlation is about ``0.106``;
* total correlation is about ``0.016``.

This makes ``z=5`` a practical compromise: enough dimensions to reconstruct and
separate the two waveform families, but small enough to keep the latent space
nearly factorised for downstream sampling and parameter-estimation workflows.

Load and Sample Models
----------------------

The package ships default weights for both waveform families:

.. code-block:: python

   from starccato_jax.waveforms import StarccatoBlip, StarccatoCCSNe

   signal_vae = StarccatoCCSNe()
   glitch_vae = StarccatoBlip()

   signal_waveforms = signal_vae.generate(n=8)
   glitch_waveforms = glitch_vae.generate(n=8)

To load a model that you trained yourself, point ``StarccatoVAE`` at the output
directory:

.. code-block:: python

   from starccato_jax import StarccatoVAE

   signal_vae = StarccatoVAE("runs/ccsne_vae")
   glitch_vae = StarccatoVAE("runs/blip_vae")

Compare Signal and Glitch Embeddings
------------------------------------

A direct comparison is to generate examples from each VAE, encode both
populations through both encoders, and compare the latent norm distributions.
This is a compact manifold-overlap check: it reduces each latent vector to its
Euclidean norm, then compares the resulting one-dimensional distributions.

This plot is useful for answering: "Does the other waveform family land in the
same broad part of this model's latent space?" It is not, by itself, the best
classifier, because two latent clouds can have similar radii while still being
separated by direction or local geometry.

.. figure:: assets/embedding_overlap.png
   :alt: Cross-encoded signal and glitch latent norm overlap
   :width: 95%
   :align: center

   Cross-encoding generated waveforms through both VAEs. With the current
   default weights, the signal latent-space norm overlap is about ``0.72`` and
   the glitch latent-space norm overlap is about ``0.005``.

Interpretation:

* Low overlap in the ``ccsne`` latent space means the signal encoder places
  blip-like waveforms away from the signal manifold.
* Low overlap in the ``blip`` latent space means the glitch encoder places
  signal-like waveforms away from the glitch manifold.
* The overlap score below is the shared area between two normalized
  histograms. ``0`` means separated; ``1`` means indistinguishable by this
  one-dimensional norm summary.
* In the current default models, the glitch VAE latent norm is highly
  discriminating, while the signal VAE latent norm is less discriminating. That
  asymmetry is useful: it says the glitch model strongly rejects signal-like
  waveforms by latent radius, but the signal model's latent radius alone is not
  enough to reject all glitch-like waveforms.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   from starccato_jax.waveforms import StarccatoBlip, StarccatoCCSNe


   def latent_norms(encoder, waveforms):
       z = np.asarray(encoder.encode(waveforms))
       return np.linalg.norm(z, axis=1)


   def histogram_overlap(a, b, bins=60):
       values = np.concatenate([a, b])
       lo, hi = np.percentile(values, [1, 99])
       edges = np.geomspace(max(lo, 1e-6), hi, bins)
       ha, _ = np.histogram(a, bins=edges, density=True)
       hb, _ = np.histogram(b, bins=edges, density=True)
       widths = np.diff(edges)
       return float(np.sum(np.minimum(ha, hb) * widths)), edges


   n = 5000
   signal_vae = StarccatoCCSNe()
   glitch_vae = StarccatoBlip()

   signal_x = signal_vae.generate(n=n)
   glitch_x = glitch_vae.generate(n=n)

   signal_in_signal = latent_norms(signal_vae, signal_x)
   glitch_in_signal = latent_norms(signal_vae, glitch_x)
   glitch_in_glitch = latent_norms(glitch_vae, glitch_x)
   signal_in_glitch = latent_norms(glitch_vae, signal_x)

   signal_space_overlap, signal_edges = histogram_overlap(
       signal_in_signal, glitch_in_signal
   )
   glitch_space_overlap, glitch_edges = histogram_overlap(
       glitch_in_glitch, signal_in_glitch
   )

   print(f"Signal latent-space overlap: {signal_space_overlap:.3f}")
   print(f"Glitch latent-space overlap: {glitch_space_overlap:.3f}")

   fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

   axes[0].hist(
       signal_in_signal,
       bins=signal_edges,
       density=True,
       histtype="step",
       lw=2,
       label="signal encoded by signal VAE",
   )
   axes[0].hist(
       glitch_in_signal,
       bins=signal_edges,
       density=True,
       histtype="step",
       lw=2,
       label="glitch encoded by signal VAE",
   )
   axes[0].set_title(f"Signal latent space, overlap={signal_space_overlap:.2f}")

   axes[1].hist(
       glitch_in_glitch,
       bins=glitch_edges,
       density=True,
       histtype="step",
       lw=2,
       label="glitch encoded by glitch VAE",
   )
   axes[1].hist(
       signal_in_glitch,
       bins=glitch_edges,
       density=True,
       histtype="step",
       lw=2,
       label="signal encoded by glitch VAE",
   )
   axes[1].set_title(f"Glitch latent space, overlap={glitch_space_overlap:.2f}")

   for ax in axes:
       ax.set_xscale("log")
       ax.set_xlabel("latent vector norm")
       ax.set_ylabel("density")
       ax.legend(frameon=False)

   fig.tight_layout()
   fig.savefig("vae_embedding_overlap.png", dpi=200)

Class Separation Score
----------------------

For classification, the strongest simple diagnostic is a cross-reconstruction
score. Reconstruct the same waveform with both VAEs and compare the mean
squared reconstruction error:

.. math::

   s(x) = \log_{10}\left(
       \frac{\mathrm{MSE}(\mathrm{glitch\ VAE}, x)}
            {\mathrm{MSE}(\mathrm{signal\ VAE}, x)}
   \right)

Positive scores favor the signal VAE. Negative scores favor the glitch VAE.
On the held-out validation waveforms sampled below, the current default
weights separate the two classes cleanly with a threshold at ``0``.

Why this works:

* Each VAE is trained to reconstruct one waveform family.
* A waveform should reconstruct better under the model whose training
  distribution it resembles.
* The log ratio makes the score symmetric and easy to threshold: ``0`` means
  both VAEs reconstruct equally well.
* The magnitude is interpretable. A score of ``+1`` means the glitch VAE has
  about ten times the reconstruction MSE of the signal VAE. A score of ``-2``
  means the glitch VAE has about one hundredth of the signal VAE's
  reconstruction MSE.

.. figure:: assets/cross_reconstruction_score.png
   :alt: Cross-reconstruction score separating held-out CCSNe and blip waveforms
   :width: 95%
   :align: center

   Cross-reconstruction error is a direct class-separation diagnostic. The
   left panel shows the signed decision score; the right panel shows the median
   reconstruction error under each model.

For the current default weights, the held-out validation sample used to make
this figure gives:

* CCSNe accuracy at threshold ``0``: ``1.000``.
* Blip accuracy at threshold ``0``: ``1.000``.
* Balanced accuracy: ``1.000``.
* Median CCSNe score: ``+1.156``.
* Median blip score: ``-1.997``.

These numbers should be treated as a model diagnostic, not a final detection
claim. For an analysis paper, recompute them on the exact held-out data,
preprocessing, and trained weights used in the experiment.

.. code-block:: python

   import numpy as np

   from starccato_jax.data import TrainValData
   from starccato_jax.waveforms import StarccatoBlip, StarccatoCCSNe


   def reconstruction_mse(model, x):
       xhat = np.asarray(model.reconstruct(x))
       return np.mean((xhat - x) ** 2, axis=1)


   signal_vae = StarccatoCCSNe()
   glitch_vae = StarccatoBlip()

   signal_x = np.asarray(TrainValData.load(source="ccsne").val)
   glitch_x = np.asarray(TrainValData.load(source="blip").val)

   signal_score = np.log10(
       reconstruction_mse(glitch_vae, signal_x)
       / reconstruction_mse(signal_vae, signal_x)
   )
   glitch_score = np.log10(
       reconstruction_mse(glitch_vae, glitch_x)
       / reconstruction_mse(signal_vae, glitch_x)
   )

   signal_accuracy = np.mean(signal_score > 0)
   glitch_accuracy = np.mean(glitch_score < 0)
   balanced_accuracy = 0.5 * (signal_accuracy + glitch_accuracy)

Latent Projection Study
-----------------------

The norm plot compresses each latent vector to one number. A second useful
view is to keep the full latent vectors, combine the signal and glitch
encodings in each VAE latent space, and project them to two dimensions with
PCA.

This plot is qualitative. It helps show whether separation is visible in the
dominant latent directions, but it should not replace a numerical score such as
the cross-reconstruction ratio. PCA is used here because it is deterministic,
dependency-free, and sufficient for a lightweight documentation diagnostic.

.. figure:: assets/latent_pca_projection.png
   :alt: PCA projection of cross-encoded signal and glitch latent vectors
   :width: 95%
   :align: center

   Two-dimensional PCA projection of the encoded latent vectors. Each panel is
   fit within one VAE latent space, then both waveform families are plotted in
   that same projected coordinate system.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   from starccato_jax.waveforms import StarccatoBlip, StarccatoCCSNe


   def encode(encoder, waveforms):
       return np.asarray(encoder.encode(waveforms))


   def pca2(z):
       z0 = z - z.mean(axis=0, keepdims=True)
       _, _, vt = np.linalg.svd(z0, full_matrices=False)
       return z0 @ vt[:2].T


   n = 2500
   signal_vae = StarccatoCCSNe()
   glitch_vae = StarccatoBlip()

   signal_x = signal_vae.generate(n=n)
   glitch_x = glitch_vae.generate(n=n)

   signal_space_xy = pca2(
       np.vstack(
           [
               encode(signal_vae, signal_x),
               encode(signal_vae, glitch_x),
           ]
       )
   )
   glitch_space_xy = pca2(
       np.vstack(
           [
               encode(glitch_vae, glitch_x),
               encode(glitch_vae, signal_x),
           ]
       )
   )

Recommended Workflow
--------------------

Use the diagnostics in this order:

1. Train one ``ccsne`` VAE and one ``blip`` VAE with matched preprocessing and
   comparable latent dimensions.
2. Check reconstruction plots from each model to confirm that the VAE learned
   the basic waveform morphology.
3. Compute cross-reconstruction scores on held-out validation waveforms. This
   is the main class-separation diagnostic.
4. Use latent norm overlap to summarize whether one model's latent space
   broadly accepts or rejects the other waveform family.
5. Use PCA projections as a visual sanity check of the full latent vectors.

If the cross-reconstruction score separates the classes but the latent norm
overlap does not, the VAEs are still useful for discrimination: the separation
is likely carried by latent direction or decoder reconstruction quality rather
than by radius alone.
