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

Train the signal VAE:

.. code-block:: bash

   uv run train-vae \
     --dataset ccsne \
     --outdir runs/ccsne_vae \
     --latent-dim 32 \
     --epochs 1000 \
     --batch-size 64

Train the glitch VAE:

.. code-block:: bash

   uv run train-vae \
     --dataset blip \
     --outdir runs/blip_vae \
     --latent-dim 32 \
     --epochs 1000 \
     --batch-size 64

Useful knobs:

* ``--latent-dim`` controls the dimension of the learned latent space.
* ``--cycles`` controls cyclical KL annealing.
* ``--use-capacity`` / ``--no-use-capacity`` switches the capacity-controlled
  KL objective.
* ``--capacity-end`` and ``--capacity-warmup-epochs`` tune the target KL ramp.

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

Interpretation:

* Low overlap in the ``ccsne`` latent space means the signal encoder places
  blip-like waveforms away from the signal manifold.
* Low overlap in the ``blip`` latent space means the glitch encoder places
  signal-like waveforms away from the glitch manifold.
* The overlap score below is the shared area between two normalized
  histograms. ``0`` means separated; ``1`` means indistinguishable by this
  one-dimensional norm summary.

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

Next Steps
----------

The latent-norm plot is a compact first pass. For publication-quality
separation studies, repeat the same cross-encoding idea on held-out validation
waveforms and add a multidimensional projection, such as PCA or UMAP, of the
latent vectors themselves.
