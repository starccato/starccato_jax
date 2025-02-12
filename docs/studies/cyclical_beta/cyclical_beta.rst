Investigating different Betas
==============================

We try three cases:

- Case 1: Beta is constant at 0 (ie KL divergence isnt used)
- Case 2: Beta is constant at 1 (vanilla VAE)
- Case 3: Beta is cyclical between 0 and 1 and stays at 1 for 50% of the time
- Case 4: Beta is monotonic, starts at 0, ramps up till 1 for firsst 50% of the time.



.. list-table::
   :header-rows: 1
   :width: 170%
   :widths: 5 30 30 30 30

   * - Plot Type
     - Case 1
     - Case 2
     - Case 3
     - Case 4
   * - Loss Plot
     - .. image:: out_models/beta_0/loss.png
           :alt: Loss Plot Case 1
           :align: center
     - .. image:: out_models/beta_1/loss.png
           :alt: Loss Plot Case 2
           :align: center
     - .. image:: out_models/beta_cyclical/loss.png
           :alt: Loss Plot Case 3
           :align: center
     - .. image:: out_models/beta_monotonic/loss.png
           :alt: Loss Plot Case 4
           :align: center
   * - CI Plot
     - .. image:: out_models/beta_0/ci_plot.png
           :alt: CI Plot Case 1
           :align: center
     - .. image:: out_models/beta_1/ci_plot.png
           :alt: CI Plot Case 2
           :align: center
     - .. image:: out_models/beta_cyclical/ci_plot.png
           :alt: CI Plot Case 3
           :align: center
     - .. image:: out_models/beta_monotonic/ci_plot.png
           :alt: CI Plot Case 4
           :align: center
   * - Reconstructions
     - .. image:: out_models/beta_0/training_reconstructions.gif
           :alt: Training Reconstructions Case 1
           :align: center
     - .. image:: out_models/beta_1/training_reconstructions.gif
           :alt: Training Reconstructions Case 2
           :align: center
     - .. image:: out_models/beta_cyclical/training_reconstructions.gif
           :alt: Training Reconstructions Case 3
           :align: center
     - .. image:: out_models/beta_monotonic/training_reconstructions.gif
           :alt: Training Reconstructions Case 4
           :align: center



Code
----

.. literalinclude:: cyclical_beta.py
   :language: python
