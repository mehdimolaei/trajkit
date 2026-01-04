Two-point microrheology
=======================

Two-point microrheology extends *single-particle* microrheology (`Mason & Weitz <https://doi.org/10.1103/PhysRevLett.74.1250>`_) by using 
**cross-correlated displacements of tracer pairs** to probe the medium’s *bulk*, long-wavelength
viscoelastic response. The key idea (Crocker et al.) is that even if an individual tracer sits in a “soft cavity”
or has imperfect local coupling, its motion still generates (and is driven by) a **long-range deformation/flow field** 
in the surrounding continuum. Measuring how a second tracer is entrained by that field lets you separate *local* probe 
effects from the *coarse-grained* material response. 

Concretely, define tracer displacements over lag time :math:`\tau` and compute the *distinct* (pair) displacement 
correlation tensor
:math:`D_{ab}(r,\tau)=\langle \Delta r_a^{(i)}(t,\tau)\,\Delta r_b^{(j)}(t,\tau)\,\delta(r-R_{ij}(t))\rangle` with :math:`i\neq j`,
then project it into **longitudinal** (:math:`D_{rr}`) and **transverse** (:math:`D_{\perp}`) 
components relative to the separation vector.
In an incompressible continuum and for separations :math:`r\gg a`, Crocker et al. show that the Laplace-domain correlations scale as
:math:`\tilde D_{rr}(r,s)\propto (k_BT)/(r\,s\,\tilde G(s))`, with :math:`D_{\perp}= \tfrac12 D_{rr}`, and (crucially) 
**do not depend on the tracer radius or boundary condition** in this limit.
This is why two-point microrheology is robust in heterogeneous soft materials: correlated motion at separation :math:`r` 
is driven only by modes with wavelength larger than :math:`r`, so it reports the *coarse-grained* modulus rather than the local microenvironment. 
This page follows that spirit: we compute two-point correlations, map them to a distinct MSD and then to :math:`G^*(\omega)` 
… [Crocker2000]_ and [Mason1995]_.

Download the prepared sample Parquet trajectory:

.. code-block:: python

   from pathlib import Path
   from huggingface_hub import snapshot_download
   from trajkit import load_trajectory_set

   subpath = "data/parquet/experiment_001_2017-08-16/exp001_t027m_r01um_2017-08-16"

   local_root = snapshot_download(
       repo_id="m-aban/air-water",
       repo_type="dataset",
       allow_patterns=[f"{subpath}/*"],
       local_dir="hf_cache",
       local_dir_use_symlinks=False,
   )

   folder = Path(local_root) / subpath
   ts = load_trajectory_set(folder)


Compute the two-point correlations:

.. code-block:: python

   import numpy as np
   from trajkit.flow import compute_two_point_correlation

   dt_values = np.ceil(np.logspace(0, 2, 10)).astype(int)
   corr = compute_two_point_correlation(
       ts,
       dt_values=dt_values,
       r_min=10,
       r_max=500,
       n_r_bins=20,
       clip_to_shared_frames=True,
   )

.. admonition:: Plot placeholder

   Correlation vs separation (longitudinal/transverse, log–log) for a representative ``dt``.

Persist and reload to skip recomputation:

.. code-block:: python

   from trajkit.flow import save_two_point_correlation, load_two_point_correlation

   save_two_point_correlation(corr, "results/two_point_corr.npz")
   corr = load_two_point_correlation("results/two_point_corr.npz")

Map correlations to MSD and viscoelastic moduli:

.. code-block:: python

   from trajkit.flow import distinct_msd_from_two_point, compute_shear_modulus_from_msd

   msd_2p = distinct_msd_from_two_point(
       corr,
       r_min=10,
       r_max=500,
       probe_radius=1.0,   # microns
       use_linear_fit=False,
   )

   moduli = compute_shear_modulus_from_msd(
       tau=msd_2p.dt,
       msd=msd_2p.msd_transverse,  # or msd_longitudinal
       probe_radius_microns=1.0,
       dim=2,
       temperature_K=298.0,
       clip=0.03,
       smoothing_window=7,
       polyorder=2,
   )

.. admonition:: Plot placeholder

   MSD (longitudinal/transverse) vs lag time, and :math:`G'`, :math:`G''` vs :math:`\omega`.

Notes: keep ``dt_values`` in frame units when using ``frame_col`` (seconds when using ``time_col``); the saved ``.npz`` is a compressed NumPy archive for fast reloads.


.. [Crocker2000] J. C. Crocker et al., “Two-Point Microrheology of Inhomogeneous Soft Materials,” Phys. Rev. Lett. 85, 888–891 (2000).
.. [Mason1995] T. G. Mason and D. A. Weitz, “Optical Measurements of Frequency-Dependent Linear Viscoelastic Moduli of Complex Fluids,” Phys. Rev. Lett. 74, 1250–1253 (1995).