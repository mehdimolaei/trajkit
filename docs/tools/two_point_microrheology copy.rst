Two-point microrheology
=======================

Cross-correlate tracer displacements to recover bulk viscoelastic response, as in Mason & Weitz :cite:`mason_optical_1995,mason_dynamic_1996` and Crocker et al. :cite:`crocker_two-point_2000`. This mirrors the MATLAB flow (``twopoint.m → msdd.m → calc_G.m``) using ``trajkit.flow`` with a light, linear walkthrough.

Download the prepared Parquet trajectory set from Hugging Face (fastest path):

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

If you want the raw CSV instead, build the ``TrajectorySet`` on the fly:

.. code-block:: python

   import pandas as pd
   from huggingface_hub import hf_hub_download
   from trajkit import TrajectorySet

   csv_path = hf_hub_download(
       repo_id="m-aban/air-water",
       filename="data/csv/experiment_001_2017-08-16/exp001_t000m_r01um_2017-08-16.csv.gz",
       repo_type="dataset",
   )

   df = pd.read_csv(csv_path)
   ts = TrajectorySet.from_dataframe(
       df,
       dataset_id="air-water",
       track_id_col="id",
       position_cols=["x", "y"],
       time_col="t",  # or frame_col="frame", frame_rate_hz=...
   )

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
