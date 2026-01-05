Correlational Displacement Velocimetry 
======================================

This tutorial mirrors the CDV notebooks (e.g., ``cdv_in_action.ipynb`` and ``cdv_thermal.ipynb``) in a linear, copy-pasteable form. You will:

1. Download a CSV of tracer trajectories from Hugging Face.
2. Build a ``TrajectorySet``.
3. Extract per-frame displacements at a chosen lag.
4. Plot a single-frame displacement quiver.
5. Compute an aggregated flow field with ``run_cdv`` and plot it.

Prerequisites
-------------
- Python 3.10+ with ``trajkit`` installed.
- ``huggingface_hub`` available (declared in project dependencies).
- ``matplotlib`` for plotting.

1) Download and build trajectories
----------------------------------

.. code-block:: python

   import pandas as pd
   from huggingface_hub import hf_hub_download
   from trajkit import TrajectorySet

   csv_path = hf_hub_download(
       repo_id="m-aban/air-water",
       filename="data/csv/experiment_001_2017-08-16/exp001_t027m_r01um_2017-08-16.csv.gz",
       repo_type="dataset",
   )

   df = pd.read_csv(csv_path)
   ts = TrajectorySet.from_dataframe(
       df,
       dataset_id="cdv_demo",
       track_id_col="id",
       position_cols=["x", "y"],
       time_col="t",
   )

2) Per-frame displacements at lag ``delta_t``
---------------------------------------------

.. code-block:: python

   import numpy as np
   import pandas as pd

   delta_t = 1.0  # seconds
   time_tol = 1e-3

   disp_rows = []
   for tid, tr in ts.trajectories.items():
       t = tr.time_seconds()
       if len(t) == 0:
           continue
       frames = tr.frame if tr.frame is not None else np.arange(len(t), dtype=int)
       targets = t + delta_t
       idx = np.searchsorted(t, targets, side="left")
       for i, j in enumerate(idx):
           if j >= len(t):
               continue
           if abs(t[j] - targets[i]) > time_tol:
               continue
           dx = tr.x[j] - tr.x[i]
           row = {"track_id": tid, "t": float(t[i]), "frame": int(frames[i])}
           row["x"] = float(tr.x[i, 0])
           row["y"] = float(tr.x[i, 1])
           row["dx"] = float(dx[0])
           row["dy"] = float(dx[1])
           disp_rows.append(row)

   disp_df = pd.DataFrame(disp_rows)
   print(f"Displacement rows: {len(disp_df)}")

3) Single-frame displacement quiver
-----------------------------------

.. code-block:: python

   import matplotlib.pyplot as plt

   frame0 = disp_df["frame"].min()
   df0 = disp_df[disp_df["frame"] == frame0]

   plt.figure(figsize=(6, 6))
   plt.quiver(
       df0["x"],
       df0["y"],
       df0["dx"],
       df0["dy"],
       angles="xy",
       scale_units="xy",
       scale=1.0,
       width=0.003,
   )
   plt.xlabel("x")
   plt.ylabel("y")
   plt.title(f"Displacement field at frame {frame0} (Δt={delta_t}s)")
   plt.axis("equal")
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()

4) Aggregated flow field with ``run_cdv``
-----------------------------------------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from trajkit.cdv import run_cdv

   # grid for accumulation (in the rotated source-x frame)
   x = np.linspace(-400.0, 400.0, 40)
   y = np.linspace(-400.0, 400.0, 40)
   X, Y = np.meshgrid(x, y, indexing="xy")
   grid = np.stack([X.ravel(), Y.ravel()], axis=1)

   mean, sum_w, counts, centers = run_cdv(
       ts,
       delta_t=delta_t,
       grid_centers=grid,
       max_pair_distance=600.0,  # drop far pairs
       kernel=30.0,              # hard cutoff radius
   )

   U = mean[:, 0].reshape(X.shape)
   V = mean[:, 1].reshape(Y.shape)

   plt.figure(figsize=(7, 6))
   plt.quiver(
       X,
       Y,
       U,
       V,
       angles="xy",
       scale_units="xy",
       scale=1.0,
       width=0.003,
   )
   plt.xlabel("x (rotated frame)")
   plt.ylabel("y (rotated frame)")
   plt.title("CDV flow field (aggregated over frames)")
   plt.axis("equal")
   plt.grid(True, alpha=0.3)
   plt.tight_layout()
   plt.show()

Notes
-----
- ``delta_t`` is in seconds because we used ``time_col="t"``. If you have frames, use ``frame_col`` and set ``delta_t`` in frames (or supply ``frame_rate_hz``).
- Adjust ``max_pair_distance`` and ``kernel`` to tune spatial smoothing and pair selection.
- The quiver in step 3 shows one frame’s raw displacements; the quiver in step 4 shows the aggregated flow field from ``run_cdv``.
