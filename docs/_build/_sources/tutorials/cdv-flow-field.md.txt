# Flow field from a trajectory set (CDV)

This tutorial shows how to build a flow field from tracked particle trajectories using the
conditional displacement/velocity (CDV) utilities. It mirrors the `examples/notebooks/cdv_thermal.ipynb`
workflow in a linear, copy-pasteable form.

## Prerequisites
- A table of trajectories with columns like `id`, `x`, `y`, `t` (or `frame`). Units should be consistent
  (e.g., pixels for positions; frames for time).
- Python 3.10+, `trajkit` installed, and `matplotlib` for plotting.

## 1) Load data and build a `TrajectorySet`
You can load from a remote CSV or an existing saved dataset (Parquet) via `load_trajectory_set`.

```python
import pandas as pd
from trajkit import TrajectorySet, load_trajectory_set

# Option A: remote/local CSV
csv_url = "https://your-database.org/path/to/res_xyti_time11.csv"  # replace with your URL
df_whole = pd.read_csv(csv_url)

# Optionally downselect for a quick run
frame_max = 120
df = df_whole[df_whole["t"] < frame_max].copy()

trajset = TrajectorySet.from_dataframe(
    df,
    dataset_id="cdv_demo",
    track_id_col="id",
    position_cols=["x", "y"],
    time_col="t",
    frame_col="t",  # set this if you have frame numbers
    units={"t": "frame", "x": "pixel"},
)

# Option B: load a saved Parquet dataset
# trajset = load_trajectory_set("examples/datasets/demo_brownian_2d", frame_rate_hz=20.0)
```

## 2) Assemble per-frame displacements
Choose a lag `delta_t_frames` (in frames) and compute displacements for each trajectory.

```python
import numpy as np
import pandas as pd

delta_t_frames = 1.0
time_tol = 1e-3

disp_rows = []
for tid, tr in trajset.trajectories.items():
    t = tr.time_seconds()
    if len(t) == 0:
        continue
    frames = tr.frame if tr.frame is not None else np.arange(len(t), dtype=int)
    targets = t + delta_t_frames
    idx = np.searchsorted(t, targets, side="left")
    for i, j in enumerate(idx):
        if j >= len(t):
            continue
        if abs(t[j] - targets[i]) > time_tol:
            continue
        dx = tr.x[j] - tr.x[i]
        row = {"track_id": tid, "t": float(t[i]), "frame": int(frames[i])}
        for k in range(tr.D):
            row[f"x{k}"] = float(tr.x[i, k])
            row[f"dx{k}"] = float(dx[k])
        disp_rows.append(row)

disp_df = pd.DataFrame(disp_rows)
print(f"Displacement rows: {len(disp_df)}")
```

## 3) Set up CDV grid, pair filter, and accumulator

```python
import numpy as np
from pathlib import Path
from trajkit.cdv import correlation_batch, CorrelationEnsembleAccumulator, distance_threshold_pair_filter

position_cols = [c for c in disp_df.columns if c.startswith("x") and not c.startswith("dx")]
motion_cols = [c for c in disp_df.columns if c.startswith("dx")]

max_pair_distance = 600  # pixels; drop tracer/source pairs farther than this
pair_filter = distance_threshold_pair_filter(max_pair_distance)

# Grid in relative-position space
x = np.linspace(-400.0, 400.0, 100)
y = np.linspace(-400.0, 400.0, 100)
X, Y = np.meshgrid(x, y, indexing="xy")
grid_centers = np.stack([X.ravel(), Y.ravel()], axis=1)

plot_dir = Path("results/cdv_frames_tutorial")
plot_dir.mkdir(parents=True, exist_ok=True)

ensemble = CorrelationEnsembleAccumulator(
    grid_centers,
    kernel=30.0,  # hard cutoff radius in rel_pos space
    value_fn=lambda rel, tracer, source, meta_row: tracer,
    weight_fn=lambda rel, tracer, source, meta_row: np.linalg.norm(source),
)
```

## 4) (Optional) Plot helper for per-frame snapshots

```python
import matplotlib.pyplot as plt

def plot_cdv_frame(
    frame_idx,
    disp_temp,
    batch_rotated,
    ensemble_accumulator,
    X,
    Y,
    *,
    save_dir=None,
    quiver_stride=6,
    motion_quiver_scale=1.0,
    flow_quiver_scale=80,
):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_disp, ax_corr, ax_flow = axes

    # Displacements at this frame
    ax_disp.quiver(
        disp_temp["x0"],
        disp_temp["x1"],
        disp_temp["dx0"],
        disp_temp["dx1"],
        angles="xy",
        scale_units="xy",
        scale=motion_quiver_scale,
        color="tab:blue",
        alpha=0.9,
    )
    ax_disp.scatter(disp_temp["x0"], disp_temp["x1"], s=16, color="tab:blue", alpha=0.9)
    ax_disp.set_title(f"Displacements (frame {frame_idx})")
    ax_disp.set_aspect("equal")

    # Rotated correlations
    ax_corr.quiver(
        batch_rotated.relative_positions[:, 0],
        batch_rotated.relative_positions[:, 1],
        batch_rotated.tracer_motion[:, 0],
        batch_rotated.tracer_motion[:, 1],
        angles="xy",
        scale_units="xy",
        scale=motion_quiver_scale,
        color="tab:orange",
        alpha=0.6,
    )
    ax_corr.set_title("Rotated tracer motion vs. rel. pos.")
    ax_corr.set_aspect("equal")
    ax_corr.set_xlabel("r1")
    ax_corr.set_ylabel("r2")

    # Ensemble flow so far
    mean, _, _ = ensemble_accumulator.finalize()
    U = mean[:, 0].reshape(X.shape)
    V = mean[:, 1].reshape(Y.shape)
    mag = np.ma.array(np.hypot(U, V), mask=np.isnan(U) | np.isnan(V))

    cf = ax_flow.contourf(X, Y, mag, levels=20, cmap="YlGnBu", alpha=0.85)
    step = max(1, quiver_stride)
    ax_flow.quiver(
        X[::step, ::step],
        Y[::step, ::step],
        np.nan_to_num(U)[::step, ::step],
        np.nan_to_num(V)[::step, ::step],
        color="white",
        scale=flow_quiver_scale,
        alpha=0.9,
    )
    ax_flow.set_title(f"Ensemble flow <= frame {frame_idx}")
    ax_flow.set_aspect("equal")
    cbar = fig.colorbar(cf, ax=ax_flow, shrink=0.8, pad=0.02)
    cbar.set_label("|flow|")

    for ax in axes:
        ax.grid(True, linestyle=":", alpha=0.3)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.tight_layout()
    save_path = None
    if save_dir is not None:
        save_path = Path(save_dir) / f"frame_{frame_idx:03d}.png"
        fig.savefig(save_path, dpi=200)
    plt.close(fig)
    return save_path
```

## 5) Run the CDV loop
Iterate over frames, correlate tracer/source pairs, rotate into source-x frame, accumulate, and
optionally save per-frame plots.

```python
frame_start = 0
frame_stop = 60  # adjust to your dataset (e.g., disp_df["frame"].max())

for i in range(frame_start, frame_stop):
    disp_temp = disp_df[disp_df["frame"] == i]
    if disp_temp.empty:
        continue
    # Optional region-of-interest filter; comment out to use all pairs
    disp_active = disp_temp[
        disp_temp["x0"].between(300, 1100) & disp_temp["x1"].between(300, 1100)
    ]
    if disp_active.empty:
        continue

    batch, _ = correlation_batch(
        disp_active,
        disp_temp,
        source_frame_col="frame",
        tracer_frame_col="frame",
        source_position_cols=position_cols,
        tracer_position_cols=position_cols,
        source_motion_cols=motion_cols,
        tracer_motion_cols=motion_cols,
        pair_filter=pair_filter,
    )
    batch_r = batch.rotate_to_source_x()
    ensemble.add(batch_r)
    plot_cdv_frame(
        i,
        disp_temp,
        batch_r,
        ensemble,
        X,
        Y,
        save_dir=plot_dir,
        quiver_stride=6,
    )
```

## 6) Finalize and save the flow field

```python
import numpy as np

mean, sum_w, counts = ensemble.finalize()
centers = ensemble.centers
np.savez("results/cdv_flow_field.npz", mean=mean, sum_w=sum_w, counts=counts, centers=centers)
mean[:5]  # peek at the first few grid points
```

## Tips
- Tune `max_pair_distance` and `kernel` to match your imaging scale.
- If your time base is irregular, ensure `frame_col` and `time_col` are set correctly when building
  the `TrajectorySet`.
- Use a coarser grid (fewer points) for faster experimentation, then refine resolution for final plots.
