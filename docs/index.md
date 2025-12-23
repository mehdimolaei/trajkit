<p align="center">
  <img src="assets/logo.png" alt="trajkit logo" width="260">
</p>

# trajkit

**trajkit** is a Python toolkit for reproducible trajectory analytics and flow-field inference
for Brownian and active colloids.

## What you can do
- Load trajectory datasets (frame-based or non-uniform time sampling)
- Clean and manipulate tracks
- Compute MSD / statistics (coming next)
- Estimate flow fields around particles (coming next)
- Build publishable visualizations + notebooks

## Quickstart
```bash
pip install trajkit


### `docs/getting-started.md`
```bash
cat > docs/getting-started.md <<'EOF'
# Getting Started

This guide shows the basic workflow:
1) create a `Trajectory`
2) add it to a `TrajectorySet`
3) save/load using Parquet

## Create a trajectory

```python
import numpy as np
from trajkit import Trajectory, TrajectorySet

x = np.cumsum(np.random.randn(200, 2), axis=0)  # (T, D)
frame = np.arange(200)

tr = Trajectory(
    track_id="p1",
    x=x,
    frame=frame,
    frame_rate_hz=20.0,
    label="passive",
    track_features={"diameter_um": 1.0},
)

ts = TrajectorySet(dataset_id="demo")
ts.add(tr)

ts.summary_table().head()

from trajkit import save_trajectory_set, load_trajectory_set

save_trajectory_set(ts, "examples/datasets/demo_brownian_2d")
ts2 = load_trajectory_set("examples/datasets/demo_brownian_2d", frame_rate_hz=20.0)


### `docs/concepts/trajectories.md`
```bash
cat > docs/concepts/trajectories.md <<'EOF'
# Trajectories

A `Trajectory` stores:
- `x`: positions shaped `(T, D)` (general ND)
- either:
  - `t`: time in seconds (supports non-uniform sampling), or
  - `frame` + `frame_rate_hz` (uniform sampling)

A `TrajectorySet` is a collection of trajectories from a single dataset/experiment,
including shared metadata (units, calibration, conditions).
