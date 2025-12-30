# Getting Started

This quick walkthrough shows how to create a trajectory, bundle it into a `TrajectorySet`, and
save/load it from disk.

## Install
```bash
pip install trajkit
```

## Create a trajectory
```python
import numpy as np
from trajkit import Trajectory

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
```

## Build a dataset and save
```python
from trajkit import TrajectorySet, save_trajectory_set, load_trajectory_set

ts = TrajectorySet(dataset_id="demo")
ts.add(tr)

# Inspect summary table
ts.summary_table()

# Persist to disk and load it back
save_trajectory_set(ts, "examples/datasets/demo_brownian_2d")
ts2 = load_trajectory_set("examples/datasets/demo_brownian_2d", frame_rate_hz=20.0)

# Compute mean-squared displacement per lag (optional)
from trajkit.stats import msd_trajectory_set
msd_df = msd_trajectory_set(ts2)
msd_pair = msd_trajectory_set(ts2, max_lag=20, aggregate=True, aggregate_mode="pair")
msd_df.head()
```

That's it—you're ready to explore your own trajectories.
