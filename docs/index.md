<p align="center">
  <img src="assets/logo.png" alt="trajkit logo" width="260">
</p>

# trajkit


**trajkit** is a Python toolkit for reproducible trajectory analytics and flow-field inference
for Brownian and active colloids.


## What you can do
- Load trajectory datasets (frame-based or non-uniform time sampling)
- Clean and manipulate tracks
- Compute MSD and displacement statistics
- Estimate flow fields around particles via CDV / correlation utilities
- Build publishable visualizations + notebooks

## Quickstart
Install:

```bash
pip install trajkit
```

Create and save a trajectory set:

```python
import numpy as np
from trajkit import (
    Trajectory,
    TrajectorySet,
    save_trajectory_set,
    load_trajectory_set,
)

x = np.cumsum(np.random.randn(200, 2), axis=0)
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
save_trajectory_set(ts, "examples/datasets/demo_brownian_2d")
ts2 = load_trajectory_set("examples/datasets/demo_brownian_2d", frame_rate_hz=20.0)
```
## Tutorials

.. toctree::
   :maxdepth: 1

   tutorials/first-notebook
   tutorials/cdv-flow-field
   tutorials/msd