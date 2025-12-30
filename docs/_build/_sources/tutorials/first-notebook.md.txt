# First notebook

Spin up a notebook and run a minimal analysis on a toy dataset. The snippet below mirrors the
workflow from the quickstart, but adds an MSD calculation so you can plot it directly in Jupyter.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from trajkit import Trajectory, TrajectorySet, save_trajectory_set, load_trajectory_set
from trajkit.stats import msd_trajectory_set

# Build a short random-walk trajectory
x = np.cumsum(np.random.randn(400, 2), axis=0)
frame = np.arange(len(x))
tr = Trajectory(track_id="demo", x=x, frame=frame, frame_rate_hz=20.0)

ts = TrajectorySet(dataset_id="notebook-demo")
ts.add(tr)
save_trajectory_set(ts, "examples/datasets/notebook_demo")

# Load and analyze
ts2 = load_trajectory_set("examples/datasets/notebook_demo", frame_rate_hz=20.0)
msd_df = msd_trajectory_set(ts2)

fig, ax = plt.subplots()
ax.plot(msd_df["lag"], msd_df["msd"], marker="o")
ax.set_xlabel("lag (frames)")
ax.set_ylabel("MSD")
ax.grid(True, linestyle=":", alpha=0.3)
```

For more advanced flow-field examples, open `examples/notebooks/cdv_in_action.ipynb` and
`examples/notebooks/cdv_displacement.ipynb` in Jupyter and rerun the cells.
