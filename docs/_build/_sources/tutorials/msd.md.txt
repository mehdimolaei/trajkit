# MSD tutorial

End-to-end MSD workflow that matches the MATLAB scripts in `matlabScripts/MSD`: load or build a
`TrajectorySet`, compute MSD for single tracks and pair-weighted aggregates, and plot.

## 1) Imports
```python
import numpy as np
import matplotlib.pyplot as plt
from trajkit import Trajectory, TrajectorySet, save_trajectory_set, load_trajectory_set
from trajkit.stats import msd, msd_trajectory_set
```

## 2) Load or build trajectories
**Load from local directory (swap this for a database download later):**
```python
trajset = load_trajectory_set("results/msd_demo_dataset", frame_rate_hz=20.0)
```

**Or build a small synthetic set (2D random walks):**
```python
np.random.seed(0)
N = 5   # number of tracks
T = 200 # length per track

trajset = TrajectorySet(dataset_id="msd_demo")
for i in range(N):
    steps = np.random.randn(T, 2)
    x = np.cumsum(steps, axis=0)
    frame = np.arange(T, dtype=int)
    tr = Trajectory(track_id=f"p{i}", x=x, frame=frame, frame_rate_hz=20.0)
    trajset.add(tr)

# Optional: persist to Parquet for reuse
save_trajectory_set(trajset, "results/msd_demo_dataset")
```

## 3) Single-trajectory MSD
```python
tr0 = trajset.get("p0")
msd_single = msd(tr0, max_lag=40, per_dim=True, return_sd=True)
msd_single.head()
```

## 4) Trajectory-set MSD (pair-weighted)
Pair-weighted aggregation matches the MATLAB behavior:
```python
msd_pair = msd_trajectory_set(
    trajset,
    max_lag=40,
    aggregate=True,
    aggregate_mode="pair",
    per_dim=True,
)
msd_pair.head()
```

## 5) Plot
```python
fig, ax = plt.subplots()
ax.plot(msd_pair["lag"], msd_pair["msd"], marker="o", label="pair-weighted msd")
ax.set_xlabel("lag (frames)")
ax.set_ylabel("MSD")
ax.grid(True, linestyle=":", alpha=0.3)
ax.legend()
plt.show()
```

## Options and tips
- `lags` / `max_lag`: choose lags in frames.
- `per_dim=True`: include per-dimension MSD columns (`msd_dim0`, …).
- `return_sd=True`: include mean displacement components (`sd_dim0`, …).
- `interpolate=True`: interpolate to contiguous frames before computing displacements
  (requires `Trajectory.frame`), helpful if frames have gaps.
- `aggregate_mode="track"` returns a per-lag mean over tracks (legacy behavior). `pair` is
  pair-weighted. The returned `n_pairs` gives the total pairs contributing to each lag.
