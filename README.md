<p align="center">
  <img src="docs/assets/logo.png" alt="trajkit logo" width="260">
</p>

# trajkit

Python toolkit for reproducible trajectory (time-series) analytics and flow-field
inference for Brownian and non-Brownian diffusion processes.
See [ACKNOWLEDGMENTS](ACKNOWLEDGMENTS.md).

## Features
- Lightweight `Trajectory` and `TrajectorySet` data model (frame- or time-based)
- Fast Parquet I/O helpers for saving/loading datasets
- MSD/statistics helpers for single trajectories and trajectory sets
- Conditional displacement/velocity (CDV) utilities to infer flow fields
- Ready-to-adapt notebooks and plotting helpers for publication-quality figures

## Install
Requires Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install trajkit
```

For development:

```bash
pip install -e ".[dev]"
pytest -q
```

## Quickstart
Create and persist a trajectory set:

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
)

ts = TrajectorySet(dataset_id="demo")
ts.add(tr)
ts.summary_table().head()

save_trajectory_set(ts, "examples/datasets/demo_brownian_2d")
ts2 = load_trajectory_set("examples/datasets/demo_brownian_2d", frame_rate_hz=20.0)
```

Compute MSD for a trajectory set:

```python
from trajkit.stats import msd_trajectory_set
msd_df = msd_trajectory_set(ts2)
```

Explore more examples in `examples/notebooks`.

## Documentation
See `docs/` for the rendered site sources (MkDocs). Key entry points:
- `docs/index.md` for an overview
- `docs/getting-started.md` for the minimal workflow
- `docs/tutorials` and `examples/notebooks` for end-to-end notebooks

## Contributing
Issues and PRs are welcome. Please see [CONTRIBUTING](CONTRIBUTING.md) for guidelines.

## License
[MIT](LICENSE)
