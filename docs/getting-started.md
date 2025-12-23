
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
