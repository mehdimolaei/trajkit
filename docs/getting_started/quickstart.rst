Quickstart
==========

Minimal workflow to build, save, and reload a trajectory set.

.. code-block:: python

    import numpy as np
    from trajkit import (
        Trajectory,
        TrajectorySet,
        save_trajectory_set,
        load_trajectory_set,
    )

    # Build a simple 2D walk
    x = np.cumsum(np.random.randn(200, 2), axis=0)
    frame = np.arange(len(x))
    tr = Trajectory(track_id="p1", x=x, frame=frame, frame_rate_hz=20.0, label="demo")

    ts = TrajectorySet(dataset_id="demo")
    ts.add(tr)

    ts.summary_table().head()
    save_trajectory_set(ts, "examples/datasets/demo_brownian_2d")
    ts2 = load_trajectory_set("examples/datasets/demo_brownian_2d", frame_rate_hz=20.0)

Next steps:

- Compute MSD: ``from trajkit.stats import msd_trajectory_set``
- Run CDV: see ``tutorials/cdv-flow-field``
