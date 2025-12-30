import numpy as np

from trajkit.traj import Trajectory, TrajectorySet
from trajkit.cdv import run_cdv


def test_run_cdv_builds_flow_field_from_trajectory_set():
    ts = TrajectorySet(dataset_id="demo")
    frame = np.array([0, 1], dtype=int)

    ts.add(
        Trajectory(
            track_id="a",
            x=np.array([[0.0, 0.0], [1.0, 0.0]]),
            frame=frame,
            frame_rate_hz=1.0,
        )
    )
    ts.add(
        Trajectory(
            track_id="b",
            x=np.array([[0.0, 1.0], [1.0, 1.0]]),
            frame=frame,
            frame_rate_hz=1.0,
        )
    )

    grid_centers = np.array([[0.0, 0.0]])

    mean, sum_w, counts, centers = run_cdv(
        ts,
        delta_t=1.0,
        grid_centers=grid_centers,
        time_tol=1e-9,
        kernel=10.0,
    )

    np.testing.assert_allclose(centers, grid_centers)
    assert mean.shape == (1, 2)
    assert sum_w.shape == (1,)
    assert counts is not None
    np.testing.assert_allclose(mean[0], np.array([1.0, 0.0]))
    np.testing.assert_allclose(sum_w[0], 4.0)
    np.testing.assert_array_equal(counts[0], 4)
