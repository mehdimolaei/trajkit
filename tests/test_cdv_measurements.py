import numpy as np

from trajkit.cdv import (
    MeasurementBatch,
    displacement_batch_from_trajectory,
    displacement_batches_from_trajectory_set,
    displacement_at_time_from_trajectory_set,
    measurement_bounds,
)
from trajkit.traj import Trajectory, TrajectorySet


def _make_traj(track_id: str, t_values, x_values):
    return Trajectory(track_id=track_id, x=np.array(x_values, dtype=float), t=np.array(t_values))


def test_measurement_bounds_reports_min_max_and_count():
    batch = MeasurementBatch(coords=np.array([[0, 1], [2, -1]], dtype=float), values=np.zeros((2, 2)))
    mn, mx, n = measurement_bounds(batch)
    np.testing.assert_allclose(mn, [0, -1])
    np.testing.assert_allclose(mx, [2, 1])
    assert n == 2


def test_displacement_batch_from_single_traj():
    tr = _make_traj("a", [0.0, 0.5, 1.0], [[0, 0], [1, 0], [2, 0]])
    batch = displacement_batch_from_trajectory(tr, delta_t=0.5, tol=1e-9)
    np.testing.assert_allclose(batch.coords, [[0, 0], [1, 0]])
    np.testing.assert_allclose(batch.values, [[1, 0], [1, 0]])


def test_displacement_batches_from_trajectory_set():
    ts = TrajectorySet(dataset_id="demo")
    ts.add(_make_traj("a", [0, 1, 2], [[0, 0], [1, 0], [2, 0]]))
    ts.add(_make_traj("b", [0, 1], [[0, 0], [0, 2]]))
    batch = displacement_batches_from_trajectory_set(ts, delta_t=1.0)
    # Expect three displacements total
    assert batch.coords.shape == (3, 2)
    assert batch.values.shape == (3, 2)


def test_displacement_at_specific_time_from_trajectory_set():
    ts = TrajectorySet(dataset_id="demo")
    ts.add(_make_traj("a", [0, 1, 2], [[0, 0], [1, 0], [2, 0]]))
    ts.add(_make_traj("b", [0, 1], [[0, 0], [0, 2]]))
    batch = displacement_at_time_from_trajectory_set(ts, t0=0.0, delta_t=1.0)
    # both trajectories have samples at t=0 and t=1
    assert batch.coords.shape == (2, 2)
    assert batch.values.shape == (2, 2)
