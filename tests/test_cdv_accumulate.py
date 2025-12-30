import numpy as np
import pytest

from trajkit.cdv import (
    EnsembleAccumulator,
    MeasurementBatch,
    RegularGridNeighborFinder,
    RegularGridSpec,
    gaussian_kernel,
    hard_cutoff_kernel,
    euclidean_distance,
)


def _make_accumulator(kernel_fn):
    grid = RegularGridSpec(
        grid_min=[-1.0, -1.0],
        grid_max=[1.0, 1.0],
        cell_size=[1.0, 1.0],
        stencil_radius=0,
    )
    neighbor = RegularGridNeighborFinder(grid)
    acc = EnsembleAccumulator(
        neighbor_finder=neighbor,
        value_dim=2,
        distance_fn=euclidean_distance,
        kernel_fn=kernel_fn,
        track_counts=True,
    )
    return acc, neighbor


def _center_index(neighbor: RegularGridNeighborFinder, center_xy) -> int:
    matches = np.where((neighbor.centers == np.asarray(center_xy)).all(axis=1))[0]
    assert matches.size == 1
    return int(matches[0])


def test_hard_kernel_single_point_hits_center_cell():
    acc, neighbor = _make_accumulator(hard_cutoff_kernel(0.6))
    batch = MeasurementBatch(coords=np.array([[0.0, 0.0]]), values=np.array([[2.0, 4.0]]))
    acc.update(batch)
    mean, w, counts = acc.finalize()

    idx_center = _center_index(neighbor, [0.0, 0.0])
    assert w[idx_center] == pytest.approx(1.0)
    assert counts[idx_center] == 1
    assert mean[idx_center, 0] == pytest.approx(2.0)
    assert mean[idx_center, 1] == pytest.approx(4.0)


def test_gaussian_kernel_weighted_mean_with_two_samples():
    acc, neighbor = _make_accumulator(gaussian_kernel(1.0))
    coords = np.array([[0.0, 0.0], [0.0, 0.0]])
    values = np.array([[1.0, 0.0], [3.0, 0.0]])
    acc.update(MeasurementBatch(coords=coords, values=values))
    mean, w, counts = acc.finalize()

    idx_center = _center_index(neighbor, [0.0, 0.0])
    assert counts[idx_center] == 2
    assert w[idx_center] == pytest.approx(2.0)  # two weights of 1.0 at the center
    assert mean[idx_center, 0] == pytest.approx(2.0)
    assert mean[idx_center, 1] == pytest.approx(0.0)


def test_points_outside_grid_are_ignored():
    acc, _ = _make_accumulator(hard_cutoff_kernel(0.5))
    batch = MeasurementBatch(coords=np.array([[5.0, 5.0]]), values=np.array([[10.0, 10.0]]))
    acc.update(batch)
    mean, w, counts = acc.finalize()

    assert np.all(w == 0)
    assert np.all(np.isnan(mean))
    assert counts is not None
    assert np.all(counts == 0)
