from __future__ import annotations

import numpy as np

from .distance import euclidean_distance
from .kernel import gaussian_kernel, hard_cutoff_kernel
from .neighbors import RegularGridNeighborFinder
from .types import MeasurementBatch


class EnsembleAccumulator:
    """
    Weighted ensemble accumulator on a fixed grid.

    This NumPy implementation is structured so that drop-in backends (Numba/GPU)
    can later replace the update logic without changing the public API.
    """

    def __init__(
        self,
        neighbor_finder: RegularGridNeighborFinder,
        value_dim: int,
        *,
        distance_fn=euclidean_distance,
        kernel_fn=hard_cutoff_kernel(1.0),
        track_counts: bool = True,
    ):
        self.neighbor_finder = neighbor_finder
        self.distance_fn = distance_fn
        self.kernel_fn = kernel_fn
        self.value_dim = int(value_dim)
        if self.value_dim < 1:
            raise ValueError("value_dim must be >= 1.")

        M = neighbor_finder.n_points
        self.sum_v = np.zeros((M, self.value_dim), dtype=float)
        self.sum_w = np.zeros(M, dtype=float)
        self.counts = np.zeros(M, dtype=int) if track_counts else None

    def update(self, batch: MeasurementBatch) -> None:
        coords = batch.coords
        values = batch.values
        if coords.shape[1] != self.neighbor_finder.centers.shape[1]:
            raise ValueError("Batch coords dimensionality does not match grid dimensionality.")
        if values.shape[1] != self.value_dim:
            raise ValueError("Batch values dimensionality does not match accumulator value_dim.")

        for p, v in zip(coords, values):
            idxs, centers_subset = self.neighbor_finder.neighbors_for_point(p)
            if idxs.size == 0:
                continue
            dists = self.distance_fn(p, centers_subset)
            weights = self.kernel_fn(dists)
            if weights.shape != dists.shape:
                raise ValueError("kernel_fn must return weights with same shape as distances.")

            # Ignore zero-weight entries to avoid unnecessary scatter
            mask = weights > 0
            if not np.any(mask):
                continue

            idxs_masked = idxs[mask]
            w_masked = weights[mask]

            np.add.at(self.sum_w, idxs_masked, w_masked)
            for j in range(self.value_dim):
                np.add.at(self.sum_v[:, j], idxs_masked, w_masked * v[j])
            if self.counts is not None:
                np.add.at(self.counts, idxs_masked, 1)

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Returns:
            mean: (M, q) weighted mean values (NaN where sum_w==0)
            sum_w: (M,) total weights
            counts: (M,) sample counts per grid point (or None)
        """
        mean = np.full_like(self.sum_v, np.nan)
        mask = self.sum_w > 0
        mean[mask] = self.sum_v[mask] / self.sum_w[mask, None]
        return mean, self.sum_w, self.counts

    def merge(self, other: "EnsembleAccumulator") -> None:
        """
        Merge another accumulator into this one (in-place).
        Shapes and configuration must match.
        """
        if self.sum_v.shape != other.sum_v.shape or self.value_dim != other.value_dim:
            raise ValueError("Accumulator shapes/value_dim do not match.")
        self.sum_v += other.sum_v
        self.sum_w += other.sum_w
        if self.counts is not None and other.counts is not None:
            self.counts += other.counts
