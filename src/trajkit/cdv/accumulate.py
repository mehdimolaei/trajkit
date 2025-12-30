from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np

from .distance import euclidean_distance
from .kernel import gaussian_kernel, hard_cutoff_kernel
from .neighbors import RegularGridNeighborFinder
from .types import MeasurementBatch, RegularGridSpec

# Type aliases for correlation ensemble accumulation.
WeightFn = Callable[[np.ndarray, np.ndarray, np.ndarray, Any], float]
ValueFn = Callable[[np.ndarray, np.ndarray, np.ndarray, Any], np.ndarray]


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


def _normalize_kernel(kernel_spec: Any) -> Callable[[np.ndarray], np.ndarray]:
    """
    Accepts a kernel spec (callable, numeric radius, or mapping) and returns a kernel function.
    """
    if callable(kernel_spec):
        return kernel_spec
    if isinstance(kernel_spec, Mapping):
        ktype = kernel_spec.get("type")
        if ktype == "hard":
            radius = float(kernel_spec["radius"])
            if radius <= 0:
                raise ValueError("Hard cutoff radius must be positive.")
            return hard_cutoff_kernel(radius)
        if ktype == "gaussian":
            sigma = float(kernel_spec["sigma"])
            if sigma <= 0:
                raise ValueError("Gaussian sigma must be positive.")
            return gaussian_kernel(sigma)
        raise ValueError("Unknown kernel spec; supported types: 'hard', 'gaussian'.")
    try:
        radius = float(kernel_spec)
    except Exception as exc:
        raise TypeError("kernel must be a callable, mapping, or numeric radius.") from exc
    if radius <= 0:
        raise ValueError("Cutoff radius must be positive.")
    return hard_cutoff_kernel(radius)


class CorrelationEnsembleAccumulator:
    """
    Accumulates correlation batches onto a grid in relative-position space with configurable binning.

    Binning is controlled via `kernel`: pass a numeric radius for hard cutoff, a mapping like
    {"type": "gaussian", "sigma": ...}, or any callable kernel(distance_array)->weights.
    Values to accumulate come from `value_fn`; weights per-row can be provided via `weight_fn`.
    """

    def __init__(
        self,
        grid: RegularGridSpec | np.ndarray,
        *,
        kernel: Any = 1.0,
        distance_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = euclidean_distance,
        value_fn: ValueFn | None = None,
        weight_fn: WeightFn | None = None,
        track_counts: bool = True,
        value_dim: int | None = None,
    ):
        if isinstance(grid, RegularGridSpec):
            self._grid = RegularGridNeighborFinder(grid)
            centers = self._grid.centers
        else:
            centers = np.asarray(grid, dtype=float)
            if centers.ndim != 2:
                raise ValueError("grid must be a 2D array of grid point centers (M, d).")
            self._grid = None
        if centers.shape[0] == 0:
            raise ValueError("grid must contain at least one center.")

        self.centers = centers
        self.dim = int(centers.shape[1])
        self.kernel_fn = _normalize_kernel(kernel)
        self.distance_fn = distance_fn
        self.value_fn = value_fn or (lambda rel, tracer, source, meta_row: tracer)
        self.weight_fn = weight_fn or (lambda rel, tracer, source, meta_row: 1.0)
        self.track_counts = bool(track_counts)
        self.value_dim: int | None = int(value_dim) if value_dim is not None else None

        M = centers.shape[0]
        self.sum_v: np.ndarray | None = (
            np.zeros((M, self.value_dim), dtype=float) if self.value_dim is not None else None
        )
        self.sum_w: np.ndarray | None = np.zeros(M, dtype=float) if self.value_dim is not None else None
        self.counts: np.ndarray | None = (
            np.zeros(M, dtype=int) if (self.value_dim is not None and self.track_counts) else None
        )
        self.total_pairs = 0

    def _ensure_arrays(self, value_vec: np.ndarray) -> None:
        if value_vec.ndim != 1:
            raise ValueError("value_fn must return a 1D array.")
        if self.value_dim is None:
            self.value_dim = int(value_vec.shape[0])
            M = self.centers.shape[0]
            self.sum_v = np.zeros((M, self.value_dim), dtype=float)
            self.sum_w = np.zeros(M, dtype=float)
            self.counts = np.zeros(M, dtype=int) if self.track_counts else None
        elif value_vec.shape[0] != self.value_dim:
            raise ValueError("value_fn output dimension does not match existing accumulator.")

    def add(self, batch: "CorrelationBatch") -> None:
        if batch.position_dim != self.dim:
            raise ValueError("Batch relative_positions dimensionality does not match grid dimension.")
        if self.value_dim is None and batch.n_pairs == 0:
            return

        for i in range(batch.n_pairs):
            rel = batch.relative_positions[i]
            tracer = batch.tracer_motion[i]
            source = batch.source_motion[i]
            meta_row = batch.meta.iloc[i]

            value_vec = np.asarray(self.value_fn(rel, tracer, source, meta_row), dtype=float)
            self._ensure_arrays(value_vec)
            if self.sum_v is None or self.sum_w is None:
                continue

            weights = np.asarray(self.kernel_fn(self.distance_fn(rel, self.centers)), dtype=float)
            if weights.shape != (self.centers.shape[0],):
                raise ValueError("kernel must return weights with shape (n_grid_points,).")

            row_weight = float(self.weight_fn(rel, tracer, source, meta_row))
            if row_weight == 0.0:
                self.total_pairs += 1
                continue

            weights = weights * row_weight
            mask = weights > 0
            if not np.any(mask):
                self.total_pairs += 1
                continue

            idxs = np.nonzero(mask)[0]
            w_masked = weights[idxs]

            np.add.at(self.sum_w, idxs, w_masked)
            for j in range(self.value_dim):
                np.add.at(self.sum_v[:, j], idxs, w_masked * value_vec[j])
            if self.counts is not None:
                np.add.at(self.counts, idxs, 1)

            self.total_pairs += 1

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if self.sum_v is None or self.sum_w is None or self.value_dim is None:
            raise ValueError("No data accumulated; call add() first.")
        mean = np.full_like(self.sum_v, np.nan)
        mask = self.sum_w > 0
        mean[mask] = self.sum_v[mask] / self.sum_w[mask, None]
        return mean, self.sum_w, self.counts

    def merge(self, other: "CorrelationEnsembleAccumulator") -> None:
        if self.centers.shape != other.centers.shape or self.dim != other.dim:
            raise ValueError("Grid centers do not match; cannot merge.")
        if self.value_dim != other.value_dim:
            raise ValueError("value_dim does not match; cannot merge.")
        if self.sum_v is None or self.sum_w is None or other.sum_v is None or other.sum_w is None:
            raise ValueError("Both accumulators must be initialized before merge.")

        self.sum_v += other.sum_v
        self.sum_w += other.sum_w
        if self.counts is not None and other.counts is not None:
            self.counts += other.counts
        self.total_pairs += other.total_pairs
