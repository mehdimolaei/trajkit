from __future__ import annotations

import itertools
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .types import RegularGridSpec


class RegularGridNeighborFinder:
    """
    Stencil-based neighbor finder for a regular grid.

    Each point is mapped to its host cell; neighbors are all cells within
    `stencil_radius` in index space. This keeps neighbor counts small and avoids
    all-to-all distance checks.
    """

    def __init__(self, grid: RegularGridSpec):
        grid_min, grid_max, cell_size = grid.as_arrays()
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.cell_size = cell_size
        self.stencil_radius = int(grid.stencil_radius)

        centers_1d: List[np.ndarray] = []
        for gmin, gmax, dx in zip(grid_min, grid_max, cell_size):
            # inclusive range of centers
            centers_axis = np.arange(gmin, gmax + 0.5 * dx, dx, dtype=float)
            if centers_axis.size == 0:
                raise ValueError("Grid axis has zero length; check grid_min/grid_max/cell_size.")
            centers_1d.append(centers_axis)

        self.grid_shape: Tuple[int, ...] = tuple(len(c) for c in centers_1d)
        mesh = np.meshgrid(*centers_1d, indexing="ij")
        stacked = np.stack(mesh, axis=-1)
        self.centers = stacked.reshape(-1, stacked.shape[-1])  # (M, d)

    def _host_index(self, p: np.ndarray) -> Optional[np.ndarray]:
        rel = (p - self.grid_min) / self.cell_size
        idx = np.floor(rel).astype(int)
        if np.any(idx < 0) or np.any(idx >= self.grid_shape):
            return None
        return idx

    def neighbors_for_point(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return neighbor indices and centers for a single point.

        Returns:
            idx_flat: (K,) flattened indices into centers array.
            centers_subset: (K, d) coordinates of those centers.
            If the point lies outside the grid, both are empty arrays.
        """

        if p.ndim != 1:
            raise ValueError("p must be 1D array (d,).")

        host = self._host_index(p)
        if host is None:
            return np.array([], dtype=int), np.empty((0, p.shape[0]), dtype=float)

        ranges = [
            range(max(0, h - self.stencil_radius), min(self.grid_shape[i], h + self.stencil_radius + 1))
            for i, h in enumerate(host)
        ]

        multi_indices = list(itertools.product(*ranges))
        idx_flat = np.ravel_multi_index(np.array(multi_indices).T, self.grid_shape)
        centers_subset = self.centers[idx_flat]
        return idx_flat, centers_subset

    @property
    def n_points(self) -> int:
        return int(self.centers.shape[0])
