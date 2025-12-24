from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

KernelCallable = Callable[[np.ndarray], np.ndarray]
DistanceCallable = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class RegularGridSpec:
    """
    Definition of a regular grid in d dimensions.

    grid_min / grid_max are inclusive ranges for grid centers.
    cell_size controls spacing between centers in each dimension.
    """

    grid_min: Sequence[float]
    grid_max: Sequence[float]
    cell_size: Sequence[float]
    stencil_radius: int = 1  # number of cells to search in each direction from the host cell

    def as_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        grid_min = np.asarray(self.grid_min, dtype=float)
        grid_max = np.asarray(self.grid_max, dtype=float)
        cell_size = np.asarray(self.cell_size, dtype=float)
        if not (grid_min.shape == grid_max.shape == cell_size.shape):
            raise ValueError("grid_min, grid_max, cell_size must have the same shape.")
        if (cell_size <= 0).any():
            raise ValueError("cell_size entries must be positive.")
        if self.stencil_radius < 0:
            raise ValueError("stencil_radius must be non-negative.")
        return grid_min, grid_max, cell_size


@dataclass(frozen=True)
class CDVConfig:
    """
    High-level configuration for the conditional ensemble engine.
    """

    grid: RegularGridSpec
    distance_fn: DistanceCallable
    kernel_fn: KernelCallable


@dataclass
class MeasurementBatch:
    """
    A batch of measurements to accumulate onto the ensemble grid.

    coords: shape (N, d)
    values: shape (N, q)
    """

    coords: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        coords = np.asarray(self.coords, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if coords.ndim != 2:
            raise ValueError("coords must be 2D array (N, d).")
        if values.ndim != 2:
            raise ValueError("values must be 2D array (N, q).")
        if coords.shape[0] != values.shape[0]:
            raise ValueError("coords and values must share the same leading dimension N.")
        self.coords = coords
        self.values = values

    @property
    def ndim(self) -> int:
        return int(self.coords.shape[1])

    @property
    def value_dim(self) -> int:
        return int(self.values.shape[1])
