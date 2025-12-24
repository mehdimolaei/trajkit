"""
Conditional ensemble averaging (CDV-style) core utilities.

This initial milestone implements a NumPy-based regular-grid accumulator with
pluggable kernels and distance functions. The API is modular so future backends
(Numba/GPU) and neighbor-finder strategies can be dropped in without changing
call sites.
"""

from .types import CDVConfig, MeasurementBatch, RegularGridSpec
from .kernel import gaussian_kernel, hard_cutoff_kernel
from .distance import euclidean_distance
from .neighbors import RegularGridNeighborFinder
from .accumulate import EnsembleAccumulator
from .cdv import run_cdv

__all__ = [
    "CDVConfig",
    "MeasurementBatch",
    "RegularGridSpec",
    "gaussian_kernel",
    "hard_cutoff_kernel",
    "euclidean_distance",
    "RegularGridNeighborFinder",
    "EnsembleAccumulator",
    "run_cdv",
]
