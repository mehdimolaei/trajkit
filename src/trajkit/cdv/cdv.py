from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from .accumulate import EnsembleAccumulator
from .neighbors import RegularGridNeighborFinder
from .types import CDVConfig, MeasurementBatch


def run_cdv(
    batches: Iterable[MeasurementBatch],
    config: CDVConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Convenience runner for the regular-grid NumPy path.

    Returns:
        mean: (M, q)
        weights: (M,)
        counts: (M,) or None
        centers: (M, d)
    """
    batches_list = list(batches)
    if len(batches_list) == 0:
        raise ValueError("No batches provided to run_cdv.")

    neighbor_finder = RegularGridNeighborFinder(config.grid)
    accumulator = EnsembleAccumulator(
        neighbor_finder,
        value_dim=batches_list[0].value_dim,
        distance_fn=config.distance_fn,
        kernel_fn=config.kernel_fn,
    )

    for batch in batches_list:
        accumulator.update(batch)

    mean, w, counts = accumulator.finalize()
    return mean, w, counts, neighbor_finder.centers
