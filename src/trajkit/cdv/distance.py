from __future__ import annotations

import numpy as np


def euclidean_distance(p: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """
    Compute Euclidean distances from a single point p (shape (d,)) to
    a set of centers (shape (M, d)).
    """
    if p.ndim != 1:
        raise ValueError("p must be 1D array (d,).")
    if centers.ndim != 2:
        raise ValueError("centers must be 2D array (M, d).")
    diff = centers - p[None, :]
    return np.sqrt(np.sum(diff * diff, axis=1))
