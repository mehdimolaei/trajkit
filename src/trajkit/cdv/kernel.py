from __future__ import annotations

import numpy as np


def hard_cutoff_kernel(cutoff: float):
    """
    Binary kernel: weight = 1 for d <= cutoff, else 0.
    Returns a callable that maps distances -> weights.
    """

    if cutoff <= 0:
        raise ValueError("cutoff must be positive for hard kernel.")

    def _fn(d: np.ndarray) -> np.ndarray:
        return (d <= cutoff).astype(float)

    return _fn


def gaussian_kernel(sigma: float):
    """
    Gaussian kernel: weight = exp(-0.5 * (d / sigma)^2).
    Returns a callable that maps distances -> weights.
    """

    if sigma <= 0:
        raise ValueError("sigma must be positive for gaussian kernel.")

    inv_sigma2 = 1.0 / (sigma * sigma)

    def _fn(d: np.ndarray) -> np.ndarray:
        return np.exp(-0.5 * (d * d) * inv_sigma2)

    return _fn
