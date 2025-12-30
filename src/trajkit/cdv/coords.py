from __future__ import annotations

import numpy as np

from .frames import rotate_into_frame


def coords_xyprime(R: np.ndarray, probe_displacements: np.ndarray) -> np.ndarray:
    """
    Rotate probe displacements (N,2) into the source-aligned frame defined by R.
    """
    return rotate_into_frame(R, probe_displacements)


def coords_polar(R: np.ndarray, probe_displacements: np.ndarray) -> np.ndarray:
    """
    Rotate probes into the source frame, then convert to polar (r, theta).
    theta uses atan2(y', x').
    """
    xy = coords_xyprime(R, probe_displacements)
    x = xy[:, 0]
    y = xy[:, 1]
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)
    return np.stack([r, theta], axis=1)
