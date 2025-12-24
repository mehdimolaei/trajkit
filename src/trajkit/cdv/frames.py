from __future__ import annotations

import numpy as np


def source_frame_from_displacement(dr: np.ndarray) -> np.ndarray:
    """
    Build a 2D rotation matrix that aligns the source displacement `dr`
    with the +x axis. Returns a 2x2 matrix R such that x' = R @ x.
    """
    dr = np.asarray(dr, dtype=float).reshape(-1)
    if dr.shape[0] != 2:
        raise ValueError("source_frame_from_displacement expects a 2D displacement.")
    norm = np.linalg.norm(dr)
    if norm == 0:
        # degenerate; fall back to identity
        return np.eye(2)
    c = dr[0] / norm
    s = dr[1] / norm
    return np.array([[c, s], [-s, c]])


def rotate_into_frame(R: np.ndarray, vecs: np.ndarray) -> np.ndarray:
    """
    Rotate vectors (N,2) into the source-aligned frame using rotation matrix R (2,2).
    """
    if R.shape != (2, 2):
        raise ValueError("R must be 2x2.")
    vecs = np.asarray(vecs, dtype=float)
    if vecs.shape[-1] != 2:
        raise ValueError("vecs must have trailing dimension 2.")
    return vecs @ R.T
