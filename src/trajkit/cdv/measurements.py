from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd

from trajkit.traj import Trajectory, TrajectorySet

from .types import MeasurementBatch
from .frames import source_frame_from_displacement, rotate_into_frame


def measurement_bounds(batch: MeasurementBatch) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Return (min, max, count) for the measurement coordinates.
    """
    if batch.coords.size == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            0,
        )
    return batch.coords.min(axis=0), batch.coords.max(axis=0), batch.coords.shape[0]


def displacement_batch_from_trajectory(
    tr: Trajectory, delta_t: float, *, tol: float = 1e-9
) -> MeasurementBatch:
    """
    Build a MeasurementBatch of displacements over a time offset delta_t for a single trajectory.

    coords: starting positions x(t)
    values: displacement vectors x(t+delta_t) - x(t)
    """
    if delta_t <= 0:
        raise ValueError("delta_t must be positive.")

    t = tr.time_seconds()
    x = tr.x
    targets = t + delta_t
    idx = np.searchsorted(t, targets, side="left")
    valid = idx < len(t)
    if np.any(valid):
        valid_indices = np.nonzero(valid)[0]
        matched = np.abs(t[idx[valid]] - targets[valid]) <= tol
        # tighten the mask
        tmp = np.zeros_like(valid, dtype=bool)
        tmp[valid_indices] = matched
        valid = tmp
    if not np.any(valid):
        return MeasurementBatch(coords=np.empty((0, x.shape[1])), values=np.empty((0, x.shape[1])))

    src_idx = np.nonzero(valid)[0]
    dst_idx = idx[valid]

    coords = x[src_idx]
    values = x[dst_idx] - x[src_idx]
    return MeasurementBatch(coords=coords, values=values)


def _find_index_at_time(t: np.ndarray, target: float, tol: float) -> Optional[int]:
    """Return index whose time is closest to `target` within `tol`, else None."""
    idx = np.searchsorted(t, target)
    candidates = []
    if idx < len(t):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    if not candidates:
        return None
    diffs = [abs(t[i] - target) for i in candidates]
    best = min(range(len(candidates)), key=lambda k: diffs[k])
    if diffs[best] <= tol:
        return int(candidates[best])
    return None


def displacement_at_time_from_trajectory(
    tr: Trajectory, t0: float, delta_t: float, *, tol: float = 1e-9
) -> MeasurementBatch:
    """
    Displacement for a single trajectory at a specific start time t0 and offset delta_t.

    Returns coords (1,d) for the position at t0 and values (1,d) for x(t0+delta_t) - x(t0),
    or empty MeasurementBatch if either sample is missing.
    """
    if delta_t <= 0:
        raise ValueError("delta_t must be positive.")

    t = tr.time_seconds()
    x = tr.x
    i0 = _find_index_at_time(t, t0, tol)
    i1 = _find_index_at_time(t, t0 + delta_t, tol)
    if i0 is None or i1 is None:
        return MeasurementBatch(coords=np.empty((0, x.shape[1])), values=np.empty((0, x.shape[1])))

    coord = x[i0][None, :]
    value = (x[i1] - x[i0])[None, :]
    return MeasurementBatch(coords=coord, values=value)


def displacement_batches_from_trajectory_set(
    ts: TrajectorySet, delta_t: float, *, tol: float = 1e-9
) -> MeasurementBatch:
    """
    Aggregate displacements over all trajectories for a given delta_t.
    """
    coords_list = []
    values_list = []
    dim: int | None = None
    for tr in ts.trajectories.values():
        if dim is None:
            dim = tr.x.shape[1]
        batch = displacement_batch_from_trajectory(tr, delta_t, tol=tol)
        if batch.coords.size == 0:
            continue
        coords_list.append(batch.coords)
        values_list.append(batch.values)

    if not coords_list:
        d = dim if dim is not None else 0
        return MeasurementBatch(coords=np.empty((0, d)), values=np.empty((0, d)))

    coords = np.concatenate(coords_list, axis=0)
    values = np.concatenate(values_list, axis=0)
    return MeasurementBatch(coords=coords, values=values)


def displacement_at_time_from_trajectory_set(
    ts: TrajectorySet, t0: float, delta_t: float, *, tol: float = 1e-9
) -> MeasurementBatch:
    """
    Collect displacements x(t0+delta_t) - x(t0) for all trajectories that have
    both timestamps present (within tolerance).
    """
    coords_list = []
    values_list = []
    dim: int | None = None
    for tr in ts.trajectories.values():
        if dim is None:
            dim = tr.x.shape[1]
        batch = displacement_at_time_from_trajectory(tr, t0, delta_t, tol=tol)
        if batch.coords.size == 0:
            continue
        coords_list.append(batch.coords)
        values_list.append(batch.values)

    if not coords_list:
        d = dim if dim is not None else 0
        return MeasurementBatch(coords=np.empty((0, d)), values=np.empty((0, d)))

    coords = np.concatenate(coords_list, axis=0)
    values = np.concatenate(values_list, axis=0)
    return MeasurementBatch(coords=coords, values=values)


def connection_batch_from_dataframes(
    source_positions: pd.DataFrame,
    source_displacements: pd.DataFrame,
    tracer_positions: pd.DataFrame,
    tracer_displacements: pd.DataFrame,
    *,
    frame_col: str = "frame",
    position_cols: Sequence[str] = ("x", "y"),
    source_disp_cols: Sequence[str] = ("dx", "dy"),
    tracer_disp_cols: Sequence[str] = ("dx", "dy"),
    tracer_extra_value_cols: Sequence[str] = (),
    align_to: str = "source_displacement",
    align_vector: Optional[Sequence[float]] = None,
) -> Tuple[MeasurementBatch, pd.DataFrame]:
    """
    Build a MeasurementBatch by pairing sources to tracers within the same frame, rotating
    both the connection vector (tracer_pos - source_pos) and tracer displacement into
    a source-aligned frame.

    Returns (batch, meta_df) where:
      - batch.coords are rotated connection vectors (shape N, 2)
      - batch.values are rotated tracer displacement vectors plus any extra tracer values
      - meta_df tracks frame + indices of the contributing source/tracer rows

    Args:
        source_positions: DataFrame with at least frame_col and position_cols.
        source_displacements: DataFrame with same length/ordering as source_positions.
            Used to define the rotation (unless align_to="fixed_vector").
        tracer_positions: DataFrame with at least frame_col and position_cols.
        tracer_displacements: DataFrame with same length/ordering as tracer_positions.
        frame_col: Column name identifying frame/time in all DataFrames.
        position_cols: Two columns for x/y positions.
        source_disp_cols: Two columns for source displacement used to build the rotation.
        tracer_disp_cols: Two columns for tracer displacement to rotate.
        tracer_extra_value_cols: Additional tracer columns to append (unrotated) to values.
        align_to: "source_displacement" (default) or "fixed_vector".
        align_vector: Optional 2-vector if align_to == "fixed_vector".

    Example:
        batch, meta = connection_batch_from_dataframes(
            source_positions=src_pos_df,
            source_displacements=src_disp_df,
            tracer_positions=tr_pos_df,
            tracer_displacements=tr_disp_df,
            frame_col="frame",
            position_cols=("x", "y"),
            source_disp_cols=("dx", "dy"),
            tracer_disp_cols=("dx", "dy"),
        )
        mean, weights, counts, centers = run_cdv([batch], config)
    """
    if len(position_cols) != 2:
        raise ValueError("position_cols must have length 2 (x, y).")
    if len(source_disp_cols) != 2 or len(tracer_disp_cols) != 2:
        raise ValueError("source_disp_cols and tracer_disp_cols must each have length 2.")
    tracer_extra_value_cols = tuple(tracer_extra_value_cols)

    def _array(df: pd.DataFrame, cols: Sequence[str], name: str) -> np.ndarray:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{name} missing columns: {missing}")
        return df.loc[:, cols].to_numpy(dtype=float)

    def _frames_match(df_a: pd.DataFrame, df_b: pd.DataFrame, label: str) -> np.ndarray:
        if frame_col not in df_a.columns or frame_col not in df_b.columns:
            raise KeyError(f"{label} requires '{frame_col}' in both DataFrames.")
        fa = df_a[frame_col].to_numpy()
        fb = df_b[frame_col].to_numpy()
        if len(fa) != len(fb):
            raise ValueError(f"{label} frame vectors differ in length.")
        if not np.array_equal(fa, fb):
            raise ValueError(f"{label} rows must align on {frame_col}.")
        return fa

    if len(source_positions) != len(source_displacements):
        raise ValueError("source_positions and source_displacements must have the same length.")
    if len(tracer_positions) != len(tracer_displacements):
        raise ValueError("tracer_positions and tracer_displacements must have the same length.")

    source_frames = _frames_match(source_positions, source_displacements, "source")
    tracer_frames = _frames_match(tracer_positions, tracer_displacements, "tracer")

    src_pos = _array(source_positions, position_cols, "source_positions")
    src_disp = _array(source_displacements, source_disp_cols, "source_displacements")
    trac_pos = _array(tracer_positions, position_cols, "tracer_positions")
    trac_disp = _array(tracer_displacements, tracer_disp_cols, "tracer_displacements")
    trac_extra = (
        _array(tracer_displacements, tracer_extra_value_cols, "tracer_extra_value_cols")
        if tracer_extra_value_cols
        else None
    )

    if align_to not in ("source_displacement", "fixed_vector"):
        raise ValueError("align_to must be 'source_displacement' or 'fixed_vector'.")
    fixed_R = None
    if align_to == "fixed_vector":
        if align_vector is None:
            raise ValueError("align_vector must be provided when align_to='fixed_vector'.")
        fixed_R = source_frame_from_displacement(np.asarray(align_vector, dtype=float))

    tracer_by_frame: dict = defaultdict(list)
    for idx, (f, p, d) in enumerate(zip(tracer_frames, trac_pos, trac_disp)):
        extra = trac_extra[idx] if trac_extra is not None else None
        tracer_by_frame[f].append((idx, p, d, extra))

    coords_rows = []
    values_rows = []
    meta_rows = []

    value_dim = len(tracer_disp_cols) + (trac_extra.shape[1] if trac_extra is not None else 0)
    for src_idx, (f, spos, sdisp) in enumerate(zip(source_frames, src_pos, src_disp)):
        R = fixed_R if fixed_R is not None else source_frame_from_displacement(sdisp)
        candidates = tracer_by_frame.get(f)
        if not candidates:
            continue
        for trac_idx, tpos, tdisp, extra in candidates:
            conn_vec = tpos - spos
            rotated_conn = rotate_into_frame(R, conn_vec.reshape(1, -1))[0]
            rotated_disp = rotate_into_frame(R, tdisp.reshape(1, -1))[0]
            values = (
                np.concatenate([rotated_disp, np.asarray(extra, dtype=float).reshape(1, -1)], axis=1)[0]
                if extra is not None
                else rotated_disp
            )
            coords_rows.append(rotated_conn)
            values_rows.append(values)
            meta_rows.append({"frame": f, "source_index": src_idx, "tracer_index": trac_idx})

    if coords_rows:
        coords_arr = np.vstack(coords_rows)
        values_arr = np.vstack(values_rows)
    else:
        coords_arr = np.empty((0, len(position_cols)), dtype=float)
        values_arr = np.empty((0, value_dim), dtype=float)

    batch = MeasurementBatch(coords=coords_arr, values=values_arr)
    meta_df = pd.DataFrame(meta_rows, columns=["frame", "source_index", "tracer_index"])
    return batch, meta_df
