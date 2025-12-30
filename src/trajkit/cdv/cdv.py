from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from .accumulate import CorrelationEnsembleAccumulator, EnsembleAccumulator
from .neighbors import RegularGridNeighborFinder
from .types import CDVConfig, MeasurementBatch
from .corr import correlation_batch, distance_threshold_pair_filter
from .distance import euclidean_distance
from trajkit.traj import TrajectorySet


def _run_cdv_batches(
    batches: Iterable[MeasurementBatch],
    config: CDVConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Original CDV runner for a precomputed set of MeasurementBatch instances.
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


def _run_cdv_from_trajset(
    ts: TrajectorySet,
    delta_t: float,
    grid_centers: np.ndarray,
    *,
    time_tol: float = 1e-3,
    kernel: float | dict | callable = 30.0,
    max_pair_distance: Optional[float] = None,
    value_fn=None,
    weight_fn=None,
    distance_fn=euclidean_distance,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Build a flow field from a TrajectorySet by:
      1) assembling per-frame displacements separated by delta_t,
      2) pairing tracers/sources within each frame via correlation_batch,
      3) rotating pairs into source-x frame, and
      4) accumulating onto grid_centers to produce a flow field.
    """
    if delta_t <= 0:
        raise ValueError("delta_t must be positive.")
    if time_tol < 0:
        raise ValueError("time_tol must be non-negative.")
    if not ts.trajectories:
        raise ValueError("TrajectorySet is empty.")

    dim: Optional[int] = None
    disp_rows = []
    for tid, tr in ts.trajectories.items():
        t = tr.time_seconds()
        if len(t) == 0:
            continue
        frames = (
            np.asarray(tr.frame, dtype=int)
            if tr.frame is not None
            else np.arange(len(t), dtype=int)
        )
        targets = t + float(delta_t)
        idx = np.searchsorted(t, targets, side="left")

        if dim is None:
            dim = tr.D
        elif tr.D != dim:
            raise ValueError("All trajectories must share the same dimensionality.")

        for i, j in enumerate(idx):
            if j >= len(t):
                continue
            if abs(float(t[j] - targets[i])) > time_tol:
                continue
            dx = tr.x[j] - tr.x[i]
            row = {
                "track_id": tid,
                "t": float(t[i]),
                "frame": int(frames[i]),
            }
            for k in range(tr.D):
                row[f"x{k}"] = float(tr.x[i, k])
                row[f"dx{k}"] = float(dx[k])
            disp_rows.append(row)

    if not disp_rows:
        raise ValueError("No displacement pairs found for the given delta_t/time_tol.")
    assert dim is not None

    disp_df = pd.DataFrame(disp_rows)
    position_cols = [f"x{k}" for k in range(dim)]
    motion_cols = [f"dx{k}" for k in range(dim)]
    pair_filter = (
        distance_threshold_pair_filter(float(max_pair_distance))
        if max_pair_distance is not None
        else None
    )

    ensemble = CorrelationEnsembleAccumulator(
        grid_centers,
        kernel=kernel,
        value_fn=value_fn or (lambda rel, tracer, source, meta_row: tracer),
        weight_fn=weight_fn or (lambda rel, tracer, source, meta_row: np.linalg.norm(source)),
        distance_fn=distance_fn,
    )

    for frame in disp_df["frame"].unique():
        disp_temp = disp_df[disp_df["frame"] == frame]
        batch, _ = correlation_batch(
            disp_temp,
            disp_temp,
            source_frame_col="frame",
            tracer_frame_col="frame",
            source_position_cols=position_cols,
            tracer_position_cols=position_cols,
            source_motion_cols=motion_cols,
            tracer_motion_cols=motion_cols,
            pair_filter=pair_filter,
        )
        ensemble.add(batch.rotate_to_source_x())

    mean, sum_w, counts = ensemble.finalize()
    centers = ensemble.centers
    return mean, sum_w, counts, centers


def run_cdv(
    data: TrajectorySet | Iterable[MeasurementBatch],
    config: Optional[CDVConfig] = None,
    *,
    delta_t: Optional[float] = None,
    grid_centers: Optional[np.ndarray] = None,
    time_tol: float = 1e-3,
    kernel: float | dict | callable = 30.0,
    max_pair_distance: Optional[float] = None,
    value_fn=None,
    weight_fn=None,
    distance_fn=euclidean_distance,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """
    Dispatching runner:
      - If `data` is a TrajectorySet, it builds displacements/correlations and accumulates
        onto grid_centers to produce a flow field.
      - If `data` is an Iterable of MeasurementBatch, it runs the original accumulator
        using the provided config.
    """
    if isinstance(data, TrajectorySet):
        if delta_t is None or grid_centers is None:
            raise ValueError("Provide delta_t and grid_centers when calling run_cdv with a TrajectorySet.")
        return _run_cdv_from_trajset(
            data,
            delta_t,
            grid_centers,
            time_tol=time_tol,
            kernel=kernel,
            max_pair_distance=max_pair_distance,
            value_fn=value_fn,
            weight_fn=weight_fn,
            distance_fn=distance_fn,
        )

    batches_list = list(data)
    if config is None:
        raise ValueError("config is required when calling run_cdv with MeasurementBatch data.")
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
