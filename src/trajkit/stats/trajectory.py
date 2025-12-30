from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from trajkit.traj.core import Trajectory, TrajectorySet


def _validate_dims(D: int, dims: Optional[Iterable[int]]) -> Tuple[int, ...]:
    dims_idx = tuple(dims) if dims is not None else tuple(range(D))
    for d in dims_idx:
        if d < 0 or d >= D:
            raise ValueError(f"dims contains out-of-bounds index {d} for D={D}")
    return dims_idx


def _prepare_grid(tr: Trajectory, interpolate: bool, dims_idx: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (positions, times) possibly interpolated to contiguous frames.
    """
    x = tr.x[:, dims_idx]
    t = tr.time_seconds()
    if not interpolate:
        return x, t

    # Require frame info to interpolate onto integer grid
    if tr.frame is None:
        raise ValueError("interpolate=True requires trajectory.frame to be set.")
    frames = tr.frame.astype(int)
    if len(frames) < 2:
        return x, t
    frame_grid = np.arange(frames[0], frames[-1] + 1, dtype=int)

    x_interp = np.column_stack([np.interp(frame_grid, frames, tr.x[:, j]) for j in dims_idx])
    t_interp = np.interp(frame_grid, frames, t)
    return x_interp, t_interp


def _lag_array(lags: Optional[Sequence[int]], max_lag: Optional[int], T: int) -> np.ndarray:
    if lags is not None:
        lags_arr = np.array(list(lags), dtype=int)
        if (lags_arr <= 0).any():
            raise ValueError("lags must be positive integers.")
    else:
        max_l = max_lag if max_lag is not None else T - 1
        lags_arr = np.arange(1, min(max_l, T - 1) + 1, dtype=int)
    lags_arr = np.unique(lags_arr)
    lags_arr = lags_arr[lags_arr < T]
    if len(lags_arr) == 0:
        raise ValueError("No valid lags to compute.")
    return lags_arr


def _msd_sums(
    x: np.ndarray, t: np.ndarray, lags_arr: np.ndarray
) -> Dict[int, Tuple[int, np.ndarray, np.ndarray, float]]:
    """
    Returns per-lag: n_pairs, sum_sq_per_dim, sum_disp_per_dim, tau_mean.
    """
    results: Dict[int, Tuple[int, np.ndarray, np.ndarray, float]] = {}
    for lag in lags_arr:
        if lag >= len(x):
            continue
        disp = x[lag:, :] - x[:-lag, :]
        n_pairs = disp.shape[0]
        if n_pairs == 0:
            continue
        sum_sq_dim = np.sum(disp * disp, axis=0)
        sum_disp_dim = np.sum(disp, axis=0)
        tau = float(np.mean(t[lag:] - t[:-lag]))
        results[int(lag)] = (n_pairs, sum_sq_dim, sum_disp_dim, tau)
    return results


def msd(
    tr: Trajectory,
    *,
    lags: Optional[Sequence[int]] = None,
    max_lag: Optional[int] = None,
    dims: Optional[Iterable[int]] = None,
    interpolate: bool = False,
    return_sd: bool = False,
    per_dim: bool = False,
) -> pd.DataFrame:
    """
    Mean squared displacement for one Trajectory.

    Args:
        tr: Trajectory to evaluate.
        lags: Iterable of integer lags (in frames/samples). Defaults to 1..min(T-1, max_lag).
        max_lag: Maximum lag (frames) if lags is not provided.
        dims: Subset of spatial dimensions to include (default: all).
        interpolate: If True, interpolate to contiguous frames before computing displacements
            (requires trajectory.frame).
        return_sd: If True, include mean displacement components (per-dimension).
        per_dim: If True, include per-dimension MSD columns (msd_dim{j}).

    Returns:
        DataFrame with columns: lag, tau_seconds (mean time difference), n (pairs), msd.
        Optional: msd_dim{j}, sd_dim{j} when requested.
    """
    T, D = tr.x.shape
    if T < 2:
        raise ValueError("Trajectory must have at least 2 samples to compute MSD.")

    dims_idx = _validate_dims(D, dims)
    x_proc, t_proc = _prepare_grid(tr, interpolate=interpolate, dims_idx=dims_idx)
    lags_arr = _lag_array(lags, max_lag, len(x_proc))
    sums = _msd_sums(x_proc, t_proc, lags_arr)
    if not sums:
        raise ValueError("No valid lags to compute.")

    records = []
    for lag in sorted(sums.keys()):
        n_pairs, sum_sq_dim, sum_disp_dim, tau = sums[lag]
        msd_dim = sum_sq_dim / n_pairs
        rec = {
            "lag": lag,
            "tau_seconds": tau,
            "n": n_pairs,
            "msd": float(np.sum(msd_dim)),
        }
        if per_dim:
            for j, d in enumerate(dims_idx):
                rec[f"msd_dim{d}"] = float(msd_dim[j])
        if return_sd:
            sd_dim = sum_disp_dim / n_pairs
            for j, d in enumerate(dims_idx):
                rec[f"sd_dim{d}"] = float(sd_dim[j])
        records.append(rec)
    return pd.DataFrame(records)


def msd_trajectory_set(
    ts: TrajectorySet,
    *,
    lags: Optional[Sequence[int]] = None,
    max_lag: Optional[int] = None,
    dims: Optional[Iterable[int]] = None,
    interpolate: bool = False,
    aggregate: bool = False,
    aggregate_mode: str = "track",
    return_sd: bool = False,
    per_dim: bool = False,
) -> pd.DataFrame:
    """
    MSD for all trajectories in a TrajectorySet.

    Args:
        ts: TrajectorySet to evaluate.
        lags: Iterable of integer lags (in frames/samples). Defaults to per-trajectory 1..T-1.
        max_lag: Maximum lag (frames) if lags is not provided.
        dims: Subset of spatial dimensions to include (default: all included).
        interpolate: If True, interpolate each trajectory to contiguous frames before computing.
        aggregate: If True, aggregate across trajectories.
        aggregate_mode: "track" (mean across tracks, prior behavior) or "pair" (pair-weighted,
            matching MATLAB scripts).
        return_sd: If True, include mean displacement components.
        per_dim: If True, include per-dimension MSD columns.

    Returns:
        If aggregate is False: DataFrame with per-track rows and columns track_id, lag,
        tau_seconds, n, msd (plus optional per-dim / sd columns).

        If aggregate is True and aggregate_mode=="track": per-lag aggregated mean with columns
        lag, tau_seconds, n_pairs, msd_mean.

        If aggregate is True and aggregate_mode=="pair": pair-weighted aggregation with columns
        lag, tau_seconds, n_pairs, msd (plus optional per-dim / sd columns).
    """
    frames = []
    dims_idx: Tuple[int, ...] | None = None

    if aggregate_mode not in ("track", "pair"):
        raise ValueError("aggregate_mode must be 'track' or 'pair'.")

    # Pair-weighted accumulators
    pair_counts: Dict[int, int] = {}
    pair_sum_sq: Dict[int, np.ndarray] = {}
    pair_sum_disp: Dict[int, np.ndarray] = {}
    pair_tau_sum: Dict[int, float] = {}

    for tid, tr in ts.trajectories.items():
        if dims_idx is None:
            dims_idx = _validate_dims(tr.D, dims)
        df = msd(
            tr,
            lags=lags,
            max_lag=max_lag,
            dims=dims_idx,
            interpolate=interpolate,
            return_sd=return_sd,
            per_dim=per_dim,
        )
        df["track_id"] = str(tid)
        frames.append(df)

        if aggregate and aggregate_mode == "pair":
            # Recompute sums to accumulate with weights
            x_proc, t_proc = _prepare_grid(tr, interpolate=interpolate, dims_idx=dims_idx)
            lags_arr = _lag_array(lags, max_lag, len(x_proc))
            sums = _msd_sums(x_proc, t_proc, lags_arr)
            for lag, (n_pairs, sum_sq_dim, sum_disp_dim, tau) in sums.items():
                pair_counts[lag] = pair_counts.get(lag, 0) + n_pairs
                if lag not in pair_sum_sq:
                    pair_sum_sq[lag] = np.zeros_like(sum_sq_dim)
                    pair_sum_disp[lag] = np.zeros_like(sum_disp_dim)
                    pair_tau_sum[lag] = 0.0
                pair_sum_sq[lag] += sum_sq_dim
                pair_sum_disp[lag] += sum_disp_dim
                pair_tau_sum[lag] += tau * n_pairs

    if not frames:
        raise ValueError("TrajectorySet is empty; no MSD computed.")

    df_all = pd.concat(frames, ignore_index=True)
    if not aggregate:
        return df_all

    if aggregate_mode == "pair":
        rows = []
        for lag in sorted(pair_counts.keys()):
            n_pairs = pair_counts[lag]
            if n_pairs == 0:
                continue
            msd_dim = pair_sum_sq[lag] / n_pairs
            rec = {
                "lag": lag,
                "tau_seconds": pair_tau_sum[lag] / n_pairs,
                "n_pairs": n_pairs,
                "msd": float(np.sum(msd_dim)),
            }
            if per_dim:
                assert dims_idx is not None
                for j, d in enumerate(dims_idx):
                    rec[f"msd_dim{d}"] = float(msd_dim[j])
            if return_sd:
                sd_dim = pair_sum_disp[lag] / n_pairs
                assert dims_idx is not None
                for j, d in enumerate(dims_idx):
                    rec[f"sd_dim{d}"] = float(sd_dim[j])
            rows.append(rec)
        return pd.DataFrame(rows)

    # aggregate_mode == "track": average msd across tracks (prior behavior)
    agg_dict: Dict[str, Tuple[str, str]] = {
        "tau_seconds": ("tau_seconds", "mean"),
        "n_pairs": ("n", "sum"),
        "msd_mean": ("msd", "mean"),
    }
    for col in df_all.columns:
        if col.startswith("msd_dim"):
            agg_dict[f"{col}_mean"] = (col, "mean")
        if col.startswith("sd_dim"):
            agg_dict[f"{col}_mean"] = (col, "mean")

    agg = df_all.groupby("lag", as_index=False).agg(**agg_dict).sort_values("lag")
    return agg
