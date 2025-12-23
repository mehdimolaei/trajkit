from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from trajkit.traj.core import Trajectory, TrajectorySet


def msd(
    tr: Trajectory,
    *,
    lags: Optional[Sequence[int]] = None,
    dims: Optional[Iterable[int]] = None,
) -> pd.DataFrame:
    """
    Mean squared displacement for one Trajectory.

    Args:
        tr: Trajectory to evaluate.
        lags: Iterable of integer lags (in frames/samples). Defaults to 1..T-1.
        dims: Subset of spatial dimensions to include (default: all).

    Returns:
        DataFrame with columns: lag (samples), tau_seconds (mean time difference), n (pairs),
        msd.
    """
    x = tr.x
    T, D = x.shape
    if T < 2:
        raise ValueError("Trajectory must have at least 2 samples to compute MSD.")

    dims_idx = tuple(dims) if dims is not None else tuple(range(D))
    for d in dims_idx:
        if d < 0 or d >= D:
            raise ValueError(f"dims contains out-of-bounds index {d} for D={D}")

    lags_arr = np.array(list(lags) if lags is not None else range(1, T), dtype=int)
    if (lags_arr <= 0).any():
        raise ValueError("lags must be positive integers.")
    lags_arr = np.unique(lags_arr)
    lags_arr = lags_arr[lags_arr < T]
    if len(lags_arr) == 0:
        raise ValueError("No valid lags to compute.")

    t = tr.time_seconds()
    results = []
    for lag in lags_arr:
        # displacement pairs
        disp = x[lag:, :] - x[:-lag, :]
        disp = disp[:, dims_idx]
        sq = np.sum(disp * disp, axis=1)
        n = len(sq)
        tau = float(np.mean(t[lag:] - t[:-lag]))
        results.append({"lag": int(lag), "tau_seconds": tau, "n": n, "msd": float(np.mean(sq))})
    return pd.DataFrame(results)


def msd_trajectory_set(
    ts: TrajectorySet,
    *,
    lags: Optional[Sequence[int]] = None,
    dims: Optional[Iterable[int]] = None,
    aggregate: bool = False,
) -> pd.DataFrame:
    """
    MSD for all trajectories in a TrajectorySet.

    Args:
        ts: TrajectorySet to evaluate.
        lags: Iterable of integer lags (in frames/samples). Defaults to per-trajectory 1..T-1.
        dims: Subset of spatial dimensions to include (default: all).
        aggregate: If True, return mean MSD across trajectories for each lag (with counts).

    Returns:
        DataFrame with columns track_id, lag, tau_seconds, n, msd. If aggregate=True,
        returns per-lag aggregated mean with columns lag, tau_seconds, n_pairs, msd_mean.
    """
    frames = []
    for tid, tr in ts.trajectories.items():
        df = msd(tr, lags=lags, dims=dims)
        df["track_id"] = str(tid)
        frames.append(df)
    if not frames:
        raise ValueError("TrajectorySet is empty; no MSD computed.")

    df_all = pd.concat(frames, ignore_index=True)
    if not aggregate:
        return df_all[["track_id", "lag", "tau_seconds", "n", "msd"]]

    agg = (
        df_all.groupby("lag", as_index=False)
        .agg(
            tau_seconds=("tau_seconds", "mean"),
            n_pairs=("n", "sum"),
            msd_mean=("msd", "mean"),
        )
        .sort_values("lag")
    )
    return agg
