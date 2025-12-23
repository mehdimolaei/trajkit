from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from trajkit.traj.core import Trajectory, TrajectorySet


def save_trajectory_set(
    ts: TrajectorySet,
    folder: str | Path,
    *,
    tracks_filename: str = "tracks.parquet",
    index_filename: str = "tracks_index.parquet",
    meta_filename: str = "meta.json",
) -> None:
    """
    Writes:
      folder/
        tracks.parquet        (per-frame long table)
        tracks_index.parquet  (per-track summary + tf_*)
        meta.json             (dataset metadata, units, calibration, conditions)
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)

    # per-frame table
    frames = [tr.to_frame_dataframe() for tr in ts.trajectories.values()]
    tracks_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    tracks_df.to_parquet(folder / tracks_filename, index=False)

    # per-track table
    index_df = ts.summary_table()
    index_df.to_parquet(folder / index_filename, index=False)

    # meta
    meta: Dict[str, Any] = {
        "dataset_id": ts.dataset_id,
        "units": ts.units,
        "calibration": ts.calibration,
        "conditions": ts.conditions,
        "meta": ts.meta,
    }
    (folder / meta_filename).write_text(json.dumps(meta, indent=2))


def load_trajectory_set(
    folder: str | Path,
    *,
    tracks_filename: str = "tracks.parquet",
    index_filename: str = "tracks_index.parquet",
    meta_filename: str = "meta.json",
    frame_rate_hz: Optional[float] = None,
) -> TrajectorySet:
    """
    Reconstruct a TrajectorySet from saved parquet + meta.
    If frame_rate_hz is provided, it will be applied to trajectories that lack explicit t.
    """
    folder = Path(folder)
    meta = json.loads((folder / meta_filename).read_text())

    ts = TrajectorySet(
        dataset_id=meta["dataset_id"],
        units=dict(meta.get("units", {})),
        calibration=dict(meta.get("calibration", {})),
        conditions=dict(meta.get("conditions", {})),
        meta=dict(meta.get("meta", {})),
    )

    tracks_df = pd.read_parquet(folder / tracks_filename)
    index_df = pd.read_parquet(folder / index_filename) if (folder / index_filename).exists() else None

    # optional: map per-track label + tf_* from index table
    per_track: Dict[str, Dict[str, Any]] = {}
    if index_df is not None and not index_df.empty:
        for _, row in index_df.iterrows():
            tid = str(row["track_id"])
            tf = {c[3:]: row[c] for c in index_df.columns if c.startswith("tf_") and pd.notna(row[c])}
            per_track[tid] = {
                "label": None if pd.isna(row.get("label", None)) else row.get("label", None),
                "track_features": tf,
            }

    if tracks_df.empty:
        return ts

    for tid, df_tid in tracks_df.groupby("track_id", sort=False):
        tid = str(tid)
        label = per_track.get(tid, {}).get("label", None)
        track_features = per_track.get(tid, {}).get("track_features", {})

        tr = Trajectory.from_frame_dataframe(
            df_tid.reset_index(drop=True),
            track_id=tid,
            frame_rate_hz=frame_rate_hz,
            label=label,
            track_features=track_features,
        )
        ts.trajectories[tid] = tr

    return ts
