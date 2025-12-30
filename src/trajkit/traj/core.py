from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _as_2d_float_array(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"x must be 2D array (T, D). Got shape {arr.shape}")
    if arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError(f"x must have shape (T>=1, D>=1). Got {arr.shape}")
    return arr


def _as_1d_array(a: Optional[np.ndarray], dtype) -> Optional[np.ndarray]:
    if a is None:
        return None
    arr = np.asarray(a, dtype=dtype)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {arr.shape}")
    return arr


def _check_len(name: str, arr: Optional[np.ndarray], T: int) -> None:
    if arr is None:
        return
    if len(arr) != T:
        raise ValueError(f"{name} length must equal T={T}. Got len={len(arr)}")


@dataclass(frozen=True)
class Trajectory:
    """
    A single tracked object (particle, bacterium, etc.)

    Core arrays:
      - x: (T, D) positions in D dimensions.
      - t: (T,) time in seconds (optional; can be non-uniform).
      - frame: (T,) frame indices (optional; integer).
      - valid: (T,) boolean mask (optional).

    Metadata:
      - label: grouping label (e.g., "passive", "bacteria", "janus_active").
      - track_features: per-trajectory scalars (diameter_um, etc.)
      - frame_features: per-frame arrays (area, theta, intensity, etc.)
      - meta: arbitrary metadata.
    """
    track_id: str
    x: np.ndarray

    # timing
    t: Optional[np.ndarray] = None
    frame: Optional[np.ndarray] = None
    frame_rate_hz: Optional[float] = None  # for converting frame->seconds when uniform

    valid: Optional[np.ndarray] = None  # (T,)

    label: Optional[str] = None
    track_features: Dict[str, Any] = field(default_factory=dict)       # scalars (or small objects)
    frame_features: Dict[str, np.ndarray] = field(default_factory=dict)  # arrays of length T
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        x = _as_2d_float_array(self.x)
        object.__setattr__(self, "x", x)
        T, _D = x.shape

        t = _as_1d_array(self.t, float)
        frame = _as_1d_array(self.frame, int)
        valid = _as_1d_array(self.valid, bool)

        _check_len("t", t, T)
        _check_len("frame", frame, T)
        _check_len("valid", valid, T)

        # Validate timing availability:
        # Need either explicit t, or (frame + frame_rate_hz) to compute time.
        if t is None:
            if frame is None or self.frame_rate_hz is None:
                raise ValueError(
                    "Provide either t (seconds) OR (frame + frame_rate_hz). "
                    "Both can be provided too."
                )
            if self.frame_rate_hz <= 0:
                raise ValueError("frame_rate_hz must be > 0")

        # Validate monotonicity (nice to enforce early)
        if t is not None:
            if not np.all(np.isfinite(t)):
                raise ValueError("t contains non-finite values")
            if np.any(np.diff(t) < 0):
                raise ValueError("t must be non-decreasing")
        if frame is not None:
            if np.any(frame < 0):
                raise ValueError("frame contains negative indices")
            if np.any(np.diff(frame) < 0):
                raise ValueError("frame must be non-decreasing")

        # Frame features must be length T
        for k, arr in self.frame_features.items():
            arr2 = np.asarray(arr)
            if arr2.ndim == 0:
                raise ValueError(f"frame_features['{k}'] must be array-like length T")
            if len(arr2) != T:
                raise ValueError(f"frame_features['{k}'] length must be T={T}, got {len(arr2)}")
            # store as numpy array
            self.frame_features[k] = arr2  # type: ignore[misc]

        # Store normalized arrays back (frozen dataclass workaround)
        object.__setattr__(self, "t", t)
        object.__setattr__(self, "frame", frame)
        object.__setattr__(self, "valid", valid)

    @property
    def T(self) -> int:
        return int(self.x.shape[0])

    @property
    def D(self) -> int:
        return int(self.x.shape[1])

    def time_seconds(self) -> np.ndarray:
        """
        Returns time array in seconds.
        - If t is provided: returns it (can be non-uniform).
        - Else: computes from frame and frame_rate_hz.
        """
        if self.t is not None:
            return self.t.astype(float, copy=False)
        assert self.frame is not None and self.frame_rate_hz is not None
        return self.frame.astype(float, copy=False) / float(self.frame_rate_hz)

    def dt(self) -> np.ndarray:
        """Per-step delta time, length T-1."""
        tt = self.time_seconds()
        if len(tt) < 2:
            return np.array([], dtype=float)
        return np.diff(tt)

    # statistics helpers
    def msd(self, *, lags: Optional[Iterable[int]] = None, dims: Optional[Iterable[int]] = None):
        """Mean squared displacement for this trajectory."""
        from trajkit.stats import msd

        return msd(self, lags=lags, dims=dims)

    def is_uniform_sampling(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        """Checks if dt is (approximately) constant."""
        d = self.dt()
        if len(d) < 2:
            return True
        return bool(np.allclose(d, d[0], rtol=rtol, atol=atol))

    def with_mask(self, mask: np.ndarray) -> "Trajectory":
        """Return a new trajectory containing only samples where mask is True."""
        mask = np.asarray(mask, dtype=bool)
        if mask.ndim != 1 or len(mask) != self.T:
            raise ValueError("mask must be boolean array of length T")
        new_frame_features = {k: v[mask] for k, v in self.frame_features.items()}
        return replace(
            self,
            x=self.x[mask],
            t=None if self.t is None else self.t[mask],
            frame=None if self.frame is None else self.frame[mask],
            valid=None if self.valid is None else self.valid[mask],
            frame_features=new_frame_features,
        )

    def to_frame_dataframe(self) -> pd.DataFrame:
        """
        Long-form per-frame table.
        Columns: track_id, t, frame, x0..x{D-1}, valid, ff_* features
        """
        data: Dict[str, Any] = {"track_id": np.repeat(self.track_id, self.T)}

        # timing
        data["t"] = self.time_seconds()
        data["frame"] = self.frame if self.frame is not None else pd.array([pd.NA] * self.T, dtype="Int64")

        # positions
        for j in range(self.D):
            data[f"x{j}"] = self.x[:, j]

        if self.valid is not None:
            data["valid"] = self.valid
        else:
            data["valid"] = True

        # per-frame features
        for k, arr in self.frame_features.items():
            data[f"ff_{k}"] = np.asarray(arr)

        return pd.DataFrame(data)

    @staticmethod
    def from_frame_dataframe(
        df: pd.DataFrame,
        *,
        track_id: Optional[str] = None,
        D: Optional[int] = None,
        frame_rate_hz: Optional[float] = None,
        label: Optional[str] = None,
        track_features: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "Trajectory":
        """
        Construct a Trajectory from a per-frame dataframe for ONE track_id.
        Expected columns: t and/or frame, x0..x{D-1}, optional valid, and ff_* features.
        """
        if track_id is None:
            if "track_id" not in df.columns:
                raise ValueError("df must contain 'track_id' or provide track_id explicitly")
            uniq = df["track_id"].unique()
            if len(uniq) != 1:
                raise ValueError("df must contain exactly one track_id")
            track_id = str(uniq[0])

        # infer D if not provided
        if D is None:
            pos_cols = [c for c in df.columns if c.startswith("x") and c[1:].isdigit()]
            if not pos_cols:
                raise ValueError("Could not infer D (no x0, x1, ... columns found)")
            D = max(int(c[1:]) for c in pos_cols) + 1

        x = np.column_stack([df[f"x{j}"].to_numpy(dtype=float) for j in range(D)])

        t = df["t"].to_numpy(dtype=float) if "t" in df.columns and df["t"].notna().all() else None

        frame = None
        if "frame" in df.columns:
            # allow nullable frames; if all NA => None
            if df["frame"].notna().any():
                frame = df["frame"].to_numpy(dtype=int)

        valid = df["valid"].to_numpy(dtype=bool) if "valid" in df.columns else None

        frame_features = {}
        for c in df.columns:
            if c.startswith("ff_"):
                frame_features[c[3:]] = df[c].to_numpy()

        return Trajectory(
            track_id=str(track_id),
            x=x,
            t=t,
            frame=frame,
            frame_rate_hz=frame_rate_hz,
            valid=valid,
            label=label,
            track_features=dict(track_features or {}),
            frame_features=frame_features,
            meta=dict(meta or {}),
        )


@dataclass
class TrajectorySet:
    """
    A collection of trajectories from one dataset/experiment.
    """
    dataset_id: str
    trajectories: Dict[str, Trajectory] = field(default_factory=dict)

    # dataset-level metadata
    units: Dict[str, str] = field(default_factory=lambda: {"t": "s", "x": "arb"})
    calibration: Dict[str, Any] = field(default_factory=dict)
    conditions: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        dataset_id: str,
        track_id_col: str,
        position_cols: Sequence[str],
        time_col: Optional[str] = None,
        frame_col: Optional[str] = None,
        frame_rate_hz: Optional[float] = None,
        label_col: Optional[str] = None,
        track_feature_cols: Sequence[str] = (),
        frame_feature_cols: Sequence[str] = (),
        valid_col: Optional[str] = None,
        units: Optional[Dict[str, str]] = None,
        calibration: Optional[Dict[str, Any]] = None,
        conditions: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
        sort_within_track: bool = True,
    ) -> "TrajectorySet":
        """
        Build a TrajectorySet from a long-form per-frame DataFrame.

        Args:
            df: Per-frame table containing coordinates and timing information.
            dataset_id: Identifier for the resulting TrajectorySet.
            track_id_col: Column holding the per-track identifier.
            position_cols: Ordered columns that make up the coordinate vector.
            time_col: Optional column giving timestamps in seconds.
            frame_col: Optional column giving frame indices.
            frame_rate_hz: Required when using frame_col without time_col.
            label_col: Optional column providing per-track labels.
            track_feature_cols: Columns treated as per-track scalars (must be constant per track).
            frame_feature_cols: Columns copied into frame_features for each trajectory.
            valid_col: Optional boolean column marking valid samples.
            sort_within_track: Sort each track by time/frame before constructing.
        """
        pos_cols = list(position_cols)
        if not pos_cols:
            raise ValueError("position_cols must be non-empty.")
        if time_col is None and frame_col is None:
            raise ValueError("Provide at least one of time_col or frame_col.")
        if time_col is None and frame_rate_hz is None:
            raise ValueError("frame_rate_hz is required when using frame_col without time_col.")

        required_cols = [track_id_col, *pos_cols]
        optional_cols = [
            c for c in (time_col, frame_col, label_col, valid_col) if c is not None
        ]
        optional_cols.extend(list(track_feature_cols))
        optional_cols.extend(list(frame_feature_cols))
        missing = [
            c for c in required_cols + optional_cols if c is not None and c not in df.columns
        ]
        if missing:
            raise KeyError(f"DataFrame missing columns: {missing}")

        ts_kwargs: Dict[str, Any] = {"dataset_id": dataset_id}
        if units is not None:
            ts_kwargs["units"] = dict(units)
        if calibration is not None:
            ts_kwargs["calibration"] = dict(calibration)
        if conditions is not None:
            ts_kwargs["conditions"] = dict(conditions)
        if meta is not None:
            ts_kwargs["meta"] = dict(meta)
        ts = cls(**ts_kwargs)

        if df.empty:
            return ts

        sort_keys = [c for c in (time_col, frame_col) if c is not None]

        for tid, df_tid in df.groupby(track_id_col, sort=False):
            df_tid = df_tid.copy()
            if sort_within_track and sort_keys:
                df_tid = df_tid.sort_values(sort_keys).reset_index(drop=True)
            else:
                df_tid = df_tid.reset_index(drop=True)

            label = None
            if label_col is not None:
                uniq = df_tid[label_col].dropna().unique()
                if len(uniq) > 1:
                    raise ValueError(f"label_col '{label_col}' varies within track '{tid}'.")
                label = None if len(uniq) == 0 else uniq[0]

            track_features: Dict[str, Any] = {}
            for col in track_feature_cols:
                vals = df_tid[col].dropna().unique()
                if len(vals) > 1:
                    raise ValueError(f"track_feature column '{col}' varies within track '{tid}'.")
                if len(vals) == 0:
                    continue
                v = vals[0]
                track_features[col] = v.item() if hasattr(v, "item") else v

            frame_features = {col: df_tid[col].to_numpy() for col in frame_feature_cols}

            label_value = None
            if label is not None:
                label_value = label.item() if hasattr(label, "item") else label

            tr = Trajectory(
                track_id=str(tid),
                x=df_tid.loc[:, pos_cols].to_numpy(dtype=float),
                t=df_tid[time_col].to_numpy(dtype=float) if time_col is not None else None,
                frame=df_tid[frame_col].to_numpy(dtype=int) if frame_col is not None else None,
                frame_rate_hz=frame_rate_hz,
                valid=df_tid[valid_col].to_numpy(dtype=bool) if valid_col is not None else None,
                label=label_value,
                track_features=track_features,
                frame_features=frame_features,
            )
            ts.add(tr)

        return ts

    def add(self, tr: Trajectory) -> None:
        if tr.track_id in self.trajectories:
            raise KeyError(f"Trajectory with track_id='{tr.track_id}' already exists")
        self.trajectories[tr.track_id] = tr

    def get(self, track_id: str) -> Trajectory:
        return self.trajectories[track_id]

    def ids(self) -> Tuple[str, ...]:
        return tuple(self.trajectories.keys())

    def summary_table(self) -> pd.DataFrame:
        """
        One row per trajectory: track_id, label, T, D, duration, plus track_features (tf_*).
        """
        rows = []
        for tid, tr in self.trajectories.items():
            tt = tr.time_seconds()
            duration = float(tt[-1] - tt[0]) if len(tt) > 1 else 0.0
            row: Dict[str, Any] = {
                "track_id": tid,
                "label": tr.label,
                "T": tr.T,
                "D": tr.D,
                "duration_s": duration,
                "uniform_sampling": tr.is_uniform_sampling(),
            }
            for k, v in tr.track_features.items():
                row[f"tf_{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)

    def filter(self, predicate) -> "TrajectorySet":
        """
        Return a new TrajectorySet with trajectories for which predicate(tr) is True.
        """
        out = TrajectorySet(
            dataset_id=self.dataset_id,
            units=dict(self.units),
            calibration=dict(self.calibration),
            conditions=dict(self.conditions),
            meta=dict(self.meta),
        )
        for tr in self.trajectories.values():
            if predicate(tr):
                out.trajectories[tr.track_id] = tr
        return out

    # statistics helpers
    def msd(
        self,
        *,
        lags: Optional[Iterable[int]] = None,
        dims: Optional[Iterable[int]] = None,
        aggregate: bool = False,
    ):
        """Mean squared displacement over all trajectories."""
        from trajkit.stats import msd_trajectory_set

        return msd_trajectory_set(self, lags=lags, dims=dims, aggregate=aggregate)
