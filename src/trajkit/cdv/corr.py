from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence, Tuple, TypeVar

import numpy as np
import pandas as pd

from .distance import euclidean_distance
from .kernel import gaussian_kernel, hard_cutoff_kernel
from .neighbors import RegularGridNeighborFinder
from .types import RegularGridSpec

T = TypeVar("T")
# General pair-filter callable; extra args/kwargs are supported.
PairFilter = Callable[..., np.ndarray | bool]
WeightFn = Callable[[np.ndarray, np.ndarray, np.ndarray, pd.Series], float]
ValueFn = Callable[[np.ndarray, np.ndarray, np.ndarray, pd.Series], np.ndarray]


@dataclass
class CorrelationBatch:
    """
    Pairwise source/tracer data aligned by frame.

    relative_positions: (N, d) tracer_position - source_position
    tracer_motion: (N, qt)
    source_motion: (N, qs)
    meta: DataFrame with columns [frame, source_index, tracer_index]
    """

    relative_positions: np.ndarray
    tracer_motion: np.ndarray
    source_motion: np.ndarray
    meta: pd.DataFrame

    def __post_init__(self) -> None:
        rel = np.asarray(self.relative_positions, dtype=float)
        trac = np.asarray(self.tracer_motion, dtype=float)
        src = np.asarray(self.source_motion, dtype=float)
        if rel.ndim != 2:
            raise ValueError("relative_positions must be 2D (N, d).")
        if trac.ndim != 2:
            raise ValueError("tracer_motion must be 2D (N, qt).")
        if src.ndim != 2:
            raise ValueError("source_motion must be 2D (N, qs).")
        if not (rel.shape[0] == trac.shape[0] == src.shape[0]):
            raise ValueError("All arrays must share the same leading dimension N.")
        if not isinstance(self.meta, pd.DataFrame):
            raise TypeError("meta must be a pandas DataFrame.")
        if len(self.meta) != rel.shape[0]:
            raise ValueError("meta length must match number of pairings.")
        self.relative_positions = rel
        self.tracer_motion = trac
        self.source_motion = src

    @property
    def n_pairs(self) -> int:
        return int(self.relative_positions.shape[0])

    @property
    def position_dim(self) -> int:
        return int(self.relative_positions.shape[1])

    @property
    def tracer_motion_dim(self) -> int:
        return int(self.tracer_motion.shape[1])

    @property
    def source_motion_dim(self) -> int:
        return int(self.source_motion.shape[1])

    def to_dataframe(
        self,
        *,
        relative_position_cols: Sequence[str] | None = None,
        tracer_motion_cols: Sequence[str] | None = None,
        source_motion_cols: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Convert the correlation batch into a flat DataFrame combining meta and arrays.

        Column ordering: meta columns, relative position columns, tracer motion columns,
        then source motion columns.
        """

        def _columns(default_prefix: str, count: int, provided: Sequence[str] | None) -> list[str]:
            if provided is None:
                return [f"{default_prefix}{i}" for i in range(count)]
            if len(provided) != count:
                raise ValueError(f"Expected {count} column names for {default_prefix}, got {len(provided)}.")
            return list(provided)

        rp_cols = _columns("rel_pos_", self.position_dim, relative_position_cols)
        trac_cols = _columns("tracer_motion_", self.tracer_motion_dim, tracer_motion_cols)
        src_cols = _columns("source_motion_", self.source_motion_dim, source_motion_cols)

        meta_df = self.meta.reset_index(drop=True).copy()
        rel_df = pd.DataFrame(self.relative_positions, columns=rp_cols)
        tracer_df = pd.DataFrame(self.tracer_motion, columns=trac_cols)
        source_df = pd.DataFrame(self.source_motion, columns=src_cols)
        return pd.concat([meta_df, rel_df, tracer_df, source_df], axis=1)

    def rotate_to_source_x(self) -> CorrelationBatch:
        """
        Rotate each pair so the source motion aligns with the x-axis.

        Returns a new CorrelationBatch with rotated relative positions and motions.
        A column ``rotation_angle_rad`` is added to meta indicating the angle (original orientation)
        that was removed. Zero-length source motion vectors keep identity rotation.
        """
        if self.source_motion_dim < 2:
            raise ValueError("Rotation is not valid for scalar source motion.")
        if self.position_dim < 2:
            raise ValueError("Rotation requires position_dim >= 2; not valid for scalar positions.")
        if self.tracer_motion_dim < 2:
            raise ValueError("Rotation requires tracer motion with at least 2 dimensions.")

        rel_rot = np.array(self.relative_positions, copy=True)
        tracer_rot = np.array(self.tracer_motion, copy=True)
        source_rot = np.array(self.source_motion, copy=True)
        angles = np.zeros(self.n_pairs, dtype=float)

        for i in range(self.n_pairs):
            vx, vy = self.source_motion[i, 0], self.source_motion[i, 1]
            angle = float(np.arctan2(vy, vx))
            angles[i] = angle
            if np.isclose(vx, 0.0) and np.isclose(vy, 0.0):
                # No direction; leave as is.
                continue
            c, s = np.cos(-angle), np.sin(-angle)
            rot = np.array([[c, -s], [s, c]])

            rel_rot[i, :2] = rot @ self.relative_positions[i, :2]
            tracer_rot[i, :2] = rot @ self.tracer_motion[i, :2]
            source_rot[i, :2] = rot @ self.source_motion[i, :2]

        meta_rot = self.meta.reset_index(drop=True).copy()
        meta_rot["rotation_angle_rad"] = angles
        return CorrelationBatch(
            relative_positions=rel_rot,
            tracer_motion=tracer_rot,
            source_motion=source_rot,
            meta=meta_rot,
        )


def correlation_batch(
    source_df: pd.DataFrame,
    tracer_df: pd.DataFrame,
    *,
    source_frame_col: str = "frame",
    tracer_frame_col: str = "frame",
    source_position_cols: Sequence[str],
    tracer_position_cols: Sequence[str],
    source_motion_cols: Sequence[str],
    tracer_motion_cols: Sequence[str],
    pair_filter: PairFilter | None = None,
    pair_filter_args: Sequence[Any] | None = None,
    pair_filter_kwargs: Mapping[str, Any] | None = None,
    ensemble_fn: Callable[[CorrelationBatch], T] | None = None,
) -> Tuple[CorrelationBatch, T | None]:
    """
    Align sources and tracers by frame and build pairwise correlation inputs.

    Each tracer in a given frame is paired with every source in the same frame.
    The relative position is tracer_position - source_position.

    Args:
        source_df: DataFrame containing source positions, motion, and frame column.
        tracer_df: DataFrame containing tracer positions, motion, and frame column.
        source_frame_col: Column name in source_df used to align rows.
        tracer_frame_col: Column name in tracer_df used to align rows.
        source_position_cols: Ordered columns in source_df representing position (length >= 1).
        tracer_position_cols: Ordered columns in tracer_df representing position
            (must match length of source_position_cols).
        source_motion_cols: Columns in source_df representing motion/displacement (length >= 1).
        tracer_motion_cols: Columns in tracer_df representing motion/displacement (length >= 1).
        pair_filter: Optional callable receiving
            (relative_positions, tracer_motion, source_motion, meta_df) for a candidate pair and
            returning a boolean or length-1 boolean mask indicating whether to keep it. Extra positional
            and keyword arguments can be passed via pair_filter_args/pair_filter_kwargs.
        pair_filter_args: Optional positional arguments forwarded to pair_filter.
        pair_filter_kwargs: Optional keyword arguments forwarded to pair_filter.
        ensemble_fn: Optional callable that consumes the CorrelationBatch and produces
            an ensemble-ready object (to be defined by caller).

    Returns:
        (batch, ensemble_output) where ensemble_output is None if ensemble_fn is not provided.
    """
    src_pos_cols = list(source_position_cols)
    trac_pos_cols = list(tracer_position_cols)
    src_motion_cols = list(source_motion_cols)
    trac_motion_cols = list(tracer_motion_cols)

    if len(src_pos_cols) == 0 or len(trac_pos_cols) == 0:
        raise ValueError("Position column selections must be non-empty.")
    if len(src_pos_cols) != len(trac_pos_cols):
        raise ValueError("Source and tracer position column selections must have the same length.")
    if len(src_motion_cols) == 0 or len(trac_motion_cols) == 0:
        raise ValueError("Motion column selections must be non-empty.")

    def _require_columns(df: pd.DataFrame, cols: Iterable[str], label: str) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise KeyError(f"{label} missing columns: {missing}")

    _require_columns(source_df, [source_frame_col, *src_pos_cols, *src_motion_cols], "source_df")
    _require_columns(tracer_df, [tracer_frame_col, *trac_pos_cols, *trac_motion_cols], "tracer_df")

    src_frames = source_df[source_frame_col].to_numpy()
    trac_frames = tracer_df[tracer_frame_col].to_numpy()

    src_pos = source_df.loc[:, src_pos_cols].to_numpy(dtype=float)
    trac_pos = tracer_df.loc[:, trac_pos_cols].to_numpy(dtype=float)
    src_motion = source_df.loc[:, src_motion_cols].to_numpy(dtype=float)
    trac_motion = tracer_df.loc[:, trac_motion_cols].to_numpy(dtype=float)

    # Track original indices for meta to make downstream joins easier.
    src_indices = source_df.index.to_numpy()
    trac_indices = tracer_df.index.to_numpy()

    src_by_frame: Dict[int, list[int]] = {}
    trac_by_frame: Dict[int, list[int]] = {}
    for i, f in enumerate(src_frames):
        src_by_frame.setdefault(f, []).append(i)
    for i, f in enumerate(trac_frames):
        trac_by_frame.setdefault(f, []).append(i)

    common_frames = sorted(set(src_by_frame) & set(trac_by_frame))
    rel_rows = []
    tracer_rows = []
    source_rows = []
    meta_rows = []
    args = tuple(pair_filter_args or ())
    kwargs = dict(pair_filter_kwargs or {})

    for f in common_frames:
        for si in src_by_frame[f]:
            spos = src_pos[si]
            smotion = src_motion[si]
            sidx = src_indices[si]
            for ti in trac_by_frame[f]:
                tpos = trac_pos[ti]
                tmotion = trac_motion[ti]
                tidx = trac_indices[ti]
                rel_vec = tpos - spos
                meta_row = {"frame": f, "source_index": sidx, "tracer_index": tidx}

                if pair_filter is not None:
                    candidate_meta = pd.DataFrame([meta_row], columns=["frame", "source_index", "tracer_index"])
                    mask = np.asarray(
                        pair_filter(
                            rel_vec.reshape(1, -1),
                            np.asarray([tmotion]),
                            np.asarray([smotion]),
                            candidate_meta,
                            *args,
                            **kwargs,
                        ),
                        dtype=bool,
                    )
                    if mask.ndim == 0:
                        keep = bool(mask)
                    elif mask.shape == (1,):
                        keep = bool(mask[0])
                    else:
                        raise ValueError(
                            "pair_filter must return a boolean or a 1-element boolean mask for each pair."
                        )
                    if not keep:
                        continue

                rel_rows.append(rel_vec)
                tracer_rows.append(tmotion)
                source_rows.append(smotion)
                meta_rows.append(meta_row)

    if rel_rows:
        rel_arr = np.vstack(rel_rows)
        tracer_arr = np.vstack(tracer_rows)
        source_arr = np.vstack(source_rows)
    else:
        d = len(src_pos_cols)
        rel_arr = np.empty((0, d), dtype=float)
        tracer_arr = np.empty((0, len(trac_motion_cols)), dtype=float)
        source_arr = np.empty((0, len(src_motion_cols)), dtype=float)

    meta_df = pd.DataFrame(meta_rows, columns=["frame", "source_index", "tracer_index"])

    batch = CorrelationBatch(
        relative_positions=rel_arr,
        tracer_motion=tracer_arr,
        source_motion=source_arr,
        meta=meta_df,
    )

    ensemble_output = ensemble_fn(batch) if ensemble_fn is not None else None
    return batch, ensemble_output


def _normalize_kernel(kernel_spec: Any) -> Callable[[np.ndarray], np.ndarray]:
    """
    Accepts a kernel spec (callable, numeric radius, or mapping) and returns a kernel function.
    """
    if callable(kernel_spec):
        return kernel_spec
    if isinstance(kernel_spec, Mapping):
        ktype = kernel_spec.get("type")
        if ktype == "hard":
            radius = float(kernel_spec["radius"])
            if radius <= 0:
                raise ValueError("Hard cutoff radius must be positive.")
            return hard_cutoff_kernel(radius)
        if ktype == "gaussian":
            sigma = float(kernel_spec["sigma"])
            if sigma <= 0:
                raise ValueError("Gaussian sigma must be positive.")
            return gaussian_kernel(sigma)
        raise ValueError("Unknown kernel spec; supported types: 'hard', 'gaussian'.")
    try:
        radius = float(kernel_spec)
    except Exception as exc:
        raise TypeError("kernel must be a callable, mapping, or numeric radius.") from exc
    if radius <= 0:
        raise ValueError("Cutoff radius must be positive.")
    return hard_cutoff_kernel(radius)


class CorrelationEnsembleAccumulator:
    """
    Accumulates correlation batches onto a grid in relative-position space with configurable binning.

    Binning is controlled via `kernel`: pass a numeric radius for hard cutoff, a mapping like
    {"type": "gaussian", "sigma": ...}, or any callable kernel(distance_array)->weights.
    Values to accumulate come from `value_fn`; weights per-row can be provided via `weight_fn`.
    """

    def __init__(
        self,
        grid: RegularGridSpec | np.ndarray,
        *,
        kernel: Any = 1.0,
        distance_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = euclidean_distance,
        value_fn: ValueFn | None = None,
        weight_fn: WeightFn | None = None,
        track_counts: bool = True,
        value_dim: int | None = None,
    ):
        if isinstance(grid, RegularGridSpec):
            self._grid = RegularGridNeighborFinder(grid)
            centers = self._grid.centers
        else:
            centers = np.asarray(grid, dtype=float)
            if centers.ndim != 2:
                raise ValueError("grid must be a 2D array of grid point centers (M, d).")
            self._grid = None
        if centers.shape[0] == 0:
            raise ValueError("grid must contain at least one center.")

        self.centers = centers
        self.dim = int(centers.shape[1])
        self.kernel_fn = _normalize_kernel(kernel)
        self.distance_fn = distance_fn
        self.value_fn = value_fn or (lambda rel, tracer, source, meta_row: tracer)
        self.weight_fn = weight_fn or (lambda rel, tracer, source, meta_row: 1.0)
        self.track_counts = bool(track_counts)
        self.value_dim: int | None = int(value_dim) if value_dim is not None else None

        M = centers.shape[0]
        self.sum_v: np.ndarray | None = (
            np.zeros((M, self.value_dim), dtype=float) if self.value_dim is not None else None
        )
        self.sum_w: np.ndarray | None = np.zeros(M, dtype=float) if self.value_dim is not None else None
        self.counts: np.ndarray | None = (
            np.zeros(M, dtype=int) if (self.value_dim is not None and self.track_counts) else None
        )
        self.total_pairs = 0

    def _ensure_arrays(self, value_vec: np.ndarray) -> None:
        if value_vec.ndim != 1:
            raise ValueError("value_fn must return a 1D array.")
        if self.value_dim is None:
            self.value_dim = int(value_vec.shape[0])
            M = self.centers.shape[0]
            self.sum_v = np.zeros((M, self.value_dim), dtype=float)
            self.sum_w = np.zeros(M, dtype=float)
            self.counts = np.zeros(M, dtype=int) if self.track_counts else None
        elif value_vec.shape[0] != self.value_dim:
            raise ValueError("value_fn output dimension does not match existing accumulator.")

    def add(self, batch: CorrelationBatch) -> None:
        if batch.position_dim != self.dim:
            raise ValueError("Batch relative_positions dimensionality does not match grid dimension.")
        if self.value_dim is None and batch.n_pairs == 0:
            return

        for i in range(batch.n_pairs):
            rel = batch.relative_positions[i]
            tracer = batch.tracer_motion[i]
            source = batch.source_motion[i]
            meta_row = batch.meta.iloc[i]

            value_vec = np.asarray(self.value_fn(rel, tracer, source, meta_row), dtype=float)
            self._ensure_arrays(value_vec)
            if self.sum_v is None or self.sum_w is None:
                # Should not happen, but guard for type checkers
                continue

            weights = np.asarray(self.kernel_fn(self.distance_fn(rel, self.centers)), dtype=float)
            if weights.shape != (self.centers.shape[0],):
                raise ValueError("kernel must return weights with shape (n_grid_points,).")

            row_weight = float(self.weight_fn(rel, tracer, source, meta_row))
            if row_weight == 0.0:
                self.total_pairs += 1
                continue

            weights = weights * row_weight
            mask = weights > 0
            if not np.any(mask):
                self.total_pairs += 1
                continue

            idxs = np.nonzero(mask)[0]
            w_masked = weights[idxs]

            np.add.at(self.sum_w, idxs, w_masked)
            for j in range(self.value_dim):
                np.add.at(self.sum_v[:, j], idxs, w_masked * value_vec[j])
            if self.counts is not None:
                np.add.at(self.counts, idxs, 1)

            self.total_pairs += 1

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if self.sum_v is None or self.sum_w is None or self.value_dim is None:
            raise ValueError("No data accumulated; call add() first.")
        mean = np.full_like(self.sum_v, np.nan)
        mask = self.sum_w > 0
        mean[mask] = self.sum_v[mask] / self.sum_w[mask, None]
        return mean, self.sum_w, self.counts

    def merge(self, other: "CorrelationEnsembleAccumulator") -> None:
        if self.centers.shape != other.centers.shape or self.dim != other.dim:
            raise ValueError("Grid centers do not match; cannot merge.")
        if self.value_dim != other.value_dim:
            raise ValueError("value_dim does not match; cannot merge.")
        if self.sum_v is None or self.sum_w is None or other.sum_v is None or other.sum_w is None:
            raise ValueError("Both accumulators must be initialized before merge.")

        self.sum_v += other.sum_v
        self.sum_w += other.sum_w
        if self.counts is not None and other.counts is not None:
            self.counts += other.counts
        self.total_pairs += other.total_pairs


def distance_threshold_pair_filter(max_distance: float) -> PairFilter:
    """
    Convenience pair filter that keeps only pairs with Euclidean distance <= max_distance.
    """

    def _filter(rel_positions: np.ndarray, _tracer_motion, _source_motion, _meta_df) -> np.ndarray:
        return np.linalg.norm(rel_positions, axis=1) <= max_distance

    return _filter
