"""
Two-point microrheology utilities adapted from classic Crocker/Mason code.

References
----------
- J. C. Crocker et al., "Two-point microrheology of inhomogeneous soft materials,"
  Phys. Rev. Lett. 85, 888 (2000).
- T. G. Mason and D. A. Weitz, "Optical measurements of frequency-dependent
  linear viscoelastic moduli of complex fluids," Phys. Rev. Lett. 74, 1250 (1995).
- Original MATLAB routines (`twopoint.m`, `msdd.m`, `calc_G.m`) in
  ``matlabScripts/Two_point`` (this module reimplements their core ideas in Python).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.special import gamma

from trajkit.traj.core import TrajectorySet


@dataclass
class TwoPointCorrelation:
    """Container for two-point displacement correlations."""

    dt: np.ndarray  # time lags (seconds)
    r: np.ndarray  # radial bin centers
    longitudinal: np.ndarray  # shape (n_dt, n_r), <Δr_i·rhat Δr_j·rhat>
    transverse: np.ndarray  # shape (n_dt, n_r), transverse projection
    counts: np.ndarray  # pairs contributing to each bin
    dim: int


def _trajectoryset_to_dataframe(ts: TrajectorySet) -> pd.DataFrame:
    """Flatten a TrajectorySet into a per-frame DataFrame with id, t, positions."""
    records = []
    default_names = ["x", "y", "z"]
    for tr in ts.trajectories.values():
        n, d = tr.x.shape
        rec = {
            "id": np.repeat(tr.track_id, n),
            "t": tr.t if tr.t is not None else tr.frame / float(tr.frame_rate_hz),
        }
        for j in range(d):
            name = default_names[j] if j < len(default_names) else f"x{j}"
            rec[name] = tr.x[:, j]
        records.append(pd.DataFrame(rec))
    if not records:
        raise ValueError("TrajectorySet is empty.")
    return pd.concat(records, ignore_index=True)


def compute_two_point_correlation(
    data: Union[pd.DataFrame, TrajectorySet],
    *,
    track_id_col: str = "id",
    time_col: str = "t",
    position_cols: Sequence[str] = ("x", "y"),
    dt_values: Optional[Iterable[int]] = None,
    max_dt: Optional[int] = None,
    r_min: float = 0.5,
    r_max: float = 20.0,
    n_r_bins: int = 12,
    clip_to_shared_frames: bool = True,
) -> TwoPointCorrelation:
    """
    Compute two-point displacement correlations in the spirit of ``twopoint.m``.

    Parameters
    ----------
    data:
        Per-frame DataFrame or TrajectorySet containing track ids, time/frame, and positions.
    track_id_col:
        Column holding unique track ids (ignored when ``data`` is a TrajectorySet).
    time_col:
        Column holding timestamps (seconds) used to compute lags.
    position_cols:
        Ordered coordinate columns (2D or 3D).
    dt_values:
        Iterable of integer frame lags to evaluate. If None, uses logarithmic lags ~10/decade.
    max_dt:
        Maximum lag (in frames). Required when ``dt_values`` is None.
    r_min, r_max, n_r_bins:
        Radial bin edges (log-spaced) for pair separations.
    clip_to_shared_frames:
        If True, only frames where both tracks are present at t and t+dt are used.

    Returns
    -------
    TwoPointCorrelation
        Correlations projected into longitudinal and transverse components relative to the
        inter-particle separation vector.
    """
    if isinstance(data, TrajectorySet):
        df = _trajectoryset_to_dataframe(data)
        track_id_col = "id"
        time_col = "t"
    else:
        df = data.copy()

    dim = len(position_cols)
    if dim < 2:
        raise ValueError("At least 2D coordinates are required for two-point microrheology.")

    if dt_values is None:
        if max_dt is None:
            raise ValueError("Provide max_dt (frames) or explicit dt_values.")
        dt_values = np.unique(np.round(1.15 ** np.arange(0, 100)).astype(int))
        dt_values = dt_values[dt_values <= max_dt]
    dt_values = np.asarray(list(dt_values), dtype=int)

    # log-spaced radial bins
    r_edges = np.logspace(np.log10(r_min), np.log10(r_max), n_r_bins + 1)
    r_centers = np.sqrt(r_edges[:-1] * r_edges[1:])

    # group by time for quick access
    frames = df.groupby(time_col)

    n_dt = len(dt_values)
    n_r = len(r_centers)
    long_acc = np.zeros((n_dt, n_r), dtype=float)
    trans_acc = np.zeros((n_dt, n_r), dtype=float)
    counts = np.zeros((n_dt, n_r), dtype=float)

    # precompute a map from time -> df for speed
    frame_map = {t: g for t, g in frames}
    times = np.array(sorted(frame_map.keys()))
    time_to_idx = {t: i for i, t in enumerate(times)}

    # Assume roughly uniform spacing; derive dt seconds from differences
    if len(times) < 2:
        raise ValueError("Not enough frames to compute displacements.")
    dt_seconds = np.median(np.diff(times))

    def _accumulate_pairs(pos0: np.ndarray, disp: np.ndarray, k: int):
        # Vectorized accumulation over all unordered pairs in a frame
        n_tracks = pos0.shape[0]
        if n_tracks < 2:
            return
        # pairwise separations
        r_vec = pos0[:, None, :] - pos0[None, :, :]  # (n, n, dim)
        dist = np.linalg.norm(r_vec, axis=2)
        iu = np.triu_indices(n_tracks, k=1)
        dist_u = dist[iu]
        mask = (dist_u >= r_min) & (dist_u < r_max)
        if not np.any(mask):
            return
        dist_u = dist_u[mask]
        r_vec_u = r_vec[iu][mask]

        # unit vectors
        r_hat = r_vec_u / dist_u[:, None]

        di = disp[iu[0][mask]]
        dj = disp[iu[1][mask]]

        proj_i = np.einsum("ij,ij->i", di, r_hat)
        proj_j = np.einsum("ij,ij->i", dj, r_hat)
        longitudinal = proj_i * proj_j
        tensor = np.einsum("ij,ij->i", di, dj)
        transverse = (tensor - longitudinal) / (dim - 1)

        # bin and accumulate
        bins = np.searchsorted(r_edges, dist_u) - 1
        valid = (bins >= 0) & (bins < n_r)
        if not np.any(valid):
            return
        bins = bins[valid]
        long_vals = longitudinal[valid]
        trans_vals = transverse[valid]

        long_acc[k] += np.bincount(bins, weights=long_vals, minlength=n_r)
        trans_acc[k] += np.bincount(bins, weights=trans_vals, minlength=n_r)
        counts[k] += np.bincount(bins, minlength=n_r)

    # iterate over dt lags
    for k, dt in enumerate(dt_values):
        lag = dt * dt_seconds
        if clip_to_shared_frames:
            mask_times = np.isin(times + lag, times)
            valid_times = times[mask_times]
        else:
            valid_times = times

        for t0 in valid_times:
            t1 = t0 + lag
            if t1 not in time_to_idx:
                continue
            df0 = frame_map[t0]
            df1 = frame_map[t1]
            merged = pd.merge(
                df0[[track_id_col, *position_cols]],
                df1[[track_id_col, *position_cols]],
                on=track_id_col,
                suffixes=("_0", "_1"),
            )
            if merged.shape[0] < 2:
                continue
            pos0 = merged[[f"{c}_0" for c in position_cols]].to_numpy()
            pos1 = merged[[f"{c}_1" for c in position_cols]].to_numpy()
            disp = pos1 - pos0
            _accumulate_pairs(pos0, disp, k)

    # avoid division by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        long_mean = np.where(counts > 0, long_acc / counts, np.nan)
        trans_mean = np.where(counts > 0, trans_acc / counts, np.nan)

    return TwoPointCorrelation(
        dt=dt_values * dt_seconds,
        r=r_centers,
        longitudinal=long_mean,
        transverse=trans_mean,
        counts=counts,
        dim=dim,
    )


@dataclass
class MSDFromTwoPoint:
    dt: np.ndarray  # seconds
    msd_longitudinal: np.ndarray
    msd_transverse: np.ndarray
    floral_longitudinal: np.ndarray
    floral_transverse: np.ndarray


def distinct_msd_from_two_point(
    corr: TwoPointCorrelation,
    *,
    r_min: float,
    r_max: float,
    probe_radius: float,
    use_linear_fit: bool = False,
) -> MSDFromTwoPoint:
    """
    Convert two-point correlations into one-point-like MSDs (msdd.m analogue).

    Parameters
    ----------
    corr:
        Two-point correlation output.
    r_min, r_max:
        Radial window (same units as positions) to average over.
    probe_radius:
        Probe radius 'a' (microns) used in the original formulas.
    use_linear_fit:
        If True, fit r*correlation vs r to a line and use the intercept at r=0.
        If False, use a simple average over the selected r-range.
    """
    r = corr.r
    mask_r = (r > r_min) & (r < r_max)
    if not mask_r.any():
        raise ValueError("No radial bins within the requested range.")

    # msd1 output columns: dt, L, T, (optional fits), err
    dt = corr.dt
    msd_L = np.full_like(dt, np.nan, dtype=float)
    msd_T = np.full_like(dt, np.nan, dtype=float)
    floral_L = np.zeros_like(dt)
    floral_T = np.zeros_like(dt)

    for i, t in enumerate(dt):
        L = corr.longitudinal[i, mask_r]
        T = corr.transverse[i, mask_r]
        r_window = r[mask_r]
        if np.all(np.isnan(L)) or np.all(np.isnan(T)):
            continue
        if use_linear_fit:
            # Fit r*L vs r and r*T vs r; intercept at r=0
            coeff_L = np.polyfit(r_window, r_window * L, 1)
            coeff_T = np.polyfit(r_window, r_window * T, 1)
            msd_L[i] = coeff_L[1] * (2.0 / (2 * probe_radius))
            msd_T[i] = coeff_T[1] * (4.0 / (2 * probe_radius))
            floral_L[i] = coeff_L[0] * (2.0 / (2 * probe_radius)) * r_window.mean()
            floral_T[i] = coeff_T[0] * (4.0 / (2 * probe_radius)) * r_window.mean()
        else:
            msd_L[i] = (2.0 / (2 * probe_radius)) * np.nanmean(r_window * L)
            msd_T[i] = (4.0 / (2 * probe_radius)) * np.nanmean(r_window * T)

    return MSDFromTwoPoint(
        dt=dt,
        msd_longitudinal=msd_L,
        msd_transverse=msd_T,
        floral_longitudinal=floral_L,
        floral_transverse=floral_T,
    )


def _logderive(x: np.ndarray, y: np.ndarray, window: int = 5, polyorder: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """Local log-derivative using Savitzky-Golay smoothing."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lx = np.log(x)
    ly = np.log(y)
    n = len(x)
    w = max(polyorder + 2, window if window % 2 == 1 else window + 1)
    w = min(w, n if n % 2 == 1 else n - 1)
    ly_smooth = savgol_filter(ly, window_length=max(3, w), polyorder=min(polyorder, w - 1))
    d1 = np.gradient(ly_smooth, lx)
    d2 = np.gradient(d1, lx)
    m = np.exp(ly_smooth)
    return m, d1, d2


@dataclass
class ShearModulusResult:
    omega: np.ndarray
    Gs: np.ndarray
    Gp: np.ndarray
    Gpp: np.ndarray
    d1: np.ndarray
    d2: np.ndarray
    d1_G: np.ndarray
    d2_G: np.ndarray


def compute_shear_modulus_from_msd(
    tau: np.ndarray,
    msd: np.ndarray,
    *,
    probe_radius_microns: float,
    dim: int,
    temperature_K: float = 298.0,
    clip: float = 0.03,
    smoothing_window: int = 7,
    polyorder: int = 2,
) -> ShearModulusResult:
    """
    Mason-Weitz viscoelastic moduli from MSD (calc_G.m analogue).

    Parameters
    ----------
    tau:
        Time lags (seconds).
    msd:
        Mean-squared displacement (same length as ``tau``), in microns^2.
    probe_radius_microns:
        Probe radius 'a' in microns.
    dim:
        Spatial dimensionality (2 or 3).
    temperature_K:
        Temperature in Kelvin.
    clip:
        Fraction of G(s) below which G'(w)/G''(w) are zeroed (noise guard).
    smoothing_window, polyorder:
        Savitzky-Golay parameters for log-derivatives.
    """
    tau = np.asarray(tau, dtype=float)
    msd = np.asarray(msd, dtype=float)
    if tau.shape != msd.shape:
        raise ValueError("tau and msd must have the same shape.")

    kB = 1.38065e-23
    a_m = probe_radius_microns * 1e-6
    omega = 1.0 / tau
    msd_m = msd * 1e-12
    C = dim * kB * temperature_K / (3 * np.pi * a_m)
    foo = (np.pi / 2) - 1.0

    m, d, dd = _logderive(tau, msd_m, window=smoothing_window, polyorder=polyorder)
    Gs = C / ((m * gamma(1 + d)) * (1 + (dd / 2)))

    g, da, dda = _logderive(omega, Gs, window=smoothing_window, polyorder=polyorder)
    Gp = g * (1.0 / (1 + dda)) * (np.cos((np.pi / 2) * da) - foo * da * dda)
    Gpp = g * (1.0 / (1 + dda)) * (np.sin((np.pi / 2) * da) - foo * (1 - da) * dda)

    # clip noisy points
    Gp = np.where(Gp < Gs * clip, 0.0, Gp)
    Gpp = np.where(Gpp < Gs * clip, 0.0, Gpp)

    return ShearModulusResult(
        omega=omega,
        Gs=Gs,
        Gp=Gp,
        Gpp=Gpp,
        d1=d,
        d2=dd,
        d1_G=da,
        d2_G=dda,
    )


def save_two_point_correlation(corr: TwoPointCorrelation, path: str | Path) -> None:
    """Persist a TwoPointCorrelation to disk (npz)."""
    path = Path(path)
    np.savez_compressed(
        path,
        dt=corr.dt,
        r=corr.r,
        longitudinal=corr.longitudinal,
        transverse=corr.transverse,
        counts=corr.counts,
        dim=np.array(corr.dim, dtype=int),
    )


def load_two_point_correlation(path: str | Path) -> TwoPointCorrelation:
    """Load a TwoPointCorrelation saved by save_two_point_correlation."""
    with np.load(path) as data:
        return TwoPointCorrelation(
            dt=data["dt"],
            r=data["r"],
            longitudinal=data["longitudinal"],
            transverse=data["transverse"],
            counts=data["counts"],
            dim=int(np.array(data["dim"]).item()),
        )
