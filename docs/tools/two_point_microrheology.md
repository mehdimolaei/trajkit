# Two-point microrheology (tool)

Brief walkthrough of correlated displacement microrheology in trajkit, following the classic two-point framework developed by Mason & Weitz (1995, 1996) and Crocker et al. (2000). The method cross-correlates thermal displacements of pairs of tracers to recover bulk viscoelastic response even in heterogeneous media.

## What you'll do
- Fetch a prepared trajectory set from Hugging Face (or build one from CSV).
- Compute two-point correlations with `trajkit.flow.compute_two_point_correlation`.
- Persist/reload results (`.npz`) to avoid recomputation.
- Convert correlations into one-point-like MSDs and viscoelastic moduli.

## Prerequisites
- Python 3.10+ with `trajkit` installed.
- If the dataset is private/gated, set `HF_TOKEN` or run `huggingface-cli login`.

## 1) Load trajectories from Hugging Face (Parquet TrajectorySet)
Use the packaged Parquet bundle to avoid re-building trajectories:

```python
from pathlib import Path
from huggingface_hub import snapshot_download
from trajkit import load_trajectory_set

subpath = "data/parquet/experiment_001_2017-08-16/exp001_t027m_r01um_2017-08-16"

local_root = snapshot_download(
    repo_id="m-aban/air-water",
    repo_type="dataset",
    allow_patterns=[f"{subpath}/*"],
    local_dir="hf_cache",
    local_dir_use_symlinks=False,
)

folder = Path(local_root) / subpath
ts = load_trajectory_set(folder)
```

## 2) Alternative: CSV → TrajectorySet
If you prefer the raw CSV:

```python
import pandas as pd
from huggingface_hub import hf_hub_download
from trajkit import TrajectorySet

csv_path = hf_hub_download(
    repo_id="m-aban/air-water",
    filename="data/csv/experiment_001_2017-08-16/exp001_t000m_r01um_2017-08-16.csv.gz",
    repo_type="dataset",
)

df = pd.read_csv(csv_path)
ts = TrajectorySet.from_dataframe(
    df,
    dataset_id="air-water",
    track_id_col="id",           # change if your column names differ
    position_cols=["x", "y"],
    time_col="t",                # or frame_col="frame", frame_rate_hz=...
)
```

## 3) Compute two-point correlations

```python
import numpy as np
from trajkit.flow import compute_two_point_correlation

dt_values = np.ceil(np.logspace(0, 2, 10)).astype(int)
corr = compute_two_point_correlation(
    ts,
    dt_values=dt_values,   # or set max_dt to auto-generate lags
    r_min=10,
    r_max=500,
    n_r_bins=20,
    clip_to_shared_frames=True,
)
```

```{admonition} Plot placeholder: correlation vs separation
Insert longitudinal/transverse correlation vs `r` (log-log) for representative `dt`.
```

## 4) Persist and reload

```python
from trajkit.flow import save_two_point_correlation, load_two_point_correlation

save_two_point_correlation(corr, "results/two_point_corr.npz")
corr = load_two_point_correlation("results/two_point_corr.npz")
```

## 5) Derive MSD and shear moduli

```python
from trajkit.flow import distinct_msd_from_two_point, compute_shear_modulus_from_msd

msd_2p = distinct_msd_from_two_point(
    corr,
    r_min=10,
    r_max=500,
    probe_radius=1.0,   # microns
    use_linear_fit=False,
)

moduli = compute_shear_modulus_from_msd(
    tau=msd_2p.dt,
    msd=msd_2p.msd_transverse,  # choose longitudinal or transverse
    probe_radius_microns=1.0,
    dim=2,
    temperature_K=298.0,
    clip=0.03,
    smoothing_window=7,
    polyorder=2,
)
```

```{admonition} Plot placeholder: MSD from two-point
Insert MSD (longitudinal/transverse) vs lag time, log-log with power-law guides.
```

```{admonition} Plot placeholder: viscoelastic moduli
Insert G'(ω) and G''(ω) curves derived from the MSD.
```

## Notes and references
- Two-point microrheology cross-correlates tracer displacements to recover bulk response in heterogeneous media, as in Crocker et al. (2000) and Mason & Weitz (1995, 1996).
- If you change coordinate columns (`position_cols`) or timing (`time_col` vs `frame_col`), keep `dt_values` in frame units when using `frame_col`, or in seconds when using `time_col`.
- The `.npz` produced by `save_two_point_correlation` is a compressed NumPy archive; reload with `load_two_point_correlation` to avoid recomputation.
