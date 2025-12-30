import numpy as np

from trajkit import Trajectory, TrajectorySet
from trajkit.stats import msd, msd_trajectory_set


def make_traj(xy: np.ndarray, track_id: str) -> Trajectory:
    frames = np.arange(len(xy), dtype=int)
    return Trajectory(track_id=track_id, x=xy, frame=frames, frame_rate_hz=1.0)


def test_msd_single_basic():
    xy = np.array([[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]])
    tr = make_traj(xy, "a")

    df = msd(tr, lags=[1, 2], per_dim=True, return_sd=True)
    df = df.set_index("lag")

    # lag 1: displacements [1, 2] -> mean sq = (1^2 + 2^2)/2 = 2.5
    assert df.loc[1, "n"] == 2
    assert np.isclose(df.loc[1, "msd"], 2.5)
    assert np.isclose(df.loc[1, "msd_dim0"], 2.5)
    assert np.isclose(df.loc[1, "sd_dim0"], 1.5)  # mean displacement

    # lag 2: displacement [3] -> sq = 9
    assert df.loc[2, "n"] == 1
    assert np.isclose(df.loc[2, "msd"], 9.0)


def test_msd_interpolate_increases_pairs():
    # Frames are 0, 2, 4 with linear motion along x; interpolation adds frames 1,3.
    frames = np.array([0, 2, 4], dtype=int)
    xy = np.column_stack([frames.astype(float), np.zeros_like(frames, dtype=float)])
    tr = Trajectory(track_id="interp", x=xy, frame=frames, frame_rate_hz=1.0)

    df_no_interp = msd(tr, lags=[1], interpolate=False)
    df_interp = msd(tr, lags=[1], interpolate=True)

    # Without interpolation: displacements 2,2 -> mean sq = 4, n=2
    assert int(df_no_interp.loc[0, "n"]) == 2
    assert np.isclose(df_no_interp.loc[0, "msd"], 4.0)

    # With interpolation: displacements 1,1,1,1 -> mean sq = 1, n=4
    assert int(df_interp.loc[0, "n"]) == 4
    assert np.isclose(df_interp.loc[0, "msd"], 1.0)


def test_msd_trajectory_set_pairwise_vs_track_mean():
    tr_a = make_traj(np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]), "a")
    tr_b = make_traj(np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]]), "b")
    ts = TrajectorySet(dataset_id="demo", trajectories={"a": tr_a, "b": tr_b})

    # Track-mean aggregation (legacy behavior)
    agg_track = msd_trajectory_set(ts, lags=[1], aggregate=True, aggregate_mode="track")
    assert np.isclose(agg_track.loc[0, "msd_mean"], 1.0)
    assert int(agg_track.loc[0, "n_pairs"]) == 4

    # Pair-weighted aggregation
    agg_pair = msd_trajectory_set(ts, lags=[1], aggregate=True, aggregate_mode="pair", per_dim=True)
    assert int(agg_pair.loc[0, "n_pairs"]) == 4
    # Total MSD should match track mean in this symmetric case
    assert np.isclose(agg_pair.loc[0, "msd"], 1.0)
    # Per-dimension components: 0.5 along x and 0.5 along y
    assert np.isclose(agg_pair.loc[0, "msd_dim0"], 0.5)
    assert np.isclose(agg_pair.loc[0, "msd_dim1"], 0.5)
