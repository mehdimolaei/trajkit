import numpy as np
import pandas as pd
import pytest

from trajkit.traj import TrajectorySet


def test_from_dataframe_builds_and_sorts_tracks():
    df = pd.DataFrame(
        {
            "id": ["a", "a", "b", "a", "b"],
            "t": [1.0, 0.0, 0.0, 2.0, 1.0],
            "x": [1.0, 0.0, 0.0, 2.0, 1.0],
            "y": [1.0, 0.0, 1.0, 2.0, 2.0],
            "label": ["alpha", "alpha", "beta", "alpha", "beta"],
            "diameter": [2.0, 2.0, 1.0, 2.0, 1.0],
            "valid": [True, True, False, True, True],
            "area": [0.1, 0.2, 0.3, 0.4, 0.5],
        }
    )

    ts = TrajectorySet.from_dataframe(
        df,
        dataset_id="demo",
        track_id_col="id",
        position_cols=["x", "y"],
        time_col="t",
        label_col="label",
        track_feature_cols=["diameter"],
        frame_feature_cols=["area"],
        valid_col="valid",
        units={"t": "frame", "x": "px"},
    )

    assert ts.units == {"t": "frame", "x": "px"}
    assert set(ts.ids()) == {"a", "b"}

    tr_a = ts.get("a")
    np.testing.assert_allclose(tr_a.time_seconds(), [0.0, 1.0, 2.0])
    np.testing.assert_allclose(tr_a.x, np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]]))
    assert tr_a.label == "alpha"
    assert tr_a.track_features["diameter"] == 2.0
    np.testing.assert_array_equal(tr_a.valid, np.array([True, True, True]))
    np.testing.assert_allclose(tr_a.frame_features["area"], [0.2, 0.1, 0.4])

    tr_b = ts.get("b")
    np.testing.assert_allclose(tr_b.time_seconds(), [0.0, 1.0])
    np.testing.assert_allclose(tr_b.x, np.array([[0.0, 1.0], [1.0, 2.0]]))
    assert tr_b.label == "beta"
    assert tr_b.track_features["diameter"] == 1.0
    np.testing.assert_array_equal(tr_b.valid, np.array([False, True]))
    np.testing.assert_allclose(tr_b.frame_features["area"], [0.3, 0.5])


def test_from_dataframe_requires_frame_rate_with_frame_only():
    df = pd.DataFrame(
        {
            "id": [0, 0],
            "frame": [0, 1],
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
        }
    )

    with pytest.raises(ValueError):
        TrajectorySet.from_dataframe(
            df,
            dataset_id="demo",
            track_id_col="id",
            position_cols=["x", "y"],
            frame_col="frame",
        )


def test_from_dataframe_rejects_non_constant_track_feature():
    df = pd.DataFrame(
        {
            "id": ["a", "a"],
            "t": [0.0, 1.0],
            "x": [0.0, 1.0],
            "y": [0.0, 1.0],
            "diameter": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError):
        TrajectorySet.from_dataframe(
            df,
            dataset_id="demo",
            track_id_col="id",
            position_cols=["x", "y"],
            time_col="t",
            track_feature_cols=["diameter"],
        )
