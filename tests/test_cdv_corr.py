import numpy as np
import pandas as pd

from trajkit.cdv import (
    CorrelationBatch,
    CorrelationEnsembleAccumulator,
    RegularGridSpec,
    correlation_batch,
    distance_threshold_pair_filter,
)


def test_correlation_pairs_sources_and_tracers_by_frame():
    source_df = pd.DataFrame(
        {
            "frame": [0, 1],
            "sx": [0.0, 10.0],
            "sy": [0.0, 0.0],
            "u": [1.0, 2.0],
            "v": [0.0, -1.0],
        }
    )
    tracer_df = pd.DataFrame(
        {
            "frame": [0, 0, 1],
            "x": [1.0, 2.0, 9.5],
            "y": [0.0, 1.0, 0.5],
            "dx": [0.1, 0.2, 0.3],
            "dy": [0.0, -0.1, 0.2],
        }
    )

    batch, ensemble = correlation_batch(
        source_df,
        tracer_df,
        source_position_cols=["sx", "sy"],  # list to ensure list/tuple both work
        tracer_position_cols=["x", "y"],
        source_motion_cols=["u", "v"],
        tracer_motion_cols=["dx", "dy"],
    )

    assert ensemble is None
    assert batch.n_pairs == 3
    np.testing.assert_allclose(
        batch.relative_positions,
        [
            [1.0, 0.0],
            [2.0, 1.0],
            [-0.5, 0.5],
        ],
    )
    np.testing.assert_allclose(batch.tracer_motion, [[0.1, 0.0], [0.2, -0.1], [0.3, 0.2]])
    np.testing.assert_allclose(batch.source_motion, [[1.0, 0.0], [1.0, 0.0], [2.0, -1.0]])
    assert list(batch.meta["frame"]) == [0, 0, 1]
    assert list(batch.meta["source_index"]) == [0, 0, 1]
    assert list(batch.meta["tracer_index"]) == [0, 1, 2]


def test_correlation_pair_filter_can_drop_pairs_by_distance():
    source_df = pd.DataFrame(
        {
            "frame": [0, 1],
            "sx": [0.0, 10.0],
            "sy": [0.0, 0.0],
            "u": [1.0, 2.0],
            "v": [0.0, -1.0],
        }
    )
    tracer_df = pd.DataFrame(
        {
            "frame": [0, 0, 1],
            "x": [1.0, 2.0, 9.5],
            "y": [0.0, 1.0, 0.5],
            "dx": [0.1, 0.2, 0.3],
            "dy": [0.0, -0.1, 0.2],
        }
    )

    batch, _ = correlation_batch(
        source_df,
        tracer_df,
        source_position_cols=["sx", "sy"],
        tracer_position_cols=["x", "y"],
        source_motion_cols=["u", "v"],
        tracer_motion_cols=["dx", "dy"],
        pair_filter=distance_threshold_pair_filter(1.1),
    )

    assert batch.n_pairs == 2
    np.testing.assert_allclose(batch.relative_positions, [[1.0, 0.0], [-0.5, 0.5]])
    np.testing.assert_allclose(batch.tracer_motion, [[0.1, 0.0], [0.3, 0.2]])
    np.testing.assert_allclose(batch.source_motion, [[1.0, 0.0], [2.0, -1.0]])
    assert list(batch.meta["frame"]) == [0, 1]
    assert list(batch.meta["source_index"]) == [0, 1]
    assert list(batch.meta["tracer_index"]) == [0, 2]
    np.testing.assert_allclose(batch.tracer_motion, [[0.1, 0.0], [0.3, 0.2]])
    np.testing.assert_allclose(batch.source_motion, [[1.0, 0.0], [2.0, -1.0]])
    assert list(batch.meta["frame"]) == [0, 1]
    assert list(batch.meta["source_index"]) == [0, 1]
    assert list(batch.meta["tracer_index"]) == [0, 2]


def test_correlation_pair_filter_accepts_args_and_kwargs():
    source_df = pd.DataFrame(
        {
            "frame": [0, 1],
            "sx": [0.0, 10.0],
            "sy": [0.0, 0.0],
            "u": [1.0, 2.0],
            "v": [0.0, -1.0],
        }
    )
    tracer_df = pd.DataFrame(
        {
            "frame": [0, 0, 1],
            "x": [1.0, 2.0, 9.5],
            "y": [0.0, 1.0, 0.5],
            "dx": [0.1, 0.2, 0.3],
            "dy": [0.0, -0.1, 0.2],
        }
    )

    def _filter(rel_positions, _tracer_motion, _source_motion, _meta_df, max_distance, axis):
        return np.linalg.norm(rel_positions, axis=axis) <= max_distance

    batch, _ = correlation_batch(
        source_df,
        tracer_df,
        source_position_cols=["sx", "sy"],
        tracer_position_cols=["x", "y"],
        source_motion_cols=["u", "v"],
        tracer_motion_cols=["dx", "dy"],
        pair_filter=_filter,
        pair_filter_args=(1.1,),
        pair_filter_kwargs={"axis": 1},
    )

    assert batch.n_pairs == 2
    np.testing.assert_allclose(batch.relative_positions, [[1.0, 0.0], [-0.5, 0.5]])


def test_correlation_respects_empty_overlap():
    source_df = pd.DataFrame({"frame": [0], "x": [0.0], "u": [1.0]})
    tracer_df = pd.DataFrame({"frame": [1], "x": [1.0], "dx": [0.1]})

    batch, _ = correlation_batch(
        source_df,
        tracer_df,
        source_position_cols=("x",),
        tracer_position_cols=("x",),
        source_motion_cols=("u",),
        tracer_motion_cols=("dx",),
    )

    assert batch.n_pairs == 0
    assert batch.relative_positions.shape == (0, 1)
    assert batch.tracer_motion.shape == (0, 1)
    assert batch.source_motion.shape == (0, 1)
    assert batch.meta.empty


def test_correlation_validates_position_dimensions_match():
    source_df = pd.DataFrame({"frame": [0], "x": [0.0], "y": [0.0], "u": [1.0]})
    tracer_df = pd.DataFrame({"frame": [0], "x": [1.0], "dx": [0.1]})

    try:
        correlation_batch(
            source_df,
            tracer_df,
            source_position_cols=("x", "y"),
            tracer_position_cols=("x",),
            source_motion_cols=("u",),
            tracer_motion_cols=("dx",),
        )
    except ValueError as exc:
        assert "same length" in str(exc)
    else:
        raise AssertionError("Expected ValueError for mismatched position dimensions.")


def test_correlation_calls_ensemble_fn_when_provided():
    source_df = pd.DataFrame({"frame": [0], "x": [0.0], "u": [1.0]})
    tracer_df = pd.DataFrame({"frame": [0], "x": [1.0], "dx": [0.1]})
    called_with = {}

    def _ensemble_fn(batch):
        called_with["n_pairs"] = batch.n_pairs
        return batch.tracer_motion.sum()

    batch, ensemble = correlation_batch(
        source_df,
        tracer_df,
        source_position_cols=("x",),
        tracer_position_cols=("x",),
        source_motion_cols=("u",),
        tracer_motion_cols=("dx",),
        ensemble_fn=_ensemble_fn,
    )

    assert batch.n_pairs == 1
    assert ensemble == 0.1
    assert called_with["n_pairs"] == 1


def test_correlation_pair_filter_validates_mask_length():
    source_df = pd.DataFrame({"frame": [0], "x": [0.0], "u": [1.0]})
    tracer_df = pd.DataFrame({"frame": [0, 0], "x": [1.0, 2.0], "dx": [0.1, 0.2]})

    def _bad_filter(rel_positions, _tracer_motion, _source_motion, _meta_df):
        # Wrong length on purpose.
        return np.array([True, False])

    try:
        correlation_batch(
            source_df,
            tracer_df,
            source_position_cols=("x",),
            tracer_position_cols=("x",),
            source_motion_cols=("u",),
            tracer_motion_cols=("dx",),
            pair_filter=_bad_filter,
        )
    except ValueError as exc:
        assert "mask" in str(exc)
    else:
        raise AssertionError("Expected ValueError for bad pair_filter mask length.")


def test_correlation_batch_to_dataframe_defaults_and_order():
    source_df = pd.DataFrame({"frame": [0], "x": [0.0], "y": [1.0], "u": [1.0], "v": [0.5]})
    tracer_df = pd.DataFrame({"frame": [0], "x": [0.5], "y": [2.0], "dx": [0.2], "dy": [0.1]})
    batch, _ = correlation_batch(
        source_df,
        tracer_df,
        source_position_cols=["x", "y"],
        tracer_position_cols=["x", "y"],
        source_motion_cols=["u", "v"],
        tracer_motion_cols=["dx", "dy"],
    )

    df = batch.to_dataframe()
    assert list(df.columns) == [
        "frame",
        "source_index",
        "tracer_index",
        "rel_pos_0",
        "rel_pos_1",
        "tracer_motion_0",
        "tracer_motion_1",
        "source_motion_0",
        "source_motion_1",
    ]
    np.testing.assert_allclose(df[["rel_pos_0", "rel_pos_1"]].to_numpy(), [[0.5, 1.0]])
    np.testing.assert_allclose(df[["tracer_motion_0", "tracer_motion_1"]].to_numpy(), [[0.2, 0.1]])
    np.testing.assert_allclose(df[["source_motion_0", "source_motion_1"]].to_numpy(), [[1.0, 0.5]])


def test_correlation_batch_to_dataframe_with_custom_columns():
    source_df = pd.DataFrame({"frame": [0], "x": [0.0], "u": [1.0]})
    tracer_df = pd.DataFrame({"frame": [0], "x": [1.0], "dx": [0.2]})
    batch, _ = correlation_batch(
        source_df,
        tracer_df,
        source_position_cols=["x"],
        tracer_position_cols=["x"],
        source_motion_cols=["u"],
        tracer_motion_cols=["dx"],
    )

    df = batch.to_dataframe(
        relative_position_cols=["dr"],
        tracer_motion_cols=["dx"],
        source_motion_cols=["u"],
    )
    assert list(df.columns) == ["frame", "source_index", "tracer_index", "dr", "dx", "u"]
    np.testing.assert_allclose(df[["dr"]].to_numpy(), [[1.0]])


def test_rotate_to_source_x_aligns_vectors_and_tracks_angle():
    source_df = pd.DataFrame({"frame": [0], "x": [0.0], "y": [0.0], "u": [0.0], "v": [1.0]})
    tracer_df = pd.DataFrame({"frame": [0], "x": [1.0], "y": [0.0], "dx": [1.0], "dy": [0.0]})
    batch, _ = correlation_batch(
        source_df,
        tracer_df,
        source_position_cols=["x", "y"],
        tracer_position_cols=["x", "y"],
        source_motion_cols=["u", "v"],
        tracer_motion_cols=["dx", "dy"],
    )

    rotated = batch.rotate_to_source_x()
    # Source motion should point along +x with same magnitude (1.0)
    np.testing.assert_allclose(rotated.source_motion, [[1.0, 0.0]], atol=1e-12)
    # Relative position [1,0] rotated by -pi/2 becomes [0,-1]
    np.testing.assert_allclose(rotated.relative_positions, [[0.0, -1.0]], atol=1e-12)
    # Tracer motion rotates similarly
    np.testing.assert_allclose(rotated.tracer_motion, [[0.0, -1.0]], atol=1e-12)
    assert "rotation_angle_rad" in rotated.meta
    np.testing.assert_allclose(rotated.meta["rotation_angle_rad"].to_numpy(), [np.pi / 2], atol=1e-12)


def test_rotate_to_source_x_handles_zero_motion_without_crash():
    source_df = pd.DataFrame({"frame": [0], "x": [0.0], "y": [0.0], "u": [0.0], "v": [0.0]})
    tracer_df = pd.DataFrame({"frame": [0], "x": [1.0], "y": [0.0], "dx": [1.0], "dy": [0.0]})
    batch, _ = correlation_batch(
        source_df,
        tracer_df,
        source_position_cols=["x", "y"],
        tracer_position_cols=["x", "y"],
        source_motion_cols=["u", "v"],
        tracer_motion_cols=["dx", "dy"],
    )

    rotated = batch.rotate_to_source_x()
    # Without a defined direction, data stays unchanged
    np.testing.assert_allclose(rotated.relative_positions, batch.relative_positions)
    np.testing.assert_allclose(rotated.tracer_motion, batch.tracer_motion)
    np.testing.assert_allclose(rotated.source_motion, batch.source_motion)
    np.testing.assert_allclose(rotated.meta["rotation_angle_rad"].to_numpy(), [0.0])


def test_rotate_to_source_x_rejects_scalar_motion():
    source_df = pd.DataFrame({"frame": [0], "x": [0.0], "u": [1.0]})
    tracer_df = pd.DataFrame({"frame": [0], "x": [1.0], "dx": [1.0]})
    batch, _ = correlation_batch(
        source_df,
        tracer_df,
        source_position_cols=["x"],
        tracer_position_cols=["x"],
        source_motion_cols=["u"],
        tracer_motion_cols=["dx"],
    )

    try:
        batch.rotate_to_source_x()
    except ValueError as exc:
        assert "scalar source motion" in str(exc)
    else:
        raise AssertionError("Expected ValueError for scalar source motion.")


def test_correlation_ensemble_accumulator_accumulates_with_weights():
    grid = np.array([[0.0, 0.0], [2.0, 0.0]])
    ensemble = CorrelationEnsembleAccumulator(
        grid,
        kernel=0.6,  # hard cutoff radius
        weight_fn=lambda rel, tracer, source, meta_row: np.linalg.norm(source),
    )

    rel = np.array([[0.1, 0.0], [2.1, 0.0]])
    tracer = np.array([[1.0, 0.0], [3.0, 0.0]])
    source = np.array([[2.0, 0.0], [1.0, 0.0]])
    meta = pd.DataFrame({"frame": [0, 0], "source_index": [0, 1], "tracer_index": [0, 1]})
    batch = CorrelationBatch(rel, tracer, source, meta)

    ensemble.add(batch)
    mean, sum_w, counts = ensemble.finalize()

    assert ensemble.total_pairs == 2
    np.testing.assert_allclose(sum_w, [2.0, 1.0])
    np.testing.assert_allclose(mean, [[1.0, 0.0], [3.0, 0.0]])
    assert counts is not None
    np.testing.assert_array_equal(counts, [1, 1])


def test_correlation_ensemble_accepts_regular_grid_spec_and_custom_value_fn():
    grid_spec = RegularGridSpec(grid_min=[0.0, 0.0], grid_max=[0.0, 0.0], cell_size=[1.0, 1.0])
    ensemble = CorrelationEnsembleAccumulator(
        grid_spec,
        kernel={"type": "gaussian", "sigma": 0.5},
        value_fn=lambda rel, tracer, source, meta_row: np.array([rel[0]]),
    )

    rel = np.array([[0.0, 0.0]])
    tracer = np.array([[0.5, 0.5]])
    source = np.array([[0.1, 0.0]])
    meta = pd.DataFrame({"frame": [0], "source_index": [0], "tracer_index": [0]})
    batch = CorrelationBatch(rel, tracer, source, meta)

    ensemble.add(batch)
    mean, sum_w, counts = ensemble.finalize()

    assert mean.shape == (1, 1)
    assert sum_w.shape == (1,)
    assert counts is None or counts.shape == (1,)


def test_correlation_ensemble_rejects_dim_mismatch():
    grid = np.array([[0.0, 0.0]])
    ensemble = CorrelationEnsembleAccumulator(grid)

    rel = np.array([[0.0]])  # 1D, should fail against 2D grid
    tracer = np.array([[1.0, 0.0]])
    source = np.array([[0.0, 0.0]])
    meta = pd.DataFrame({"frame": [0], "source_index": [0], "tracer_index": [0]})
    batch = CorrelationBatch(rel, tracer, source, meta)

    try:
        ensemble.add(batch)
    except ValueError as exc:
        assert "dimensionality" in str(exc)
    else:
        raise AssertionError("Expected ValueError for dimensionality mismatch.")
