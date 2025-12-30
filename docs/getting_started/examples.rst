Examples
========

Notebooks
---------

- ``tutorials/first-notebook``
- ``tutorials/cdv-flow-field``
- ``tutorials/msd``

Minimal script outline
----------------------

.. code-block:: python

    import pandas as pd
    from trajkit import TrajectorySet
    from trajkit.stats import msd_trajectory_set

    df = pd.read_csv("your_tracks.csv")  # columns: id, x, y, t (or frame)
    ts = TrajectorySet.from_dataframe(
        df,
        dataset_id="your_dataset",
        track_id_col="id",
        position_cols=["x", "y"],
        time_col="t",
        frame_col="frame",      # omit if not available
        frame_rate_hz=20.0,     # required if only frames
    )
    msd_df = msd_trajectory_set(ts, aggregate=True, aggregate_mode="pair")
