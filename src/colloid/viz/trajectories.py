from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping, Optional, Sequence

import pandas as pd

from colloid.traj import TrajectorySet


def _require_plotly():
    try:
        import plotly.express as px  # noqa: F401
    except ImportError as exc:  # pragma: no cover - thin import guard
        raise ImportError("Install plotly to use colloid.viz (pip install plotly>=5.24)") from exc


def _frame_table(ts: TrajectorySet, track_ids: Optional[Iterable[str]] = None) -> pd.DataFrame:
    ids = list(track_ids) if track_ids is not None else list(ts.ids())
    frames = []
    for tid in ids:
        tr = ts.get(str(tid))
        df = tr.to_frame_dataframe()
        df["track_id"] = df["track_id"].astype(str)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def plot_trajectories(
    ts: TrajectorySet,
    *,
    mode: str = "2d",
    dims: Sequence[int] = (0, 1),
    track_ids: Optional[Iterable[str]] = None,
    time_axis: str = "t",
    color_by: Optional[str] = "track_id",
    symbol_by: Optional[str] = None,
    color_sequence: Optional[Sequence[str]] = None,
    line_kwargs: Optional[Mapping[str, object]] = None,
    marker_kwargs: Optional[Mapping[str, object]] = None,
    connect_points: bool = True,
    title: Optional[str] = None,
    labels: Optional[MutableMapping[str, str]] = None,
    hover_data: Optional[Sequence[str]] = None,
):
    """
    Plot trajectories from a TrajectorySet with flexible styling.

    Args:
        ts: TrajectorySet to visualize.
        mode: "time" for dimension vs time, "2d" or "3d" for spatial plots.
        dims: Which position dimensions to use (e.g., (0,), (0, 1), or (0, 1, 2)).
        track_ids: Optional subset of track_ids to plot.
        time_axis: Column to use for time in \"time\" mode (\"t\" or \"frame\").
        color_by: Column to map colors (e.g., \"track_id\", \"t\", or a frame feature like \"ff_speed\").
        symbol_by: Column to map marker symbols.
        color_sequence: Custom colors to pass to Plotly.
        line_kwargs: Extra styling for lines (e.g., {\"width\": 2}).
        marker_kwargs: Extra styling for markers (e.g., {\"size\": 6}).
        connect_points: If False in spatial modes, draw scatter instead of connected lines.
        title: Figure title.
        labels: Optional label overrides for axes.
        hover_data: Columns to show on hover.
    Returns:
        plotly.graph_objects.Figure
    """
    _require_plotly()
    import plotly.express as px

    df = _frame_table(ts, track_ids)
    if df.empty:
        raise ValueError("No frames available to plot.")

    labels = labels or {}
    labels.setdefault("t", "time (s)")
    labels.setdefault("frame", "frame")
    for j in range(5):
        labels.setdefault(f"x{j}", f"x{j}")

    if color_by and color_by not in df.columns and color_by != "component":
        raise ValueError(f"color_by='{color_by}' not found in frame table columns.")
    if symbol_by and symbol_by not in df.columns and symbol_by != "component":
        raise ValueError(f"symbol_by='{symbol_by}' not found in frame table columns.")

    hover = list(hover_data) if hover_data is not None else [c for c in ["track_id", "t", "frame"] if c in df.columns]

    line_kwargs = dict(line_kwargs or {})
    marker_kwargs = dict(marker_kwargs or {})

    mode = mode.lower()
    if mode == "time":
        if len(dims) < 1:
            raise ValueError("Provide at least one dimension in dims for time plots.")
        y_cols = [f"x{d}" for d in dims]
        long = df.melt(id_vars=["track_id", time_axis], value_vars=y_cols, var_name="component", value_name="position")
        color_col = color_by or "track_id"
        fig = px.line(
            long,
            x=time_axis,
            y="position",
            color=color_col,
            line_group="track_id",
            symbol=symbol_by if symbol_by in long.columns else None,
            hover_data=hover,
            color_discrete_sequence=color_sequence,
            labels=labels,
            title=title,
        )
        if marker_kwargs:
            fig.update_traces(mode="lines+markers", marker=marker_kwargs)
    elif mode == "2d":
        if len(dims) < 2:
            raise ValueError("dims must provide at least two dimensions for 2d mode.")
        xcol, ycol = f"x{dims[0]}", f"x{dims[1]}"
        if connect_points:
            fig = px.line(
                df,
                x=xcol,
                y=ycol,
                color=color_by,
                symbol=symbol_by,
                hover_data=hover,
                color_discrete_sequence=color_sequence,
                labels=labels,
                title=title,
            )
            if marker_kwargs:
                fig.update_traces(mode="lines+markers", marker=marker_kwargs)
            if line_kwargs:
                fig.update_traces(line=line_kwargs)
        else:
            fig = px.scatter(
                df,
                x=xcol,
                y=ycol,
                color=color_by,
                symbol=symbol_by,
                hover_data=hover,
                color_discrete_sequence=color_sequence,
                labels=labels,
                title=title,
            )
            if marker_kwargs:
                fig.update_traces(marker=marker_kwargs)
    elif mode == "3d":
        if len(dims) < 3:
            raise ValueError("dims must provide three dimensions for 3d mode.")
        xcol, ycol, zcol = f"x{dims[0]}", f"x{dims[1]}", f"x{dims[2]}"
        if connect_points:
            fig = px.line_3d(
                df,
                x=xcol,
                y=ycol,
                z=zcol,
                color=color_by,
                hover_data=hover,
                color_discrete_sequence=color_sequence,
                labels=labels,
                title=title,
            )
            if marker_kwargs:
                fig.update_traces(mode="lines+markers", marker=marker_kwargs)
            if line_kwargs:
                fig.update_traces(line=line_kwargs)
        else:
            fig = px.scatter_3d(
                df,
                x=xcol,
                y=ycol,
                z=zcol,
                color=color_by,
                symbol=symbol_by,
                hover_data=hover,
                color_discrete_sequence=color_sequence,
                labels=labels,
                title=title,
            )
            if marker_kwargs:
                fig.update_traces(marker=marker_kwargs)
    else:
        raise ValueError("mode must be one of {'time', '2d', '3d'}.")

    return fig
