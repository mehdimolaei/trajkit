"""
trajkit.viz.three_d

Utilities to visualize 3D droplet trajectories over time.

Assumed input: a pandas DataFrame with columns:
    id, t, x, y, z, r

Each row = one droplet at one time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class TrajectoryFrameSpec:
    """Specification of column names in the DataFrame."""

    id_col: str = "id"
    t_col: str = "t"
    x_col: str = "x"
    y_col: str = "y"
    z_col: str = "z"
    r_col: Optional[str] = "r"  # can be None if you don't have radii


def ensure_dataframe(
    data: pd.DataFrame | Iterable[dict],
    spec: TrajectoryFrameSpec = TrajectoryFrameSpec(),
) -> pd.DataFrame:
    """
    Ensure we have a DataFrame with the expected columns.

    Parameters
    ----------
    data
        Either a DataFrame or an iterable of dicts with the required keys.
    spec
        Column name specification.

    Returns
    -------
    DataFrame
        Copy of the input with the required columns.
    """
    if isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        df = pd.DataFrame(list(data))

    required = [spec.id_col, spec.t_col, spec.x_col, spec.y_col, spec.z_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in trajectory data: {missing}")

    # Cast types nicely
    df[spec.t_col] = df[spec.t_col].astype(float)
    df[spec.x_col] = df[spec.x_col].astype(float)
    df[spec.y_col] = df[spec.y_col].astype(float)
    df[spec.z_col] = df[spec.z_col].astype(float)

    if spec.r_col is not None and spec.r_col in df.columns:
        df[spec.r_col] = df[spec.r_col].astype(float)

    return df


def animate_3d_plotly(
    data: pd.DataFrame,
    spec: TrajectoryFrameSpec = TrajectoryFrameSpec(),
    marker_scale: float = 1.0,
    max_traces_per_frame: int = 1,
    axis_padding: float = 0.05,
    camera_eye: Tuple[float, float, float] = (1.6, 1.6, 1.2),
    play_frame_duration_ms: int = 60,
):
    """
    Interactive 3D animation using Plotly.

    - 3D scatter, sized by radius, colored by track id.
    - Time slider + play/pause controls.

    Parameters
    ----------
    data
        Trajectory table with columns (id, t, x, y, z, r).
    spec
        Column name specification.
    marker_scale
        Factor to scale radii into marker sizes.
    max_traces_per_frame
        Number of traces per frame. Leave at 1 unless you want to split by label later.
    axis_padding
        Fractional padding added to min/max on each axis to keep points away from the box.
    camera_eye
        Default camera eye position; reset button returns to this view.
    play_frame_duration_ms
        Playback speed (per-frame duration in ms) for the Play button.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    import itertools

    import plotly.express as px
    import plotly.graph_objects as go

    df = ensure_dataframe(data, spec)
    tcol, idcol = spec.t_col, spec.id_col
    xcol, ycol, zcol = spec.x_col, spec.y_col, spec.z_col
    rcol = spec.r_col

    time_points = np.sort(df[tcol].unique())
    ids = df[idcol].unique()

    # Axis ranges locked across frames
    def _axis_range(s: pd.Series) -> Tuple[float, float]:
        mn, mx = float(s.min()), float(s.max())
        pad = (mx - mn) * axis_padding if mx != mn else 1.0
        return mn - pad, mx + pad

    x_range = _axis_range(df[xcol])
    y_range = _axis_range(df[ycol])
    z_range = _axis_range(df[zcol])

    # Map each id → color
    palette = itertools.cycle(px.colors.qualitative.Dark24)
    color_by_id = {i: next(palette) for i in ids}

    frames = []
    for t in time_points:
        df_t = df[df[tcol] == t]
        frame_name = str(t)

        marker_sizes = None
        if rcol is not None and rcol in df_t.columns:
            # Plotly marker.size is in pixels; radius is arbitrary units
            marker_sizes = df_t[rcol].to_numpy() * marker_scale

        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=df_t[xcol],
                        y=df_t[ycol],
                        z=df_t[zcol],
                        mode="markers",
                        marker=dict(
                            size=marker_sizes,
                            color=[color_by_id[i] for i in df_t[idcol]],
                            opacity=1,
                        ),
                    )
                ],
                name=frame_name,
            )
        )

    # Initial frame
    df0 = df[df[tcol] == time_points[0]]
    marker_sizes0 = None
    if spec.r_col is not None and spec.r_col in df0.columns:
        marker_sizes0 = df0[spec.r_col].to_numpy() * marker_scale

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=df0[xcol],
                y=df0[ycol],
                z=df0[zcol],
                mode="markers",
                marker=dict(
                    size=marker_sizes0,
                    color=[color_by_id[i] for i in df0[idcol]],
                    opacity=1.0,
                ),
            )
        ],
        frames=frames,
    )

    # Slider + buttons
    slider_steps = []
    for t in time_points:
        frame_name = str(t)
        slider_steps.append(
            {
                "label": f"{t:.3g}",
                "method": "animate",
                "args": [
                    [frame_name],
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
            }
        )

    fig.update_layout(
        scene=dict(
            xaxis=dict(title=xcol, range=x_range),
            yaxis=dict(title=ycol, range=y_range),
            zaxis=dict(title=zcol, range=z_range),
            aspectmode="data",
            camera={"eye": {"x": camera_eye[0], "y": camera_eye[1], "z": camera_eye[2]}},
            dragmode="orbit",
        ),
        uirevision="traj3d",
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "y": 1.15,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": play_frame_duration_ms, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "y": -0.05,
                "x": 0.1,
                "len": 0.9,
                "steps": slider_steps,
            }
        ],
        margin=dict(l=0, r=0, b=0, t=40),
    )

    return fig


def animate_3d_mpl(
    data: pd.DataFrame,
    spec: TrajectoryFrameSpec = TrajectoryFrameSpec(),
    marker_scale: float = 50.0,
    fps: int = 20,
    output: Optional[str] = None,
):
    """
    3D scatter animation using Matplotlib.

    - Uses a single 3D scatter object updated per frame.
    - Good for MP4 or GIF export.

    Parameters
    ----------
    data
        Trajectory table with (id, t, x, y, z, r).
    spec
        Column name specification.
    marker_scale
        Size factor. Matplotlib `s` is in points^2, so we typically square radii*scale.
    fps
        Frames per second for saved video.
    output
        If given, path to save animation ('.mp4' or '.gif').
        If None, returns the animation object.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import matplotlib.cm as cm

    df = ensure_dataframe(data, spec)
    tcol, idcol = spec.t_col, spec.id_col
    xcol, ycol, zcol = spec.x_col, spec.y_col, spec.z_col
    rcol = spec.r_col

    time_points = np.sort(df[tcol].unique())
    ids = np.unique(df[idcol])
    n_ids = len(ids)

    # Color map for IDs
    cmap = cm.get_cmap("tab20", n_ids)
    id_to_color = {i: cmap(k / max(1, n_ids - 1)) for k, i in enumerate(ids)}

    # Initial data
    df0 = df[df[tcol] == time_points[0]]
    sizes0 = None
    if rcol is not None and rcol in df0.columns:
        sizes0 = (df0[rcol].to_numpy() * marker_scale) ** 2
    else:
        sizes0 = np.full(len(df0), 30.0)

    colors0 = [id_to_color[i] for i in df0[idcol]]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
        df0[xcol],
        df0[ycol],
        df0[zcol],
        s=sizes0,
        c=colors0,
        depthshade=True,
    )

    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_zlabel(zcol)
    ax.set_title(f"t = {time_points[0]:.3g}")

    # Fix aspect ratio roughly
    def _set_equal_aspect(ax):
        xs = df[xcol]
        ys = df[ycol]
        zs = df[zcol]
        max_range = np.array(
            [
                xs.max() - xs.min(),
                ys.max() - ys.min(),
                zs.max() - zs.min(),
            ]
        ).max()
        mid_x = 0.5 * (xs.max() + xs.min())
        mid_y = 0.5 * (ys.max() + ys.min())
        mid_z = 0.5 * (zs.max() + zs.min())
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    _set_equal_aspect(ax)

    def update(frame_index: int):
        t = time_points[frame_index]
        df_t = df[df[tcol] == t]

        xs = df_t[xcol].to_numpy()
        ys = df_t[ycol].to_numpy()
        zs = df_t[zcol].to_numpy()

        if rcol is not None and rcol in df_t.columns:
            sizes = (df_t[rcol].to_numpy() * marker_scale) ** 2
        else:
            sizes = np.full(len(df_t), 30.0)

        colors = [id_to_color[i] for i in df_t[idcol]]

        # Update scatter
        sc._offsets3d = (xs, ys, zs)
        sc.set_sizes(sizes)
        sc.set_color(colors)

        ax.set_title(f"t = {t:.3g}")
        return (sc,)

    anim = FuncAnimation(
        fig,
        update,
        frames=len(time_points),
        interval=1000.0 / fps,
        blit=False,
    )

    if output is not None:
        if output.lower().endswith(".mp4"):
            from matplotlib.animation import FFMpegWriter

            writer = FFMpegWriter(fps=fps, codec="libx264")
            anim.save(output, writer=writer, dpi=150)
        elif output.lower().endswith(".gif"):
            anim.save(output, writer="imagemagick", dpi=100)
        else:
            raise ValueError("output must end with .mp4 or .gif")

        plt.close(fig)
        return output

    return anim


def animate_3d_pyvista(
    data: pd.DataFrame,
    spec: TrajectoryFrameSpec = TrajectoryFrameSpec(),
    output: Optional[str] = "droplets.gif",
    sphere_resolution: int = 24,
    cmap: str = "tab20",
):
    """
    Real 3D spheres with PyVista.

    - Creates a sphere mesh per droplet per frame.
    - Colors by track id.
    - Writes a GIF or MP4 if output is provided.

    Parameters
    ----------
    data
        Trajectory data with (id, t, x, y, z, r).
    spec
        Column name specification.
    output
        Output filename (`.gif` or `.mp4`). If None, shows an interactive window.
    sphere_resolution
        Theta/phi resolution of the spheres.
    cmap
        Colormap name for IDs.
    """
    import pyvista as pv
    import matplotlib.cm as cm

    df = ensure_dataframe(data, spec)
    tcol, idcol = spec.t_col, spec.id_col
    xcol, ycol, zcol = spec.x_col, spec.y_col, spec.z_col
    rcol = spec.r_col

    if rcol is None or rcol not in df.columns:
        raise ValueError("PyVista animation requires a radius column (spec.r_col).")

    time_points = np.sort(df[tcol].unique())
    ids = np.unique(df[idcol])
    n_ids = len(ids)

    cmap_obj = cm.get_cmap(cmap, n_ids)
    id_to_color = {i: cmap_obj(k / max(1, n_ids - 1)) for k, i in enumerate(ids)}

    # Off-screen if saving
    off_screen = output is not None

    plotter = pv.Plotter(off_screen=off_screen)

    if output is not None:
        if output.lower().endswith(".gif"):
            plotter.open_gif(output)
        elif output.lower().endswith(".mp4"):
            plotter.open_movie(output, framerate=20)
        else:
            raise ValueError("Output must end with .gif or .mp4")

    # Optional: set a reasonable camera once using the full data
    xs, ys, zs = df[xcol], df[ycol], df[zcol]
    center = (xs.mean(), ys.mean(), zs.mean())
    plotter.set_focus(center)

    for t in time_points:
        plotter.clear()
        df_t = df[df[tcol] == t]

        for _, row in df_t.iterrows():
            radius = float(row[rcol])
            center = (float(row[xcol]), float(row[ycol]), float(row[zcol]))
            droplet_id = row[idcol]
            color = id_to_color[droplet_id]

            sphere = pv.Sphere(
                radius=radius,
                center=center,
                theta_resolution=sphere_resolution,
                phi_resolution=sphere_resolution,
            )

            plotter.add_mesh(
                sphere,
                color=color,
                smooth_shading=True,
                show_edges=False,
            )

        plotter.add_text(f"t = {t:.3g}", font_size=14)

        if output is not None:
            plotter.write_frame()
        else:
            plotter.show(auto_close=False)

    if output is not None:
        plotter.close()

    return output


def render_static_pyvista_scene(
    data: pd.DataFrame,
    spec: TrajectoryFrameSpec = TrajectoryFrameSpec(),
    bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
    axis_padding: float = 0.05,
    wall_opacity: float = 0.08,
    wall_color: str = "slateblue",
    edge_radius: float = 1.5,
    edge_color: str = "black",
    sphere_resolution: int = 30,
    clip_to_bounds: bool = True,
    clip_strategy: str = "auto",  # auto | boolean | box
    bounds_pad_fraction: float = 0.0,
    background: str = "white",
    screenshot: Optional[str] = None,
    camera_position: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]] = "iso",
    depth_peeling: bool = True,
    eye_dome_lighting: bool = True,
    anti_aliasing: bool = True,
    use_pbr: bool = True,
    specular: float = 0.35,
    specular_power: float = 20.0,
    metallic: float = 0.2,
    roughness: float = 0.35,
    add_default_lights: bool = True,
    camera_distance_scale: float = 1.35,
    fill_clipped_holes: bool = True,
    fill_holes_radius_scale: float = 0.25,
):
    """
    Static PyVista rendering similar to the legacy VPython SGM notebook.

    - Draws a bounding box (3 translucent walls + edge tubes).
    - Renders spheres at (x, y, z) with radius r, clipped to the box bounds.
    - Colors tracks by ID for readability.

    Parameters
    ----------
    data
        Trajectory table with columns (id, t, x, y, z, r).
    spec
        Column name specification.
    bounds
        Optional (xmin, xmax, ymin, ymax, zmin, zmax). If None, inferred from data with padding.
    axis_padding
        Fractional padding added when inferring bounds.
    wall_opacity
        Opacity for the translucent planes.
    wall_color
        Color for the walls/box surfaces.
    edge_radius
        Tube radius for box edges.
    edge_color
        Color for the edge tubes.
    sphere_resolution
        Theta/phi resolution for spheres.
    clip_to_bounds
        If True, clip spheres to the bounding box (approximates the VPython hemisphere+cover effect).
    background
        Plotter background color.
    screenshot
        If provided, path to save a screenshot (off-screen).
    """
    import matplotlib.cm as cm
    import pyvista as pv

    df = ensure_dataframe(data, spec)
    xcol, ycol, zcol = spec.x_col, spec.y_col, spec.z_col
    rcol = spec.r_col
    if rcol is None or rcol not in df.columns:
        raise ValueError("render_static_pyvista_scene requires a radius column (spec.r_col).")

    # Bounds: infer from data with padding if not provided
    def _pad_range(arr: pd.Series) -> Tuple[float, float]:
        mn, mx = float(arr.min()), float(arr.max())
        span = mx - mn
        pad = span * axis_padding if span > 0 else 1.0
        return mn - pad, mx + pad

    if bounds is None:
        xmin, xmax = _pad_range(df[xcol])
        ymin, ymax = _pad_range(df[ycol])
        zmin, zmax = _pad_range(df[zcol])
        bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = bounds

    box = pv.Box(bounds=bounds)
    edges = box.extract_all_edges().tube(radius=edge_radius)

    # Slightly padded bounds to reduce numerical artifacts at faces/corners during clipping
    span = max(xmax - xmin, ymax - ymin, zmax - zmin)
    pad = bounds_pad_fraction * span
    padded_bounds = (
        xmin + pad,
        xmax - pad,
        ymin + pad,
        ymax - pad,
        zmin + pad,
        zmax - pad,
    )
    padded_box = pv.Box(bounds=padded_bounds)

    def _clip_mesh_to_box(mesh: "pv.DataSet") -> Tuple[Optional["pv.DataSet"], bool]:
        if not clip_to_bounds:
            return mesh, False
        strategies = []
        if clip_strategy == "boolean":
            strategies = ["boolean"]
        elif clip_strategy == "box":
            strategies = ["box"]
        else:
            strategies = ["boolean", "box"]

        for strat in strategies:
            try:
                if strat == "boolean":
                    clipped = mesh.boolean_intersection(padded_box)
                    if clipped is not None and clipped.n_cells > 0:
                        return clipped, True
                else:
                    clipped = mesh.clip_box(bounds=padded_bounds, invert=False)
                    if clipped is not None and clipped.n_cells > 0:
                        return clipped, False
            except Exception:
                continue

        return None, False

    # Color palette per track id (fall back to a single color)
    ids = df[spec.id_col].unique() if spec.id_col in df.columns else np.array([0])
    cmap = cm.get_cmap("tab20", len(ids))
    color_by_id = {i: cmap(k / max(1, len(ids) - 1))[:3] for k, i in enumerate(ids)}

    plotter = pv.Plotter(off_screen=screenshot is not None)
    if depth_peeling:
        plotter.enable_depth_peeling()
    if eye_dome_lighting:
        plotter.enable_eye_dome_lighting()
    if anti_aliasing:
        plotter.enable_anti_aliasing()
    if add_default_lights:
        plotter.remove_all_lights()
        plotter.add_light(pv.Light(position=(2, 2, 2), color="white", light_type="scene light"))
        plotter.add_light(pv.Light(position=(-2, -1, 3), color="white", light_type="scene light"))
        plotter.add_light(pv.Light(color="white", light_type="headlight"))
    plotter.set_background(background)

    # Three orthogonal walls to match the VPython scene
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    walls = [
        pv.Plane(center=(xmin, 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)), direction=(1, 0, 0),
                 i_size=(ymax - ymin), j_size=(zmax - zmin)),  # left
        pv.Plane(center=(xmax, 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)), direction=(-1, 0, 0),
                 i_size=(ymax - ymin), j_size=(zmax - zmin)),  # right
        pv.Plane(center=(0.5 * (xmin + xmax), ymin, 0.5 * (zmin + zmax)), direction=(0, 1, 0),
                 i_size=(xmax - xmin), j_size=(zmax - zmin)),  # front
        pv.Plane(center=(0.5 * (xmin + xmax), ymax, 0.5 * (zmin + zmax)), direction=(0, -1, 0),
                 i_size=(xmax - xmin), j_size=(zmax - zmin)),  # back
        pv.Plane(center=(0.5 * (xmin + xmax), 0.5 * (ymin + ymax), zmin), direction=(0, 0, 1),
                 i_size=(xmax - xmin), j_size=(ymax - ymin)),  # bottom
        pv.Plane(center=(0.5 * (xmin + xmax), 0.5 * (ymin + ymax), zmax), direction=(0, 0, -1),
                 i_size=(xmax - xmin), j_size=(ymax - ymin)),  # top
    ]
    for wall in walls:
        plotter.add_mesh(wall, color=wall_color, opacity=wall_opacity, smooth_shading=False)

    plotter.add_mesh(edges, color=edge_color)

    def _intersects_box(center: Tuple[float, float, float], radius: float) -> bool:
        cx, cy, cz = center
        return not (
            cx + radius < xmin
            or cx - radius > xmax
            or cy + radius < ymin
            or cy - radius > ymax
            or cz + radius < zmin
            or cz - radius > zmax
        )

    for _, row in df.iterrows():
        center = (float(row[xcol]), float(row[ycol]), float(row[zcol]))
        radius = float(row[rcol])
        color = color_by_id.get(row.get(spec.id_col, None), (0.2, 0.6, 0.3))

        if not _intersects_box(center, radius):
            continue

        sphere = pv.Sphere(
            radius=radius,
            center=center,
            theta_resolution=sphere_resolution,
            phi_resolution=sphere_resolution,
        )

        clipped_via_boolean = False
        if clip_to_bounds:
            sphere, clipped_via_boolean = _clip_mesh_to_box(sphere)
            if sphere is None:
                continue

        def _prepare_mesh(mesh: "pv.DataSet", radius_local: float) -> "pv.PolyData":
            # Ensure we have a PolyData with normals even after boolean ops (which can yield UnstructuredGrid).
            if hasattr(mesh, "compute_normals"):
                surf = mesh.triangulate()
            else:
                surf = mesh.extract_surface().triangulate()

            if fill_clipped_holes and not clipped_via_boolean:
                try:
                    fill_r = max(radius_local * fill_holes_radius_scale, 1e-3)
                    surf = surf.fill_holes(fill_r)
                except Exception:
                    pass

            surf = surf.clean()
            return surf.compute_normals(cell_normals=False, auto_orient_normals=True)

        sphere = _prepare_mesh(sphere, radius)

        mesh_kwargs = dict(
            color=color,
            specular=specular,
            specular_power=specular_power,
            smooth_shading=True,
        )
        if use_pbr:
            mesh_kwargs.update(pbr=True, metallic=metallic, roughness=roughness)

        plotter.add_mesh(sphere, **mesh_kwargs)

    if screenshot:
        plotter.show(auto_close=True, screenshot=screenshot)
        return screenshot

    # Fit and set camera after geometry is added to keep FOV centered.
    plotter.reset_camera(bounds=box.bounds)
    if camera_position == "iso":
        cx, cy, cz = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)
        span = max(xmax - xmin, ymax - ymin, zmax - zmin)
        dist = camera_distance_scale * span if span > 0 else 10.0
        eye = (cx + dist, cy + dist, cz + dist)
        plotter.camera_position = (eye, (cx, cy, cz), (0, 0, 1))
    elif camera_position:
        plotter.camera_position = camera_position

    plotter.show()
    return plotter


def animate_static_pyvista_scene(
    data: pd.DataFrame,
    spec: TrajectoryFrameSpec = TrajectoryFrameSpec(),
    bounds: Optional[Tuple[float, float, float, float, float, float]] = None,
    axis_padding: float = 0.05,
    wall_opacity: float = 0.08,
    wall_color: str = "slateblue",
    edge_radius: float = 1.5,
    edge_color: str = "black",
    sphere_resolution: int = 24,
    clip_to_bounds: bool = True,
    clip_strategy: str = "auto",  # auto | boolean | box
    bounds_pad_fraction: float = 0.0,
    background: str = "white",
    output: str = "scene.gif",
    fps: int = 20,
    camera_position: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]] = "iso",
    depth_peeling: bool = True,
    eye_dome_lighting: bool = True,
    anti_aliasing: bool = True,
    use_pbr: bool = True,
    specular: float = 0.35,
    specular_power: float = 20.0,
    metallic: float = 0.2,
    roughness: float = 0.35,
    add_default_lights: bool = True,
    camera_distance_scale: float = 1.35,
    fill_clipped_holes: bool = True,
    fill_holes_radius_scale: float = 0.25,
):
    """
    Animate the static PyVista scene (walls + clipped spheres) and save to GIF/MP4.

    Frames are cleared before each draw, so objects from prior frames do not overlay.
    """
    import matplotlib.cm as cm
    import pyvista as pv

    df = ensure_dataframe(data, spec)
    xcol, ycol, zcol = spec.x_col, spec.y_col, spec.z_col
    rcol, tcol = spec.r_col, spec.t_col

    if rcol is None or rcol not in df.columns:
        raise ValueError("animate_static_pyvista_scene requires a radius column (spec.r_col).")
    if tcol not in df.columns:
        raise ValueError("animate_static_pyvista_scene requires a time column (spec.t_col).")

    time_points = np.sort(df[tcol].unique())

    def _pad_range(arr: pd.Series) -> Tuple[float, float]:
        mn, mx = float(arr.min()), float(arr.max())
        span = mx - mn
        pad = span * axis_padding if span > 0 else 1.0
        return mn - pad, mx + pad

    if bounds is None:
        xmin, xmax = _pad_range(df[xcol])
        ymin, ymax = _pad_range(df[ycol])
        zmin, zmax = _pad_range(df[zcol])
        bounds = (xmin, xmax, ymin, ymax, zmin, zmax)
    else:
        xmin, xmax, ymin, ymax, zmin, zmax = bounds

    box = pv.Box(bounds=bounds)
    edges = box.extract_all_edges().tube(radius=edge_radius)

    span = max(xmax - xmin, ymax - ymin, zmax - zmin)
    pad = bounds_pad_fraction * span
    padded_bounds = (
        xmin + pad,
        xmax - pad,
        ymin + pad,
        ymax - pad,
        zmin + pad,
        zmax - pad,
    )
    padded_box = pv.Box(bounds=padded_bounds)

    def _clip_mesh_to_box(mesh: "pv.DataSet") -> Tuple[Optional["pv.DataSet"], bool]:
        if not clip_to_bounds:
            return mesh, False
        strategies = []
        if clip_strategy == "boolean":
            strategies = ["boolean"]
        elif clip_strategy == "box":
            strategies = ["box"]
        else:
            strategies = ["boolean", "box"]

        for strat in strategies:
            try:
                if strat == "boolean":
                    clipped = mesh.boolean_intersection(padded_box)
                    if clipped is not None and clipped.n_cells > 0:
                        return clipped, True
                else:
                    clipped = mesh.clip_box(bounds=padded_bounds, invert=False)
                    if clipped is not None and clipped.n_cells > 0:
                        return clipped, False
            except Exception:
                continue

        return None, False

    ids = df[spec.id_col].unique() if spec.id_col in df.columns else np.array([0])
    cmap = cm.get_cmap("tab20", len(ids))
    color_by_id = {i: cmap(k / max(1, len(ids) - 1))[:3] for k, i in enumerate(ids)}

    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    def _add_walls_and_edges(plotter: pv.Plotter):
        walls = [
            pv.Plane(center=(xmin, 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)), direction=(1, 0, 0),
                     i_size=(ymax - ymin), j_size=(zmax - zmin)),  # left
            pv.Plane(center=(xmax, 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)), direction=(-1, 0, 0),
                     i_size=(ymax - ymin), j_size=(zmax - zmin)),  # right
            pv.Plane(center=(0.5 * (xmin + xmax), ymin, 0.5 * (zmin + zmax)), direction=(0, 1, 0),
                     i_size=(xmax - xmin), j_size=(zmax - zmin)),  # front
            pv.Plane(center=(0.5 * (xmin + xmax), ymax, 0.5 * (zmin + zmax)), direction=(0, -1, 0),
                     i_size=(xmax - xmin), j_size=(zmax - zmin)),  # back
            pv.Plane(center=(0.5 * (xmin + xmax), 0.5 * (ymin + ymax), zmin), direction=(0, 0, 1),
                     i_size=(xmax - xmin), j_size=(ymax - ymin)),  # bottom
            pv.Plane(center=(0.5 * (xmin + xmax), 0.5 * (ymin + ymax), zmax), direction=(0, 0, -1),
                     i_size=(xmax - xmin), j_size=(ymax - ymin)),  # top
        ]
        for wall in walls:
            plotter.add_mesh(wall, color=wall_color, opacity=wall_opacity, smooth_shading=False)
        plotter.add_mesh(edges, color=edge_color)

    plotter = pv.Plotter(off_screen=output is not None)
    if depth_peeling:
        plotter.enable_depth_peeling()
    if eye_dome_lighting:
        plotter.enable_eye_dome_lighting()
    if anti_aliasing:
        plotter.enable_anti_aliasing()
    if add_default_lights:
        plotter.remove_all_lights()
        plotter.add_light(pv.Light(position=(2, 2, 2), color="white", light_type="scene light"))
        plotter.add_light(pv.Light(position=(-2, -1, 3), color="white", light_type="scene light"))
        plotter.add_light(pv.Light(color="white", light_type="headlight"))
    plotter.set_background(background)

    if output:
        lower = output.lower()
        if lower.endswith(".gif"):
            plotter.open_gif(output)
        elif lower.endswith(".mp4"):
            try:
                import imageio_ffmpeg  # noqa: F401
            except Exception as exc:  # pragma: no cover - runtime dependency check
                raise RuntimeError(
                    "MP4 output requires the imageio-ffmpeg plugin. "
                    "Install it via `python -m pip install \"imageio[ffmpeg]\"`."
                ) from exc
            plotter.open_movie(output, framerate=fps)
        else:
            raise ValueError("output must end with .gif or .mp4")

    # Lock the camera to a deterministic iso view based on bounds so FOV stays fixed
    def _set_camera(plotter: pv.Plotter):
        if camera_position == "iso":
            cx, cy, cz = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax), 0.5 * (zmin + zmax)
            span = max(xmax - xmin, ymax - ymin, zmax - zmin)
            dist = camera_distance_scale * span if span > 0 else 10.0
            eye = (cx + dist, cy + dist, cz + dist)
            plotter.camera_position = (eye, (cx, cy, cz), (0, 0, 1))
        elif camera_position:
            plotter.camera_position = camera_position

    for t in time_points:
        plotter.clear()
        _add_walls_and_edges(plotter)

        df_t = df[df[tcol] == t]

        for _, row in df_t.iterrows():
            center = (float(row[xcol]), float(row[ycol]), float(row[zcol]))
            radius = float(row[rcol])
            color = color_by_id.get(row.get(spec.id_col, None), (0.2, 0.6, 0.3))

            # Skip if the sphere is fully outside the bounds
            if (
                center[0] + radius < xmin
                or center[0] - radius > xmax
                or center[1] + radius < ymin
                or center[1] - radius > ymax
                or center[2] + radius < zmin
                or center[2] - radius > zmax
            ):
                continue

            sphere = pv.Sphere(
                radius=radius,
                center=center,
                theta_resolution=sphere_resolution,
                phi_resolution=sphere_resolution,
            )

            clipped_via_boolean = False
            if clip_to_bounds:
                sphere, clipped_via_boolean = _clip_mesh_to_box(sphere)
                if sphere is None:
                    continue

            def _prepare_mesh(mesh: "pv.DataSet") -> "pv.PolyData":
                if hasattr(mesh, "compute_normals"):
                    surf = mesh.triangulate()
                else:
                    surf = mesh.extract_surface().triangulate()
                if fill_clipped_holes and not clipped_via_boolean:
                    try:
                        span = max(xmax - xmin, ymax - ymin, zmax - zmin)
                        radius = max(span * fill_holes_radius_scale, 1e-3)
                        surf = surf.fill_holes(radius)
                    except Exception:
                        pass
                surf = surf.clean()
                return surf.compute_normals(cell_normals=False, auto_orient_normals=True)

            sphere = _prepare_mesh(sphere)

            mesh_kwargs = dict(
                color=color,
                specular=specular,
                specular_power=specular_power,
                smooth_shading=True,
            )
            if use_pbr:
                mesh_kwargs.update(pbr=True, metallic=metallic, roughness=roughness)

            plotter.add_mesh(sphere, **mesh_kwargs)

        plotter.add_text(f"{tcol} = {t}", font_size=10)

        # Keep FOV fixed: fit to box and set camera each frame.
        plotter.reset_camera(bounds=box.bounds)
        _set_camera(plotter)

        if output:
            plotter.write_frame()

    if output:
        plotter.close()
        return output

    plotter.show(auto_close=False)
    return plotter


def napari_view_tracks(
    data: pd.DataFrame,
    image: Optional[np.ndarray] = None,
    spec: TrajectoryFrameSpec = TrajectoryFrameSpec(z_col="z"),
    time_unit: str = "frame",
):
    """
    View trajectories as tracks in Napari, optionally on top of a 4D image.

    Expected image shape: (T, Z, Y, X) or (T, Y, X) if 2D in space.

    Napari tracks format is array of shape (N, 5):
        (track_id, t, z, y, x)

    Parameters
    ----------
    data
        Trajectory data with (id, t, x, y, z, r).
    image
        4D confocal stack (T, Z, Y, X) or (T, Y, X). Optional.
    spec
        Column name specification.
    time_unit
        Label for time axis (for metadata only).
    """
    import napari

    df = ensure_dataframe(data, spec)
    tcol, idcol = spec.t_col, spec.id_col
    xcol, ycol, zcol = spec.x_col, spec.y_col, spec.z_col

    # Napari tracks: columns = [id, t, z, y, x]
    tracks_data = df[[idcol, tcol, zcol, ycol, xcol]].to_numpy(dtype=float)

    viewer = napari.Viewer()

    if image is not None:
        # Handle 3D vs 4D
        if image.ndim == 4:
            viewer.add_image(image, name="confocal", scale=(1, 1, 1, 1))
        elif image.ndim == 3:
            viewer.add_image(image, name="confocal", scale=(1, 1, 1))
        else:
            raise ValueError("image must be 3D (T, Y, X) or 4D (T, Z, Y, X).")

    viewer.add_tracks(
        tracks_data,
        name="droplets",
        tail_length=10,
        tail_width=2,
        colormap="tab20",
        blending="translucent",
        metadata={"time_unit": time_unit},
    )

    # Napari starts its own event loop
    napari.run()
