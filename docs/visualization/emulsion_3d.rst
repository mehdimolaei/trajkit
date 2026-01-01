Emulsion droplet 3D scene (confocal)
====================================

This example shows how to load a droplet trajectory table from Hugging Face and render a 3D scene with PyVista using ``trajkit.viz.three_d``. It matches the scene used to generate ``docs/figures/sgm_scene_v1.mp4``.

Key pieces:

* Data source: CSV with columns ``id, t (or frame), x, y, z, r`` downloaded from Hugging Face.
* Visualization: ``animate_static_pyvista_scene`` draws a boxed field-of-view and clips droplets at the boundaries, producing an MP4/GIF.
* Notebook: see ``examples/visualization/Trajectory_StableDistribution.ipynb`` for the full workflow.

Rendered video
--------------

.. raw:: html

   <video width="720" controls muted loop>
     <source src="../figures/sgm_scene_v1.mp4" type="video/mp4">
     Your browser does not support the video tag.
   </video>

Workflow highlights
-------------------

1. Read the CSV directly from Hugging Face (or your mirror) and ensure columns match a ``TrajectoryFrameSpec``.
2. Optionally build a ``TrajectorySet`` for analysis.
3. Call ``animate_static_pyvista_scene`` with explicit bounds to clip droplets to the field of view. Tweak:

   * ``clip_strategy``: ``box`` (default) or ``boolean`` for the clipping backend.
   * ``fill_clipped_holes`` and ``fill_holes_radius_scale``: keep clipped droplets solid instead of hollow shells.
   * ``camera_distance_scale`` and ``background``: adjust framing and scene tone.

Run it yourself
---------------

Open and execute ``examples/visualization/Trajectory_StableDistribution.ipynb``. The notebook downloads the CSV, builds a ``TrajectorySet``, and writes an MP4 (defaults to ``sgm_scene.mp4`` in the notebook folder). Adjust ``bounds`` to your experiment's field of view and set ``output`` to your desired path.
