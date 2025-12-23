# Trajectories

`trajkit` centers around two core objects:

- **Trajectory**: a single track with positions `x` shaped `(T, D)` plus either `t` (seconds) or
  `frame` + `frame_rate_hz` for uniformly sampled data. Optional fields let you attach per-frame
  masks (`valid`), per-track metadata, and frame-level features.
- **TrajectorySet**: a collection of trajectories from one dataset/experiment with shared metadata
  such as units, calibration, and experimental conditions.

Key utilities:
- `Trajectory.time_seconds()` to compute time in seconds
- `Trajectory.msd()` and `TrajectorySet.msd()` helpers for mean squared displacement
- `TrajectorySet.summary_table()` to get a tidy overview of all tracks
