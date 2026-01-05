:orphan:

Data model
==========

Core arrays
-----------

- ``x`` : (T, D) positions
- ``t`` : (T,) time in seconds (non-uniform allowed)
- ``frame`` : (T,) integer frame index (optional)
- ``valid`` : (T,) boolean mask

Metadata
--------

- ``label`` : grouping/class
- ``track_features`` : per-trajectory scalars
- ``frame_features`` : per-frame arrays
- ``meta`` : arbitrary extras

Timing rules
------------

- Provide either ``t``, or (``frame`` + ``frame_rate_hz``).
- Time and frame must be non-decreasing.

TrajectorySet
-------------

- Collection of ``Trajectory`` objects plus dataset-level ``units``, ``calibration``,
  ``conditions``, and ``meta``.
- Build from long-form tables via ``TrajectorySet.from_dataframe(...)``.
