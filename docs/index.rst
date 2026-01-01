.. image:: assets/long_logo.png
   :width: 260
   :align: center

Trajkit — Spatio-Temporal Trajectory Toolkit
============================================

Trajkit is a Python toolkit for analyzing **spatio-temporal trajectories and structured time-series**.
It is built around a principled, physics-informed data model that separates:

* geometry (positions evolving in time),
* time structure (frames or continuous timestamps),
* and metadata (labels, features, experimental context).

This makes trajkit useful across many domains:

* Brownian and non-Brownian colloids
* active matter and swimming microorganisms
* hydrodynamic flow inference and correlated displacement velocimetry (CDV)
* cell and subcellular dynamics
* imaging-based tracking experiments
* general multi-dimensional trajectories with meaningful distance metrics

Trajkit treats trajectories not as simple lists of points,
but as **structured objects evolving in time, embedded in a geometric space,
and enriched with metadata**. This enables rigorous statistics,
meaningful geometry, and reproducible scientific workflows.


Core Philosophy
---------------

Trajkit is designed around three core principles:

1. **Geometry First**

   Positions live in a well-defined metric space so displacement, distance,
   and gradients are always meaningful.

   .. code-block::

       Core Arrays
       -----------
       x      : (T, D)  coordinates in D-dimensional space
       t      : (T,)    time stamps (non-uniform allowed)
       frame  : (T,)    discrete frame index (optional)
       valid  : (T,)    boolean mask for missing / invalid samples

2. **Metadata is powerful — but separate from geometry**

   Real experiments encode richness beyond coordinates.
   Trajkit supports structured metadata without polluting the core geometry.

   .. code-block::

       Metadata
       --------
       label           : grouping / class identifier
       track_features  : per-trajectory scalar features
       frame_features  : per-frame feature arrays
       meta            : arbitrary metadata / provenance

3. **Built for statistics, modeling, and discovery**

   Trajkit provides tools for:

   * diffusion and anomalous transport statistics
   * van Hove distributions and correlations
   * velocity and displacement statistics
   * correlated displacement velocimetry (CDV)
   * handling missing data, smoothing, interpolation
   * visualization, diagnostics, and dataset-level analysis

Everything works consistently because every analysis routine
assumes the same clean data model.


Why This Structure?
--------------------

Spatio-temporal data naturally carries three layers of meaning:

=================  ===========================================
Layer              Meaning
=================  ===========================================
Geometry           where things are and how they move
Time               how evolution unfolds
Context            what the trajectory represents
=================  ===========================================

Separating these cleanly makes trajkit:

* physically meaningful
* domain-agnostic
* scalable
* machine-learning friendly
* reproducible
* future-proof


Typical Workflow
-----------------

A common usage pattern looks like:

1. Load raw tracking data (CSV, experiment output, simulation, etc.)
2. Group into trajectories
3. Construct trajkit objects (``x``, ``t`` or ``frame``, optional ``valid``)
4. Optionally attach labels, per-track and per-frame features, metadata
5. Apply analysis tools:

   * Mean-Squared Displacement
   * van Hove distributions
   * correlation statistics
   * hydrodynamic inference (CDV)
   * visualization and summaries


Vision
-------

Trajkit is built from experience across:

* soft matter and colloidal systems
* active matter dynamics
* microscopy and computational imaging
* rare-event and ultra-sensitive detection pipelines
* scientific software engineering

The goal is a **scientifically rigorous, modern, and practical**
foundation for studying trajectories — while remaining flexible enough
to grow into broader structured time-series analysis.


Documentation
--------------

Use the sections below to explore concepts, tutorials, and API details.


.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   getting_started/installation
   getting_started/quickstart
   getting_started/data_model
   getting_started/examples


.. toctree::
   :maxdepth: 2
   :caption: Core Concepts

   concepts/trajectories


.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/first-notebook
   tutorials/cdv-flow-field
   tutorials/msd

.. toctree::
   :maxdepth: 2
   :caption: Visualization

   visualization/index


.. toctree::
   :maxdepth: 2
   :caption: API Reference

   reference/index


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
