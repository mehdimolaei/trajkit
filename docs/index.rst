.. image:: _static/long_logo.png
   :width: 360
   :align: center

Spatio-Temporal Trajectory Toolkit
============================================

Trajkit is a Python toolkit for analyzing **spatio-temporal trajectories and structured time-series** data.
It is built on a principled, physics-informed data model that explicitly separates:

* **state-space coordinates** — positions (and potentially higher-dimensional states) evolving in time
* **time structure** — discrete frames or continuous timestamps defining temporal sampling
* **metadata** — labels, features, and experimental or contextual information

Trajkit supports trajectory analysis across diverse scientific and engineering contexts, including physical transport processes, biological motion, imaging-based tracking, and general multi-dimensional dynamical systems.

Data Model
---------------

Trajkit adopts a structured representation of trajectories grounded in state-space geometry, temporal organization, and contextual metadata:

1. **State-space Representation**

   Trajectories are embedded in a metric space, enabling well-defined notions of displacement,
    distance, and gradient fields.
   

   .. code-block::

       Core Arrays
       -----------
       x      : (T, D)  coordinates in D-dimensional space
       t      : (T,)    time stamps (non-uniform allowed)
       frame  : (T,)    discrete frame index (optional)
       valid  : (T,)    boolean mask for missing / invalid samples

2. **Context**

   Metadata is designed to remain closely associated with the trajectory data. 
   Experimental labels, derived features, and modality-specific annotations stay synchronized with frames and tracks, enabling reproducible multimodal analysis and reliable data provenance.
   .. code-block::

       Metadata
       --------
       label           : grouping / class identifier
       track_features  : per-trajectory scalar features
       frame_features  : per-frame feature arrays
       meta            : arbitrary metadata / provenance

3. **Statistics, and modeling**

Trajkit supports a broad range of applications in the analysis of dynamical systems and motion data, including:

   * characterization of transport behavior and motion regimes
   * quantification of fluctuations, variability, and heterogeneity in trajectories
   * inference of underlying flow and collective dynamics (e.g., CDV workflows)
   * robust handling of experimental imperfections such as missing or irregular samples
   * exploration, visualization, and systematic interrogation of large trajectory datasets
   
   By centering analysis on a consistent, multimodal trajectory representation, Trajkit provides a foundation for studying physical, biological, and engineered systems where entities evolve in time within a meaningful state space.


Why This Structure?
--------------------

Spatio-temporal data naturally carries three layers of meaning:

=================  ===========================================
Layer              Meaning
=================  ===========================================
State-Space        where things are and how they move
Time               how evolution unfolds
Context            what the trajectory represents
=================  ===========================================

Separating these cleanly makes trajkit:

* physically meaningful
* domain-agnostic
* scalable
* AI friendly
* reproducible


Typical Workflow
-----------------

A common usage pattern looks like:

1. Load raw tracking data (CSV, experiment output, simulation, etc.)
2. Group into trajectories
3. Construct trajkit objects (``x``, ``t`` or ``frame``, optional ``valid``)
4. Optionally attach labels, per-track and per-frame features, metadata
5. Apply analysis tools


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

.. toctree::
   :maxdepth: 2
   :caption: Tools

   tools/cdv_flow_inference
   tools/two_point_microrheology


.. toctree::
   :maxdepth: 2
   :caption: Tutorials

   tutorials/index
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
