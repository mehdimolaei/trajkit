.. image:: assets/logo.png
   :width: 260
   :align: center

trajkit: trajectory analytics and flow-field inference
======================================================

trajkit is a Python toolkit for reproducible trajectory analytics and flow-field inference
for Brownian and active colloids.

**Quick install**

.. code:: bash

   pip install trajkit

**What you can do**

- Load trajectory datasets (frame-based or non-uniform time sampling)
- Clean and manipulate tracks
- Compute MSD and displacement statistics
- Estimate flow fields around particles via CDV / correlation utilities
- Build publishable visualizations + notebooks

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
   :caption: API Reference

   reference/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
