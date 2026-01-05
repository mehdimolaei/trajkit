Flow Inference (CDV)
====================

Correlated Displacement Velocimetry (CDV) is a data-driven method for inferring **underlying correlated dynamics**
from ensembles of trajectories in systems where motion is stochastic but not independent.

Expanded high-level idea
------------------------
Many systems consist of **entities** (particles, tracers, cells, proteins, or other degrees of freedom) whose states
evolve in time while being embedded in—or coupled through—an underlying **medium**. In physical settings, the medium
may be a viscous fluid, a viscoelastic network, a membrane, or a crowded cytoplasm; in more abstract settings it can
be an effective interaction field. Even when individual trajectories appear random, they are often **correlated**
because the medium transmits disturbances, constraints, or stresses across space and time.

A compact way to express this is through **coupled stochastic dynamics**:

.. math::

   \dot{x}_i(t)=a_i(\{x\},t)+\sum_j \mathcal{K}_{ij}(\{x\},t)\,\xi_j(t),

where :math:`a_i` is a deterministic drift term and :math:`\xi_j(t)` are stochastic drives. The key point is that the
effective noise seen by different entities can be **structured and correlated** through the coupling operator
:math:`\mathcal{K}` (which encodes the medium).

A complementary decomposition is to separate motion into an entity-specific component and a shared, medium-mediated
component. CDV targets this shared component: it uses **cross-displacement statistics** to recover coherent structure
that is invisible at the level of single trajectories.

Two equivalent perspectives
---------------------------
There are two complementary (and, in many regimes, equivalent) ways to interpret correlated stochastic motion.
CDV sits precisely at the interface of these views.

Event → response (impulse-response view)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Each entity’s stochastic step can be treated as a small **impulse-like perturbation** applied to the medium.
The medium responds by generating a spatially structured field (e.g., a flow, deformation, or stress redistribution),
which then biases the displacements of neighboring entities. In this view, CDV reconstructs an empirical
**response kernel** by aggregating many spontaneous micro-impulses, effectively turning passive trajectory data into a
set of “virtual experiments.”

If the medium response is approximately linear at the scale of an event, the induced field can be expressed using a
Green’s-function (kernel) representation:

.. math::

   u(r,t)\approx \int G(r-r',t-t')\,f(r',t')\,dr'\,dt',

where :math:`f(r',t')` is an effective localized forcing (the “impulse”) and :math:`G` is the medium response kernel.
CDV aims to estimate the structured part of this response directly from data.

Fluctuating field drives everything (Eulerian view)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Instead of attributing randomness to independent “kicks” acting on each entity, one can treat the **field itself**
as stochastic: the medium carries a fluctuating velocity/deformation field that is correlated in space and time.
Entities are then advected, dragged, or constrained by this shared fluctuating field, naturally producing correlated
displacements across the ensemble.

A common representation is to decompose the field into a mean and a fluctuation:

.. math::

   u(r,t)=\bar{u}(r,t)+u'(r,t),

and then write the entity dynamics as advection plus other (possibly local) contributions:

.. math::

   \dot{x}_i(t)=u(x_i,t)+\cdots

These perspectives are two sides of the same coupled system: in linear-response settings, “random impulses” and
“random fields” are dual descriptions. CDV is best understood as an **Eulerian inference method expressed in
Lagrangian data**: it uses trajectories to reconstruct the medium-mediated structure (response kernel and/or
correlated driving) that links entities together.

Where the CDV estimator fits
----------------------------
Operationally, CDV estimates a spatial map of correlated dynamics by computing **event-aligned conditional averages
or cross-correlations** of displacements. Typical estimators take the form of conditional expectations or
cross-covariances:

.. math::

   \langle \Delta r_p \mid \Delta r_s \rangle
   \qquad \text{or} \qquad
   \langle \Delta r_p\,\Delta r_s\rangle,

evaluated as functions of spatial offset :math:`R` (and often lag time :math:`\tau`). The key statistical principle is
simple:

- Motion contributions that are **uncorrelated** with the chosen source/event average toward zero.
- The **reproducible, medium-mediated response** survives ensemble averaging and emerges as a coherent vector field.

This is why CDV can reveal underlying flow-like structure even when raw trajectories look dominated by stochasticity.

.. figure:: /_static/videos/FlowField_airwater_1um_time11min.gif
   :alt: CDV flow field visualization
   :align: center
   :width: 100%

   **Video:** Left panel shows trajectories of Brownian particles; center panel displays transferred displacement vectors; right panel illustrates how the flow field around individual particles emerges as the ensemble size increases over time.

What CDV computes
-----------------
- Local average displacement vectors over spatial bins and time windows.
- Optional drift removal or background subtraction.
- Diagnostics: vector fields, magnitude maps, and residuals.

Typical workflow
----------------
1. Load or build a ``TrajectorySet`` with positions and time/frame info.
2. Choose spatial binning (e.g., grid spacing) and temporal windowing.
3. Run CDV to compute displacement/velocity vectors.
...

   # result.field -> displacement/velocity vectors per bin
   # result.meta  -> settings, units, diagnostic stats

Validation and visualization
----------------------------
- Plot quiver/streamline maps of the inferred field.
- Check residuals or variance in each bin.
- Try multiple bin sizes/time windows to assess stability.

Next steps
----------
- Add a worked example notebook in ``examples/visualization`` for CDV.
- Document the concrete API signature once finalized (module/function names may differ).
