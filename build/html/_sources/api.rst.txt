.. _api_ref:

.. currentmodule:: systole

Functions
=========

.. contents:: Table of Contents
   :depth: 3

Detection
---------

.. _detection:

.. autosummary::
 :toctree: generated/

  oxi_peaks
  artefact_correction
  missed_beat
  hrv_subspaces
  interpolate_clipping

Heart Rate Variability
----------------------

.. _hrv:

.. autosummary::
 :toctree: generated/

  nnX
  pnnX
  rmssd
  time_domain
  frequency_domain
  nonlinear

Plotting
--------

.. _plotting:

.. autosummary::
 :toctree: generated/

  plot_hr
  plot_events
  plot_oximeter
  plot_subspaces
  circular
  plot_circular

Recording
---------

.. _recording:

.. autosummary::
  :toctree: generated/

   recording.Oximeter

Utils
-----

.. _utils:

.. autosummary::
 :toctree: generated/

  norm_triggers
  time_shift
  heart_rate
  to_angles
  to_epochs
