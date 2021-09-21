.. _api_ref:

.. currentmodule:: systole

Functions
=========

.. contents:: Table of Contents
   :depth: 3

Correction
----------

.. currentmodule:: systole.correction

.. _correction:

.. autosummary::
   :toctree: generated/correction

    correct_extra
    correct_missed
    interpolate_bads
    correct_rr
    correct_peaks
    correct_missed_peaks

Detection
---------

.. currentmodule:: systole.detection

.. _detection:

.. autosummary::
   :toctree: generated/detection

    ppg_peaks
    ecg_peaks
    rr_artefacts
    interpolate_clipping

Heart Rate Variability
----------------------

.. currentmodule:: systole.hrv

.. _hrv:

.. autosummary::
   :toctree: generated/hrv

    nnX
    pnnX
    rmssd
    time_domain
    frequency_domain
    nonlinear

Plots
-----

.. currentmodule:: systole.plots

.. _plots:

.. autosummary::
   :toctree: generated/plots

    plot_circular
    plot_ectopic
    plot_events
    plot_evoked
    plot_frequency
    plot_pointcare
    plot_raw
    plot_rr
    plot_shortlong
    plot_subspaces

Recording
---------

.. currentmodule:: systole.recording

.. _recording:

.. autosummary::
   :toctree: generated/recording

    Oximeter
    BrainVisionExG
    findOximeter

Utils
-----

.. currentmodule:: systole.utils

.. _utils:

.. autosummary::
   :toctree: generated/utils

    norm_triggers
    time_shift
    heart_rate
    to_angles
    to_epochs
    simulate_rr
    to_neighbour
    input_conversion
