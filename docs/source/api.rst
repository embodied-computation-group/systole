.. _api_ref:

.. currentmodule:: systole

Functions
=========

.. contents:: Table of Contents
   :depth: 2

Correction
----------

.. currentmodule:: systole.correction

.. _correction:

.. autosummary::
   :toctree: generated/correction

    correct_extra_rr
    correct_missed_rr
    interpolate_rr
    correct_rr
    correct_peaks
    correct_missed_peaks
    correct_ectopic_peaks

Detection
---------

.. currentmodule:: systole.detection

.. _detection:

.. autosummary::
   :toctree: generated/detection

    ppg_peaks
    ecg_peaks
    rsp_peaks
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
    psd
    frequency_domain
    nonlinear_domain
    poincare
    recurrence
    recurrence_matrix
    all_domain

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
    plot_poincare
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

Reports
-------

.. currentmodule:: systole.reports

.. _reports:

.. autosummary::
   :toctree: generated/reports

    time_table
    frequency_table
    nonlinear_table

Viewer
-------

.. currentmodule:: systole.viewer

.. _reports:

.. autosummary::
   :toctree: generated/viewer

    Viewer
    Editor

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
    nan_cleaning
    find_clipping
    get_valid_segments
    norm_bad_segments