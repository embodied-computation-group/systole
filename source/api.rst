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
    correct_extra_peaks

Detection
---------

.. currentmodule:: systole.detection

.. _detection:

.. autosummary::
   :toctree: generated/detection

    oxi_peaks
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

Plotly
--------

.. currentmodule:: systole.plotly

.. _plotly:

.. autosummary::
   :toctree: generated/plotly

    plot_raw
    plot_ectopic
    plot_shortLong
    plot_subspaces
    plot_frequency
    plot_nonlinear
    plot_timedomain

Plotting
--------

.. currentmodule:: systole.plotting

.. _plotting:

.. autosummary::
   :toctree: generated/plotting

    plot_raw
    plot_events
    plot_oximeter
    plot_subspaces
    plot_psd
    circular
    plot_circular

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
    to_rr
