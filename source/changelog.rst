.. _Changelog:

What's new
##########

.. contents:: Table of Contents
   :depth: 2

v0.1.3 (April 2021)
-------------------

**Enhancements**
a. :py:func:`systole.plotly.plot_raw()`: add `ecg_method` parameter to control the ECG peak detection method used.
b. Download dataset directly from GitHub instead of copying the files at install.
c. Haromonisation of :py:func:`systole.plotting.plot_raw()` and :py:func:`systole.plotting.plot_raw()` (replace the `plot_hr()` function), and :py:func:`systole.plotly.plot_subspaces()` and :py:func:`systole.plotly.plot_subspaces()`.
d. The :py:class:`systole.recording.Oximeter()` class has been improved:
   * :py:func:`systole.recording.Oximeter.setup()` has an `nAttempts` argument so it will not run forever if no valid signal is recordedfor a given number of attempts (default is 100).
   * :py:func:`systole.recording.Oximeter.check()` has been updated and accept data format #7 from Xpods, allowing more flexibility.
   * :py:func:`systole.recording.Oximeter.save()` will now save additional channels and support `.txt` and `.npy` file extensions.
   * Create a :py:func:`systole.recording.Oximeter.reset()` method to avoid improper use of `__init__()`.
e. Add pre-commit hooks, flake8, black and isort CI tests.
f. Add type hints and CI testing with mypy.

v0.1.2 (September 2020)
-----------------------

 **New functions**

a. Add :py:func:`systole.utils.to_rr()`. for peaks or index vectors convertion to RR intervals
b. Add :py:func:`systole.recording.BrainVisionExG()`, a class to read physio recording from BrainVision ExG products via TCP/IP connection.
c. Add :py:func:`systole.recording.findOximeter()`, find the USB port where Nonin Oximeter is plugged by looping through the USB port and checking the input.
d. Add :py:func:`systole.detection.ecg_peaks()`. A wrapper around py-ecg-detectors for basic ECG peaks detection.

**Enhancements**
a. Improved documentation and examples.
b. Simplification of PPG example data import.
c. Improved interactive plotting functions.


v0.1.1 (June 2020)
------------------

**New functions**

a. Add the **plotly** sub-module, a set of Plotly functions comprising :py:func:`systole.plotly.plot_raw`, :py:func:`systole.plotly.plot_subspaces`, :py:func:`systole.plotly.plot_ectopic`, :py:func:`systole.plotly.plot_shortLong`, :py:func:`systole.plotly.plot_frequency`, :py:func:`systole.plotly.plot_nonlinear`, :py:func:`systole.plotly.plot_timedomain`.
b. Add :py:func:`systole.utils.simulate_rr()`, for random RR interval simulation with different kind of artefacts. Can also return peak vector.
c. The **correction** sub-module has been largely rewritten and now include :py:func:`systole.correction.correct_extra`, :py:func:`systole.correction.correct_missed`, :py:func:`systole.correction.interpolate_bads`, :py:func:`systole.correction.correct_rr`, :py:func:`systole.correction.correct_peaks`, :py:func:`systole.correction.correct_missed_peaks`, :py:func:`systole.correction.correct_extra_peaks`. These function can correct artefacts either using peaks addition/removal or by interpolation of the RR time series.

**Enhancements**

a. The **detection** sub-module has been improved. It is now about 10x faster and returns more information. The main function has been renamed to :py:func:`systole.detection.rr_artefacts`.

**Bugfixes**

a. :py:func:`systole.correction.interpolate_clipping`: add exception in case of clipping artefacts at the edge of the signal segment. This can cause cash during recording. The default behavior is now to decrement the last/first item in case of threshold value. The threshold can be changed manually. This procedure can result in slightly inaccurate interpolation, using a longer recording should always be preferred when possible.
b. The PPG signal simulator used for testing can now run infinitely.

**Contributors**

* `Jan C. Brammer <jan.c.brammer@gmail.com>`_


v0.1.0 (January 2020)
---------------------

Initial release.

**Detection**

a. oxi_peaks()
b. hr_subspaces()
c. interpolate_clipping()
d. rr_outliers()


**HRV**

a. nnX()
b. pnX()
c. rmssd()
d. time_domain()
e. frequency_domain()
f. nonlinear()


**Plotting**

a. plot_hr()
b. plot_events()
c. plot_oximeter()
d. plot_subspaces()
e. plot_psd()
f. circular()
g. plot_circular()


**Recording**

a. Oximeter()


**Report**

a. report_oxi()


**Utils**

a. norm_triggers()
b. time_shift()
c. heart_rate()
d. to_angles()
e. to_epochs()

v0.0.1 (January 2020)
---------------------

Alpha release.
